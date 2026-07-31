#!/usr/bin/env python3
"""Generate small, sanitized Qwen3.5 tokenizer fixtures with Transformers."""

from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from pathlib import Path

from transformers import PreTrainedTokenizerFast


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer-json", required=True, type=Path)
    parser.add_argument("--tokenizer-config", required=True, type=Path)
    parser.add_argument("--oracle-output", type=Path)
    parser.add_argument("--slice-output", type=Path)
    args = parser.parse_args()
    if (args.oracle_output is None) != (args.slice_output is None):
        parser.error("--oracle-output and --slice-output must be used together")

    config = json.loads(args.tokenizer_config.read_text(encoding="utf-8"))
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(args.tokenizer_json),
        chat_template=config["chat_template"],
        eos_token=config["eos_token"],
        pad_token=config["pad_token"],
    )

    text_cases = [
        ("ascii", "Hello, tokenizer!"),
        ("unicode", "naïve café — 東京"),
        ("emoji", "Ship it 🚀🙂"),
        ("combining", "Cafe\u0301 vs café"),
        ("whitespace", "  alpha\tbeta\n\nomega  "),
        ("punctuation", "Wait... what?! (yes/no)"),
        ("code", "const x = (n: number) => n + 1;\n"),
        ("replacement-byte-path", "A\u0000B\ufffdC"),
    ]
    chat_cases = [
        (
            "user-generation",
            [{"role": "user", "content": "Hello"}],
            True,
            True,
        ),
        (
            "system-user-generation-no-thinking",
            [
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "Give one fact."},
            ],
            True,
            False,
        ),
        (
            "multi-message",
            [
                {"role": "system", "content": "Use plain text."},
                {"role": "user", "content": "First"},
                {"role": "assistant", "content": "One"},
                {"role": "user", "content": "Second"},
            ],
            True,
            True,
        ),
        (
            "typed-image-marker",
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe: "},
                        {"type": "image"},
                    ],
                }
            ],
            True,
            False,
        ),
    ]

    result = {
        "source": "Qwen/Qwen3.5-4B",
        "revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        "tokenizerSha256": (
            "5f9e4d4901a92b997e463c1f46055088"
            "b6cca5ca61a6522d1b9f64c4bb81cb42"
        ),
        "text": [
            {
                "name": name,
                "input": value,
                "ids": tokenizer.encode(value, add_special_tokens=False),
            }
            for name, value in text_cases
        ],
        "chat": [],
    }
    for name, messages, add_generation_prompt, enable_thinking in chat_cases:
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
        )
        ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
        )
        result["chat"].append(
            {
                "name": name,
                "messages": messages,
                "addGenerationPrompt": add_generation_prompt,
                "enableThinking": enable_thinking,
                "rendered": rendered,
                "ids": ids,
            }
        )

    artifact = json.loads(args.tokenizer_json.read_text(encoding="utf-8"))
    vocabulary = artifact["model"]["vocab"]
    id_to_token = {token_id: token for token, token_id in vocabulary.items()}
    merge_lookup = {}
    for rank, merge in enumerate(artifact["model"]["merges"]):
        left, right = merge.split(" ", 1)
        result_id = vocabulary[left + right]
        merge_lookup[(vocabulary[left], vocabulary[right])] = {
            "rank": rank,
            "left": vocabulary[left],
            "right": vocabulary[right],
            "result": result_id,
        }

    # The runtime recognizes added-token literals before normalization and
    # pre-tokenization. Longest-first matching agrees with the production
    # tokenizer because the pinned added-token strings have no equal literals.
    added_by_first = {}
    for added_token in artifact["added_tokens"]:
        added_by_first.setdefault(added_token["content"][0], []).append(
            added_token
        )
    for candidates in added_by_first.values():
        candidates.sort(key=lambda item: -len(item["content"]))

    def split_added_tokens(value: str):
        segments = []
        text_start = 0
        offset = 0
        while offset < len(value):
            match = next(
                (
                    candidate
                    for candidate in added_by_first.get(value[offset], [])
                    if value.startswith(candidate["content"], offset)
                ),
                None,
            )
            if match is None:
                offset += 1
                continue
            if offset > text_start:
                segments.append(("text", value[text_start:offset]))
            segments.append(("added", match["id"]))
            offset += len(match["content"])
            text_start = offset
        if text_start < len(value) or not segments:
            segments.append(("text", value[text_start:]))
        return segments

    required_ids = set(range(256))
    required_merges = {}
    closure_traces = []
    proof_counts = {
        "stateCount": 0,
        "adjacentPairObservations": 0,
        "mergeCandidateObservations": 0,
        "losingCandidateObservations": 0,
        "staleCandidateOpportunities": 0,
    }

    def trace_piece(
        case_kind: str,
        case_name: str,
        piece_index: int,
        piece: str,
    ):
        symbols = [vocabulary[character] for character in piece]
        initial = list(symbols)
        required_ids.update(symbols)
        states = []
        while True:
            candidates = []
            proof_counts["stateCount"] += 1
            proof_counts["adjacentPairObservations"] += max(
                0, len(symbols) - 1
            )
            for left_index in range(len(symbols) - 1):
                merge = merge_lookup.get(
                    (symbols[left_index], symbols[left_index + 1])
                )
                if merge is None:
                    continue
                candidate = [
                    left_index,
                    merge["left"],
                    merge["right"],
                    merge["rank"],
                    merge["result"],
                ]
                candidates.append(candidate)
                required_merges[merge["rank"]] = merge
                required_ids.update(
                    [merge["left"], merge["right"], merge["result"]]
                )
            states.append(candidates)
            proof_counts["mergeCandidateObservations"] += len(candidates)
            if not candidates:
                break
            selected = min(candidates, key=lambda item: (item[3], item[0]))
            proof_counts["losingCandidateObservations"] += len(candidates) - 1
            proof_counts["staleCandidateOpportunities"] += sum(
                candidate is not selected
                and abs(candidate[0] - selected[0]) <= 1
                for candidate in candidates
            )
            symbols[selected[0] : selected[0] + 2] = [selected[4]]
        closure_traces.append(
            [case_kind, case_name, piece_index, initial, states, symbols]
        )
        return symbols

    def trace_case(case_kind: str, case):
        encoded = []
        piece_index = 0
        value = case["input"] if case_kind == "text" else case["rendered"]
        for segment_kind, segment_value in split_added_tokens(value):
            if segment_kind == "added":
                encoded.append(segment_value)
                required_ids.add(segment_value)
                continue
            normalized = unicodedata.normalize("NFC", segment_value)
            for piece, _ in tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(
                normalized
            ):
                encoded.extend(
                    trace_piece(
                        case_kind,
                        case["name"],
                        piece_index,
                        piece,
                    )
                )
                piece_index += 1
        if encoded != case["ids"]:
            raise RuntimeError(
                f"closure simulation differs from oracle for {case['name']}"
            )

    for case in result["text"]:
        trace_case("text", case)
    for case in result["chat"]:
        trace_case("chat", case)

    for added_token in artifact["added_tokens"]:
        required_ids.add(added_token["id"])
    runtime_merges = [
        required_merges[rank] for rank in sorted(required_merges)
    ]
    canonical_traces = json.dumps(
        closure_traces,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    result["closureProof"] = {
        "version": 1,
        "algorithm": "all-adjacent-candidates-leftmost-min-rank-v1",
        "sourceMergeCount": len(artifact["model"]["merges"]),
        "oracleCaseCount": len(result["text"]) + len(result["chat"]),
        "pieceCount": len(closure_traces),
        **proof_counts,
        "uniqueCandidateMergeCount": len(runtime_merges),
        "traceSha256": hashlib.sha256(canonical_traces).hexdigest(),
    }
    result["closureTraces"] = closure_traces
    result["runtimeSlice"] = {
        "baseVocabSize": len(vocabulary),
        "tokenCount": len(vocabulary) + len(artifact["added_tokens"]),
        "tokens": [
            {"id": token_id, "value": id_to_token[token_id]}
            for token_id in sorted(required_ids)
            if token_id < len(vocabulary)
        ],
        "merges": runtime_merges,
        "addedTokens": artifact["added_tokens"],
    }

    if args.oracle_output is None:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    oracle = {
        key: result[key]
        for key in ("source", "revision", "tokenizerSha256", "text", "chat")
    }
    runtime_slice = {
        key: result[key]
        for key in (
            "source",
            "revision",
            "tokenizerSha256",
            "closureProof",
            "closureTraces",
            "runtimeSlice",
        )
    }
    args.oracle_output.write_text(
        json.dumps(oracle, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.slice_output.write_text(
        json.dumps(runtime_slice, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
