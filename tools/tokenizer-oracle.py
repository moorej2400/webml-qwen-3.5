#!/usr/bin/env python3
"""Generate small, sanitized Qwen3.5 tokenizer fixtures with Transformers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import PreTrainedTokenizerFast


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer-json", required=True, type=Path)
    parser.add_argument("--tokenizer-config", required=True, type=Path)
    args = parser.parse_args()

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
    reverse_merges = {}
    for rank, merge in enumerate(artifact["model"]["merges"]):
        left, right = merge.split(" ", 1)
        result_id = vocabulary[left + right]
        reverse_merges.setdefault(
            result_id,
            {
                "rank": rank,
                "left": vocabulary[left],
                "right": vocabulary[right],
                "result": result_id,
            },
        )

    required_ids = set(range(256))
    for case in result["text"] + result["chat"]:
        required_ids.update(
            token_id for token_id in case["ids"] if token_id < len(vocabulary)
        )

    required_merges = {}

    def include_ancestry(token_id: int) -> None:
        merge = reverse_merges.get(token_id)
        if merge is None or token_id in required_merges:
            return
        required_merges[token_id] = merge
        include_ancestry(merge["left"])
        include_ancestry(merge["right"])

    for token_id in tuple(required_ids):
        include_ancestry(token_id)
    for merge in required_merges.values():
        required_ids.add(merge["left"])
        required_ids.add(merge["right"])
        required_ids.add(merge["result"])

    result["runtimeSlice"] = {
        "baseVocabSize": len(vocabulary),
        "tokenCount": len(vocabulary) + len(artifact["added_tokens"]),
        "tokens": [
            {"id": token_id, "value": id_to_token[token_id]}
            for token_id in sorted(required_ids)
        ],
        "merges": sorted(required_merges.values(), key=lambda item: item["rank"]),
        "addedTokens": artifact["added_tokens"],
    }

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
