import type {
  KernelDefinition,
  KernelRegistry,
} from "./kernel-registry.js";

const FULL_ATTENTION_PREPARE_WGSL = /* wgsl */ `
struct Params {
  position: u32,
  capacity: u32,
  temporal_position: u32,
  height_position: u32,
  width_position: u32,
  pad0: u32,
  pad1: u32,
  pad2: u32,
}

@group(0) @binding(0) var<storage, read> query_gate_values: array<f32>;
@group(0) @binding(1) var<storage, read> key_values: array<f32>;
@group(0) @binding(2) var<storage, read> value_values: array<f32>;
@group(0) @binding(3) var<storage, read> query_norm_weights: array<f32>;
@group(0) @binding(4) var<storage, read> key_norm_weights: array<f32>;
@group(0) @binding(5) var<storage, read_write> prepared_query_gate: array<f32>;
@group(0) @binding(6) var<storage, read_write> packed_key_cache: array<u32>;
@group(0) @binding(7) var<storage, read_write> packed_value_cache: array<u32>;
@group(0) @binding(8) var<uniform> params: Params;

fn frequency_owner(frequency: u32) -> u32 {
  let coordinate = frequency % 3u;
  if (coordinate == 1u && frequency < 33u) { return 1u; }
  if (coordinate == 2u && frequency < 30u) { return 2u; }
  return 0u;
}

fn coordinate_position(owner: u32) -> u32 {
  if (owner == 1u) { return params.height_position; }
  if (owner == 2u) { return params.width_position; }
  return params.temporal_position;
}

fn rotate(
  value: f32,
  partner: f32,
  lane: u32,
) -> f32 {
  if (lane >= 64u) { return value; }
  let frequency = lane % 32u;
  let angle =
    f32(coordinate_position(frequency_owner(frequency))) /
    pow(10000000.0f, f32(frequency * 2u) / 64.0f);
  let tau = 6.28318548f;
  let reduced_angle = angle - floor(angle / tau) * tau;
  let cosine = cos(reduced_angle);
  let sine = sin(reduced_angle);
  return select(
    value * cosine + partner * sine,
    value * cosine - partner * sine,
    lane < 32u,
  );
}

fn normalized_query(query_head: u32, lane: u32, inverse_rms: f32) -> f32 {
  let query_base = query_head * 512u;
  return
    query_gate_values[query_base + lane] *
    inverse_rms * query_norm_weights[lane];
}

fn rotated_query(query_head: u32, lane: u32, inverse_rms: f32) -> f32 {
  if (lane >= 64u) {
    return normalized_query(query_head, lane, inverse_rms);
  }
  let partner_lane = select(lane - 32u, lane + 32u, lane < 32u);
  return rotate(
    normalized_query(query_head, lane, inverse_rms),
    normalized_query(query_head, partner_lane, inverse_rms),
    lane,
  );
}

fn normalized_key(kv_head: u32, lane: u32, inverse_rms: f32) -> f32 {
  return
    key_values[kv_head * 256u + lane] *
    inverse_rms * key_norm_weights[lane];
}

fn rotated_key(kv_head: u32, lane: u32, inverse_rms: f32) -> f32 {
  if (lane >= 64u) {
    return normalized_key(kv_head, lane, inverse_rms);
  }
  let partner_lane = select(lane - 32u, lane + 32u, lane < 32u);
  return rotate(
    normalized_key(kv_head, lane, inverse_rms),
    normalized_key(kv_head, partner_lane, inverse_rms),
    lane,
  );
}

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) invocation: vec3<u32>) {
  let query_head = invocation.x;
  if (
    query_head >= 16u ||
    params.capacity == 0u ||
    params.capacity > 16384u ||
    params.position >= params.capacity
  ) { return; }

  let query_base = query_head * 512u;
  var query_sum = 0.0f;
  for (var lane = 0u; lane < 256u; lane += 1u) {
    let raw = query_gate_values[query_base + lane];
    query_sum += raw * raw;
  }
  let query_inverse_rms = inverseSqrt(query_sum / 256.0f + 0.000001f);
  for (var lane = 0u; lane < 256u; lane += 1u) {
    prepared_query_gate[query_base + lane] =
      rotated_query(query_head, lane, query_inverse_rms);
    prepared_query_gate[query_base + 256u + lane] =
      query_gate_values[query_base + 256u + lane];
  }

  if (query_head < 4u) {
    let kv_head = query_head;
    var key_sum = 0.0f;
    for (var lane = 0u; lane < 256u; lane += 1u) {
      let raw = key_values[kv_head * 256u + lane];
      key_sum += raw * raw;
    }
    let key_inverse_rms = inverseSqrt(key_sum / 256.0f + 0.000001f);
    let cache_base = params.position * 512u + kv_head * 128u;
    for (var lane = 0u; lane < 256u; lane += 2u) {
      let word = cache_base + lane / 2u;
      packed_key_cache[word] = pack2x16float(vec2<f32>(
        rotated_key(kv_head, lane, key_inverse_rms),
        rotated_key(kv_head, lane + 1u, key_inverse_rms),
      ));
      let value_base = kv_head * 256u;
      packed_value_cache[word] = pack2x16float(vec2<f32>(
        value_values[value_base + lane],
        value_values[value_base + lane + 1u],
      ));
    }
  }
}`;

const FULL_ATTENTION_ONLINE_WGSL = /* wgsl */ `
struct Params {
  token_count: u32,
  page_index: u32,
  page_count: u32,
  pad0: u32,
}

@group(0) @binding(0) var<storage, read> prepared_query_gate: array<f32>;
@group(0) @binding(1) var<storage, read> packed_key_values: array<u32>;
@group(0) @binding(2) var<storage, read> packed_value_values: array<u32>;
@group(0) @binding(3) var<storage, read_write> output_values: array<f32>;
@group(0) @binding(4) var<storage, read_write> online_state: array<f32>;
@group(0) @binding(5) var<uniform> params: Params;

fn sigmoid(value: f32) -> f32 {
  if (value >= 0.0f) {
    return 1.0f / (1.0f + exp(-value));
  }
  let exponential = exp(value);
  return exponential / (1.0f + exponential);
}

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) invocation: vec3<u32>) {
  let query_head = invocation.x;
  if (
    query_head >= 16u ||
    params.token_count == 0u ||
    params.token_count > 16384u ||
    params.page_count == 0u ||
    params.page_index >= params.page_count
  ) { return; }
  let query_base = query_head * 512u;
  let kv_head = query_head / 4u;
  var accumulator: array<f32, 256>;
  let first_page = params.page_index == 0u;
  let final_page = params.page_index + 1u == params.page_count;
  for (var lane = 0u; lane < 256u; lane += 1u) {
    let output_index = query_head * 256u + lane;
    accumulator[lane] = select(output_values[output_index], 0.0f, first_page);
  }
  let state_base = query_head * 2u;
  var running_maximum = select(
    online_state[state_base],
    -3.402823466e+38f,
    first_page,
  );
  var running_denominator = select(
    online_state[state_base + 1u],
    0.0f,
    first_page,
  );
  for (var token = 0u; token < params.token_count; token += 1u) {
    var dot = 0.0f;
    for (var lane = 0u; lane < 256u; lane += 1u) {
      let scalar_index = ((token * 4u + kv_head) * 256u) + lane;
      let packed = unpack2x16float(packed_key_values[scalar_index / 2u]);
      let key_value = select(packed.x, packed.y, (scalar_index & 1u) == 1u);
      dot += prepared_query_gate[query_base + lane] * key_value;
    }
    let logit = dot * 0.0625f;
    let next_maximum = max(running_maximum, logit);
    var old_scale = 0.0f;
    if (running_denominator != 0.0f) {
      old_scale = exp(running_maximum - next_maximum);
    }
    let token_scale = exp(logit - next_maximum);
    running_denominator = running_denominator * old_scale + token_scale;
    for (var lane = 0u; lane < 256u; lane += 1u) {
      let scalar_index = ((token * 4u + kv_head) * 256u) + lane;
      let packed = unpack2x16float(packed_value_values[scalar_index / 2u]);
      let value = select(packed.x, packed.y, (scalar_index & 1u) == 1u);
      accumulator[lane] = accumulator[lane] * old_scale + token_scale * value;
    }
    running_maximum = next_maximum;
  }
  online_state[state_base] = running_maximum;
  online_state[state_base + 1u] = running_denominator;
  for (var lane = 0u; lane < 256u; lane += 1u) {
    let output_index = query_head * 256u + lane;
    output_values[output_index] = select(
      accumulator[lane],
      (accumulator[lane] / running_denominator) *
        sigmoid(prepared_query_gate[query_base + 256u + lane]),
      final_page,
    );
  }
}`;

const DELTANET_CONV_WGSL = /* wgsl */ `
@group(0) @binding(0) var<storage, read> raw_qkv: array<f32>;
@group(0) @binding(1) var<storage, read> weight_values: array<f32>;
@group(0) @binding(2) var<storage, read_write> state_values: array<f32>;
@group(0) @binding(3) var<storage, read_write> output_values: array<f32>;

fn silu(value: f32) -> f32 {
  return value / (1.0f + exp(-value));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) invocation: vec3<u32>) {
  let channel = invocation.x;
  if (channel >= 8192u) { return; }
  let base = channel * 4u;
  state_values[base] = state_values[base + 1u];
  state_values[base + 1u] = state_values[base + 2u];
  state_values[base + 2u] = state_values[base + 3u];
  state_values[base + 3u] = raw_qkv[channel];
  var sum = 0.0f;
  for (var tap = 0u; tap < 4u; tap += 1u) {
    sum += state_values[base + tap] * weight_values[base + tap];
  }
  output_values[channel] = silu(sum);
}`;

const DELTANET_PARAMETERS_WGSL = /* wgsl */ `
@group(0) @binding(0) var<storage, read> beta_input_values: array<f32>;
@group(0) @binding(1) var<storage, read> a_values: array<f32>;
@group(0) @binding(2) var<storage, read> dt_values: array<f32>;
@group(0) @binding(3) var<storage, read> ssm_a_values: array<f32>;
@group(0) @binding(4) var<storage, read_write> beta_values: array<f32>;
@group(0) @binding(5) var<storage, read_write> decay_values: array<f32>;

fn sigmoid(value: f32) -> f32 {
  if (value >= 0.0f) {
    return 1.0f / (1.0f + exp(-value));
  }
  let exponential = exp(value);
  return exponential / (1.0f + exponential);
}

fn softplus(value: f32) -> f32 {
  return max(value, 0.0f) + log(1.0f + exp(-abs(value)));
}

@compute @workgroup_size(32)
fn main(@builtin(global_invocation_id) invocation: vec3<u32>) {
  let head = invocation.x;
  if (head >= 32u) { return; }
  beta_values[head] = sigmoid(beta_input_values[head]);
  let gate = ssm_a_values[head] * softplus(a_values[head] + dt_values[head]);
  decay_values[head] = exp(gate);
}`;

const DELTANET_RECURRENT_WGSL = /* wgsl */ `
@group(0) @binding(0) var<storage, read> convolved_qkv: array<f32>;
@group(0) @binding(1) var<storage, read> beta_values: array<f32>;
@group(0) @binding(2) var<storage, read> decay_values: array<f32>;
@group(0) @binding(3) var<storage, read_write> state_values: array<f32>;
@group(0) @binding(4) var<storage, read_write> output_values: array<f32>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) invocation: vec3<u32>) {
  let value_head = invocation.x;
  if (value_head >= 32u) { return; }
  let qk_head = value_head % 16u;
  var query_sum = 0.0f;
  var key_sum = 0.0f;
  for (var lane = 0u; lane < 128u; lane += 1u) {
    let query_raw = convolved_qkv[qk_head * 128u + lane];
    let key_raw = convolved_qkv[2048u + qk_head * 128u + lane];
    query_sum += query_raw * query_raw;
    key_sum += key_raw * key_raw;
  }
  let query_scale =
    inverseSqrt(query_sum + 0.000001f) * inverseSqrt(128.0f);
  let key_scale = inverseSqrt(key_sum + 0.000001f);

  for (var value_lane = 0u; value_lane < 128u; value_lane += 1u) {
    var memory = 0.0f;
    for (var key_lane = 0u; key_lane < 128u; key_lane += 1u) {
      let state_index =
        value_head * 16384u + key_lane * 128u + value_lane;
      let decayed = state_values[state_index] * decay_values[value_head];
      state_values[state_index] = decayed;
      let key_value =
        convolved_qkv[2048u + qk_head * 128u + key_lane] * key_scale;
      memory += key_value * decayed;
    }
    let target_value =
      convolved_qkv[4096u + value_head * 128u + value_lane];
    let delta = beta_values[value_head] * (target_value - memory);
    var head_output = 0.0f;
    for (var key_lane = 0u; key_lane < 128u; key_lane += 1u) {
      let state_index =
        value_head * 16384u + key_lane * 128u + value_lane;
      let key_value =
        convolved_qkv[2048u + qk_head * 128u + key_lane] * key_scale;
      let updated = state_values[state_index] + key_value * delta;
      state_values[state_index] = updated;
      let query_value =
        convolved_qkv[qk_head * 128u + key_lane] * query_scale;
      head_output += query_value * updated;
    }
    output_values[value_head * 128u + value_lane] = head_output;
  }
}`;

const DELTANET_GATED_NORM_WGSL = /* wgsl */ `
@group(0) @binding(0) var<storage, read> recurrent_values: array<f32>;
@group(0) @binding(1) var<storage, read> z_values: array<f32>;
@group(0) @binding(2) var<storage, read> norm_weight_values: array<f32>;
@group(0) @binding(3) var<storage, read_write> output_values: array<f32>;

fn silu(value: f32) -> f32 {
  return value / (1.0f + exp(-value));
}

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) invocation: vec3<u32>) {
  let value_head = invocation.x;
  if (value_head >= 32u) { return; }
  let base = value_head * 128u;
  var sum = 0.0f;
  for (var lane = 0u; lane < 128u; lane += 1u) {
    let value = recurrent_values[base + lane];
    sum += value * value;
  }
  let inverse_rms = inverseSqrt(sum / 128.0f + 0.000001f);
  for (var lane = 0u; lane < 128u; lane += 1u) {
    let index = base + lane;
    output_values[index] =
      recurrent_values[index] * inverse_rms *
      norm_weight_values[lane] * silu(z_values[index]);
  }
}`;

function definition(
  operation: string,
  layout: string,
  source: string,
): KernelDefinition {
  return Object.freeze({
    id: `qwen35-${operation}-portable-f32`,
    key: Object.freeze({
      operation,
      layout,
      phase: "decode",
      profile: "portable-f32",
    }),
    source,
  });
}

/**
 * These kernels are fixed Qwen3.5 4B decode programs. Keeping the five stages
 * explicit prevents a generic tensor abstraction from hiding recurrent order
 * or allocating an attention matrix.
 */
export const QWEN35_HYBRID_KERNELS: readonly KernelDefinition[] =
  Object.freeze([
    definition(
      "full-attention-prepare",
      "fp16-kv-pages",
      FULL_ATTENTION_PREPARE_WGSL,
    ),
    definition(
      "full-attention-online",
      "fp16-kv-pages",
      FULL_ATTENTION_ONLINE_WGSL,
    ),
    definition(
      "deltanet-conv",
      "fp32-recurrent-state",
      DELTANET_CONV_WGSL,
    ),
    definition(
      "deltanet-parameters",
      "fp32-recurrent-state",
      DELTANET_PARAMETERS_WGSL,
    ),
    definition(
      "deltanet-recurrent",
      "fp32-recurrent-state",
      DELTANET_RECURRENT_WGSL,
    ),
    definition(
      "deltanet-gated-norm",
      "fp32-recurrent-state",
      DELTANET_GATED_NORM_WGSL,
    ),
  ]);

export function planQwen35FullAttentionKvWrite(
  position: number,
  capacity: number,
): {
  readonly wordOffset: number;
  readonly wordLength: number;
  readonly endWord: number;
} {
  if (
    !Number.isSafeInteger(capacity) ||
    capacity < 1 ||
    capacity > 16_384
  ) {
    throw new Error("Full-attention K/V capacity is invalid");
  }
  if (
    !Number.isSafeInteger(position) ||
    position < 0 ||
    position >= capacity
  ) {
    throw new Error("Full-attention K/V position is invalid");
  }
  const wordLength = 4 * 256 / 2;
  const wordOffset = position * wordLength;
  return Object.freeze({
    wordOffset,
    wordLength,
    endWord: wordOffset + wordLength,
  });
}

export type Qwen35HybridOperation =
  | "full-attention-prepare"
  | "full-attention-online"
  | "deltanet-conv"
  | "deltanet-parameters"
  | "deltanet-recurrent"
  | "deltanet-gated-norm";

export interface Qwen35HybridDispatchPlan {
  readonly operation: Qwen35HybridOperation;
  readonly workgroups: {
    readonly x: number;
    readonly y: 1;
    readonly z: 1;
  };
  readonly uniforms:
    | Readonly<{
        position: number;
        capacity: number;
        positions: readonly [number, number, number];
      }>
    | Readonly<{
        tokenCount: number;
        position: number;
        capacity: number;
      }>
    | null;
}

function requireU32(value: number | undefined, label: string): number {
  if (
    value === undefined ||
    !Number.isSafeInteger(value) ||
    value < 0 ||
    value > 0xffff_ffff
  ) {
    throw new Error(`${label} must fit u32`);
  }
  return value;
}

function requireContextAddress(input: {
  readonly position?: number;
  readonly capacity?: number;
}): { readonly position: number; readonly capacity: number } {
  const position = requireU32(input.position, "Hybrid position");
  const capacity = requireU32(input.capacity, "Hybrid capacity");
  if (capacity < 1 || capacity > 16_384) {
    throw new Error("Hybrid capacity must be from 1 through 16384");
  }
  if (position >= capacity) {
    throw new Error("Hybrid position must be below capacity");
  }
  return { position, capacity };
}

export function planQwen35HybridDispatch(input: {
  readonly operation: Qwen35HybridOperation;
  readonly maxComputeWorkgroupsPerDimension: number;
  readonly position?: number;
  readonly capacity?: number;
  readonly tokenCount?: number;
  readonly positions?: readonly [number, number, number];
}): Qwen35HybridDispatchPlan {
  const workgroupLimit = requireU32(
    input.maxComputeWorkgroupsPerDimension,
    "Hybrid workgroup limit",
  );
  if (workgroupLimit < 1) {
    throw new Error("Hybrid workgroup limit must be positive");
  }
  const workgroupCounts: Readonly<Record<Qwen35HybridOperation, number>> = {
    "full-attention-prepare": 16,
    "full-attention-online": 16,
    "deltanet-conv": 128,
    "deltanet-parameters": 1,
    "deltanet-recurrent": 32,
    "deltanet-gated-norm": 32,
  };
  const workgroupCount = workgroupCounts[input.operation];
  if (workgroupCount === undefined) {
    throw new Error("Unsupported Qwen3.5 hybrid operation");
  }
  if (workgroupCount > workgroupLimit) {
    throw new Error("Hybrid dispatch exceeds the device workgroup limit");
  }

  let uniforms: Qwen35HybridDispatchPlan["uniforms"] = null;
  if (input.operation === "full-attention-prepare") {
    const address = requireContextAddress(input);
    if (
      input.positions === undefined ||
      input.positions.some(
        (position) =>
          !Number.isSafeInteger(position) ||
          position < 0 ||
          position > 0xffff_ffff,
      )
    ) {
      throw new Error("Hybrid M-RoPE positions must fit u32");
    }
    uniforms = Object.freeze({
      ...address,
      positions: Object.freeze([...input.positions]) as readonly [
        number,
        number,
        number,
      ],
    });
  } else if (input.operation === "full-attention-online") {
    const address = requireContextAddress(input);
    const tokenCount = requireU32(
      input.tokenCount,
      "Hybrid attention token count",
    );
    if (tokenCount < 1 || tokenCount > address.position + 1) {
      throw new Error(
        "Hybrid attention token count exceeds the published K/V position",
      );
    }
    uniforms = Object.freeze({ tokenCount, ...address });
  }

  return Object.freeze({
    operation: input.operation,
    workgroups: Object.freeze({ x: workgroupCount, y: 1, z: 1 }),
    uniforms,
  });
}

export function registerQwen35HybridKernels(
  registry: Pick<KernelRegistry, "register">,
): void {
  for (const kernel of QWEN35_HYBRID_KERNELS) {
    registry.register(kernel);
  }
}
