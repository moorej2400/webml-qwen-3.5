/**
 * Development-only Qwen3.5 vision math oracle.
 *
 * This module mirrors pinned PyTorch semantics on small tensors so WebGPU
 * kernels can be checked without placing a generic tensor runtime in the app.
 */

export const QWEN35_VISION_CPU_ORACLE_REFERENCE = Object.freeze({
  transformersRevision: "3717b9cda226af7377da1944f395af0eac6e2b51",
  pytorchVersion: "2.8.0",
  pytorchDevice: "cpu",
  pytorchDtype: "float32",
  arithmetic: Object.freeze({
    bf16Decode: "upper-16-bits-to-f32",
    linearReduction: "sequential-f32-multiply-add",
    layerNormReduction: "origin-shifted-two-pass-f32",
    ropeFrequency: "pytorch-f32-inverse-frequency",
  }),
  // CPU matrix kernels may reduce in a different order from this scalar oracle.
  pytorchFixtureTolerance: Object.freeze({
    absoluteTolerance: 0.0005,
    relativeTolerance: 0.0000001,
  }),
});

const LAYER_NORM_EPSILON = 0.000001;
const MAX_ATTENTION_SCORE_EVALUATIONS = 16_777_216;

function fail(message: string): never {
  throw new Error(`Qwen3.5 vision CPU oracle: ${message}`);
}

function positiveInteger(value: number, name: string): void {
  if (!Number.isSafeInteger(value) || value < 1) fail(`${name} must be a positive integer`);
}

function finiteArray(values: ArrayLike<number>, name: string): void {
  for (let index = 0; index < values.length; index += 1) {
    if (!Number.isFinite(values[index]!)) fail(`${name} must contain only finite values`);
  }
}

function expectedLength(name: string, values: ArrayLike<number>, length: number): void {
  if (values.length !== length) fail(`${name} shape is invalid`);
}

function checkedProduct(name: string, ...values: readonly number[]): number {
  let product = 1;
  for (const value of values) {
    positiveInteger(value, name);
    product *= value;
    if (!Number.isSafeInteger(product)) fail(`${name} is too large`);
  }
  return product;
}

function f32(value: number): number {
  if (!Number.isFinite(value)) fail("calculation produced a non-finite value");
  return Math.fround(value);
}

function decodeBfloat16Word(word: number): number {
  const storage = new ArrayBuffer(4);
  const view = new DataView(storage);
  view.setUint32(0, word << 16, true);
  return view.getFloat32(0, true);
}

/** Decodes stored BF16 words without rounding them through an intermediate FP16 value. */
export function decodeBfloat16(words: Uint16Array): Float32Array {
  const output = new Float32Array(words.length);
  for (let index = 0; index < words.length; index += 1) {
    output[index] = decodeBfloat16Word(words[index]!);
  }
  return output;
}

function linearBf16(options: {
  readonly input: Float32Array;
  readonly tokenCount: number;
  readonly inputWidth: number;
  readonly outputWidth: number;
  readonly weight: Uint16Array;
  readonly bias: Float32Array;
}): Float32Array {
  const { input, tokenCount, inputWidth, outputWidth, weight, bias } = options;
  expectedLength("linear input", input, checkedProduct("linear input", tokenCount, inputWidth));
  expectedLength("linear BF16 weight", weight, checkedProduct("linear BF16 weight", inputWidth, outputWidth));
  expectedLength("linear bias", bias, outputWidth);
  finiteArray(input, "linear input");
  finiteArray(bias, "linear bias");
  const output = new Float32Array(checkedProduct("linear output", tokenCount, outputWidth));
  for (let token = 0; token < tokenCount; token += 1) {
    for (let outputChannel = 0; outputChannel < outputWidth; outputChannel += 1) {
      let sum = bias[outputChannel]!;
      for (let inputChannel = 0; inputChannel < inputWidth; inputChannel += 1) {
        // GGUF dimension zero is contiguous, so every output row owns inputWidth values.
        sum = f32(sum + f32(input[token * inputWidth + inputChannel]! * decodeBfloat16Word(weight[outputChannel * inputWidth + inputChannel]!)));
      }
      output[token * outputWidth + outputChannel] = f32(sum);
    }
  }
  return output;
}

function linearF32(input: {
  readonly input: Float32Array;
  readonly tokenCount: number;
  readonly inputWidth: number;
  readonly outputWidth: number;
  readonly weight: Float32Array;
  readonly bias: Float32Array;
}): Float32Array {
  const { input: values, tokenCount, inputWidth, outputWidth, weight, bias } = input;
  expectedLength("linear input", values, checkedProduct("linear input", tokenCount, inputWidth));
  expectedLength("linear FP32 weight", weight, checkedProduct("linear FP32 weight", inputWidth, outputWidth));
  expectedLength("linear bias", bias, outputWidth);
  finiteArray(values, "linear input");
  finiteArray(weight, "linear FP32 weight");
  finiteArray(bias, "linear bias");
  const output = new Float32Array(checkedProduct("linear output", tokenCount, outputWidth));
  for (let token = 0; token < tokenCount; token += 1) {
    for (let outputChannel = 0; outputChannel < outputWidth; outputChannel += 1) {
      let sum = bias[outputChannel]!;
      for (let inputChannel = 0; inputChannel < inputWidth; inputChannel += 1) {
        sum = f32(sum + f32(values[token * inputWidth + inputChannel]! * weight[outputChannel * inputWidth + inputChannel]!));
      }
      output[token * outputWidth + outputChannel] = f32(sum);
    }
  }
  return output;
}

/** Qwen vision Conv3D uses two temporal slices, then applies its bias exactly once. */
export function visionTemporalConv3dPatchEmbed(input: {
  readonly temporalSlices: readonly [Float32Array, Float32Array];
  readonly temporalWeights: readonly [Float32Array, Float32Array];
  readonly bias: Float32Array;
  readonly inputChannels: number;
  readonly patchHeight: number;
  readonly patchWidth: number;
  readonly outputWidth: number;
}): Float32Array {
  const { temporalSlices, temporalWeights, bias, inputChannels, patchHeight, patchWidth, outputWidth } = input;
  const inputWidth = checkedProduct("Conv3D input shape", inputChannels, patchHeight, patchWidth);
  positiveInteger(outputWidth, "Conv3D output width");
  expectedLength("Conv3D first temporal input", temporalSlices[0], inputWidth);
  expectedLength("Conv3D second temporal input", temporalSlices[1], inputWidth);
  expectedLength("Conv3D first temporal weight", temporalWeights[0], checkedProduct("Conv3D first temporal weight", inputWidth, outputWidth));
  expectedLength("Conv3D second temporal weight", temporalWeights[1], checkedProduct("Conv3D second temporal weight", inputWidth, outputWidth));
  expectedLength("Conv3D bias", bias, outputWidth);
  finiteArray(temporalSlices[0], "Conv3D first temporal input");
  finiteArray(temporalSlices[1], "Conv3D second temporal input");
  finiteArray(temporalWeights[0], "Conv3D first temporal weight");
  finiteArray(temporalWeights[1], "Conv3D second temporal weight");
  finiteArray(bias, "Conv3D bias");
  const output = new Float32Array(outputWidth);
  for (let outputChannel = 0; outputChannel < outputWidth; outputChannel += 1) {
    let sum = 0;
    for (let temporal = 0; temporal < 2; temporal += 1) {
      for (let inputChannel = 0; inputChannel < inputWidth; inputChannel += 1) {
        sum = f32(sum + f32(
          temporalSlices[temporal]![inputChannel]! * temporalWeights[temporal]![outputChannel * inputWidth + inputChannel]!,
        ));
      }
    }
    output[outputChannel] = f32(sum + bias[outputChannel]!);
  }
  return output;
}

/** Produces the Qwen vision block-major spatial sequence used by positions and merger input. */
export function visionMergeBlockCoordinates(input: {
  readonly gridHeight: number;
  readonly gridWidth: number;
  readonly mergeSize: number;
}): Int32Array {
  const { gridHeight, gridWidth, mergeSize } = input;
  positiveInteger(gridHeight, "grid height");
  positiveInteger(gridWidth, "grid width");
  positiveInteger(mergeSize, "merge size");
  if (gridHeight % mergeSize !== 0 || gridWidth % mergeSize !== 0) fail("grid shape must divide merge size");
  const coordinates = new Int32Array(checkedProduct("coordinate count", gridHeight, gridWidth, 2));
  let cursor = 0;
  for (let blockY = 0; blockY < gridHeight; blockY += mergeSize) {
    for (let blockX = 0; blockX < gridWidth; blockX += mergeSize) {
      for (let innerY = 0; innerY < mergeSize; innerY += 1) {
        for (let innerX = 0; innerX < mergeSize; innerX += 1) {
          coordinates[cursor] = blockY + innerY;
          coordinates[cursor + 1] = blockX + innerX;
          cursor += 2;
        }
      }
    }
  }
  return coordinates;
}

function bilinearCoordinate(index: number, destinationSize: number, sourceSize: number): number {
  return destinationSize === 1 ? 0 : (index * (sourceSize - 1)) / (destinationSize - 1);
}

/** Interpolates Qwen's learned 48x48 table with PyTorch align_corners=true semantics. */
export function visionInterpolatedPositionEmbedding(input: {
  readonly table: Float32Array;
  readonly tableHeight: number;
  readonly tableWidth: number;
  readonly hiddenSize: number;
  readonly gridHeight: number;
  readonly gridWidth: number;
  readonly mergeSize: number;
}): Float32Array {
  const { table, tableHeight, tableWidth, hiddenSize, gridHeight, gridWidth, mergeSize } = input;
  positiveInteger(tableHeight, "position table height");
  positiveInteger(tableWidth, "position table width");
  positiveInteger(hiddenSize, "position hidden size");
  expectedLength("position table", table, checkedProduct("position table", tableHeight, tableWidth, hiddenSize));
  finiteArray(table, "position table");
  const coordinates = visionMergeBlockCoordinates({ gridHeight, gridWidth, mergeSize });
  const output = new Float32Array(checkedProduct("position output", gridHeight, gridWidth, hiddenSize));
  for (let token = 0; token < gridHeight * gridWidth; token += 1) {
    const y = bilinearCoordinate(coordinates[token * 2]!, gridHeight, tableHeight);
    const x = bilinearCoordinate(coordinates[token * 2 + 1]!, gridWidth, tableWidth);
    const y0 = Math.floor(y);
    const x0 = Math.floor(x);
    const y1 = Math.min(y0 + 1, tableHeight - 1);
    const x1 = Math.min(x0 + 1, tableWidth - 1);
    const wy = y - y0;
    const wx = x - x0;
    for (let channel = 0; channel < hiddenSize; channel += 1) {
      const p00 = table[(y0 * tableWidth + x0) * hiddenSize + channel]!;
      const p01 = table[(y0 * tableWidth + x1) * hiddenSize + channel]!;
      const p10 = table[(y1 * tableWidth + x0) * hiddenSize + channel]!;
      const p11 = table[(y1 * tableWidth + x1) * hiddenSize + channel]!;
      output[token * hiddenSize + channel] = f32(
        f32(f32(p00 * (1 - wx)) + f32(p01 * wx)) * (1 - wy) +
        f32(f32(p10 * (1 - wx)) + f32(p11 * wx)) * wy,
      );
    }
  }
  return output;
}

/** Evaluates the vision LayerNorm form used before attention, MLP, and merge shuffle. */
export function visionLayerNorm(input: {
  readonly input: Float32Array;
  readonly tokenCount: number;
  readonly hiddenSize: number;
  readonly weight: Float32Array;
  readonly bias: Float32Array;
  readonly epsilon?: number;
}): Float32Array {
  const { input: values, tokenCount, hiddenSize, weight, bias } = input;
  const epsilon = input.epsilon ?? LAYER_NORM_EPSILON;
  expectedLength("LayerNorm input", values, checkedProduct("LayerNorm input", tokenCount, hiddenSize));
  expectedLength("LayerNorm weight", weight, hiddenSize);
  expectedLength("LayerNorm bias", bias, hiddenSize);
  if (!Number.isFinite(epsilon) || epsilon <= 0) fail("LayerNorm epsilon must be finite and positive");
  finiteArray(values, "LayerNorm input");
  finiteArray(weight, "LayerNorm weight");
  finiteArray(bias, "LayerNorm bias");
  const output = new Float32Array(values.length);
  for (let token = 0; token < tokenCount; token += 1) {
    // Shift by one input before reducing. This retains low bits when every lane
    // has a large common offset while keeping each arithmetic step in FP32.
    const origin = values[token * hiddenSize]!;
    let centeredSum = 0;
    for (let channel = 0; channel < hiddenSize; channel += 1) {
      centeredSum = f32(centeredSum + f32(values[token * hiddenSize + channel]! - origin));
    }
    const mean = f32(origin + f32(centeredSum / hiddenSize));
    let variance = 0;
    for (let channel = 0; channel < hiddenSize; channel += 1) {
      const delta = f32(values[token * hiddenSize + channel]! - mean);
      variance = f32(variance + f32(delta * delta));
    }
    const inverseDeviation = f32(1 / Math.sqrt(f32(variance / hiddenSize) + epsilon));
    for (let channel = 0; channel < hiddenSize; channel += 1) {
      output[token * hiddenSize + channel] = f32(
        f32(f32(values[token * hiddenSize + channel]! - mean) * inverseDeviation) * weight[channel]! + bias[channel]!,
      );
    }
  }
  return output;
}

/** Applies a BF16 GGUF matrix whose first shape dimension is contiguous input. */
export function visionLinearBf16(input: {
  readonly input: Float32Array;
  readonly tokenCount: number;
  readonly inputWidth: number;
  readonly outputWidth: number;
  readonly weight: Uint16Array;
  readonly bias: Float32Array;
}): Float32Array {
  positiveInteger(input.tokenCount, "linear token count");
  positiveInteger(input.inputWidth, "linear input width");
  positiveInteger(input.outputWidth, "linear output width");
  return linearBf16(input);
}

/** Applies Qwen vision's split-half 2D rotary embeddings to Q and K only. */
export function applyVision2dRotary(input: {
  readonly query: Float32Array;
  readonly key: Float32Array;
  readonly tokenCount: number;
  readonly headCount: number;
  readonly headDimension: number;
  /** [height, width] for every token, already in merge-block sequence order. */
  readonly positions: Int32Array;
  readonly theta: number;
}): Readonly<{ query: Float32Array; key: Float32Array }> {
  const { query, key, tokenCount, headCount, headDimension, positions, theta } = input;
  const vectorCount = checkedProduct("rotary vector count", tokenCount, headCount, headDimension);
  expectedLength("rotary query", query, vectorCount);
  expectedLength("rotary key", key, vectorCount);
  expectedLength("rotary positions", positions, checkedProduct("rotary positions", tokenCount, 2));
  if (headDimension % 4 !== 0) fail("rotary head dimension must divide four sections");
  if (!Number.isFinite(theta) || theta <= 1) fail("rotary theta must be finite and greater than one");
  finiteArray(query, "rotary query");
  finiteArray(key, "rotary key");
  const queryOutput = new Float32Array(query);
  const keyOutput = new Float32Array(key);
  const halfDimension = headDimension / 2;
  const frequenciesPerAxis = halfDimension / 2;
  for (let token = 0; token < tokenCount; token += 1) {
    const positionY = positions[token * 2]!;
    const positionX = positions[token * 2 + 1]!;
    for (let head = 0; head < headCount; head += 1) {
      const headOffset = (token * headCount + head) * headDimension;
      for (let lane = 0; lane < halfDimension; lane += 1) {
        const axis = lane < frequenciesPerAxis ? positionY : positionX;
        const frequency = lane % frequenciesPerAxis;
        // Qwen builds [height frequencies, width frequencies], duplicates it,
        // then rotate_half pairs lane i with i + headDimension / 2.
        // Match PyTorch's FP32 inv_freq construction and FP32 position product.
        const exponent = f32((2 * frequency) / halfDimension);
        const inverseFrequency = f32(1 / f32(theta ** exponent));
        const angle = f32(f32(axis) * inverseFrequency);
        const cosine = f32(Math.cos(angle));
        const sine = f32(Math.sin(angle));
        const left = headOffset + lane;
        const right = left + halfDimension;
        const qLeft = queryOutput[left]!;
        const qRight = queryOutput[right]!;
        const kLeft = keyOutput[left]!;
        const kRight = keyOutput[right]!;
        queryOutput[left] = f32(qLeft * cosine - qRight * sine);
        queryOutput[right] = f32(qRight * cosine + qLeft * sine);
        keyOutput[left] = f32(kLeft * cosine - kRight * sine);
        keyOutput[right] = f32(kRight * cosine + kLeft * sine);
      }
    }
  }
  return Object.freeze({ query: queryOutput, key: keyOutput });
}

/** Runs one non-causal attention segment per image so patches never cross image boundaries. */
export function visionNonCausalAttention(input: {
  readonly query: Float32Array;
  readonly key: Float32Array;
  readonly value: Float32Array;
  readonly tokenCount: number;
  readonly headCount: number;
  readonly headDimension: number;
  readonly segmentLengths: readonly number[];
}): Float32Array {
  const { query, key, value, tokenCount, headCount, headDimension, segmentLengths } = input;
  const vectorLength = checkedProduct("attention vector length", tokenCount, headCount, headDimension);
  let partitionedTokens = 0;
  let scoreEvaluations = 0;
  if (segmentLengths.length === 0) fail("attention segment lengths must partition tokens");
  for (const segmentLength of segmentLengths) {
    if (!Number.isSafeInteger(segmentLength) || segmentLength < 1) {
      fail("attention segment lengths must partition tokens");
    }
    partitionedTokens += segmentLength;
    scoreEvaluations += segmentLength * segmentLength * headCount;
    if (!Number.isSafeInteger(scoreEvaluations) || scoreEvaluations > MAX_ATTENTION_SCORE_EVALUATIONS) {
      fail("attention work bound exceeded");
    }
  }
  if (partitionedTokens !== tokenCount) fail("attention segment lengths must partition tokens");
  expectedLength("attention query", query, vectorLength);
  expectedLength("attention key", key, vectorLength);
  expectedLength("attention value", value, vectorLength);
  finiteArray(query, "attention query");
  finiteArray(key, "attention key");
  finiteArray(value, "attention value");
  const output = new Float32Array(vectorLength);
  const scale = f32(1 / Math.sqrt(headDimension));
  let segmentOffset = 0;
  for (const segmentLength of segmentLengths) {
    for (let queryToken = 0; queryToken < segmentLength; queryToken += 1) {
      for (let head = 0; head < headCount; head += 1) {
        let maximum = -Infinity;
        const scores = new Float32Array(segmentLength);
        for (let keyToken = 0; keyToken < segmentLength; keyToken += 1) {
          let dot = 0;
          for (let dimension = 0; dimension < headDimension; dimension += 1) {
            dot = f32(dot + f32(
              query[((segmentOffset + queryToken) * headCount + head) * headDimension + dimension]! *
              key[((segmentOffset + keyToken) * headCount + head) * headDimension + dimension]!,
            ));
          }
          scores[keyToken] = f32(dot * scale);
          maximum = Math.max(maximum, scores[keyToken]!);
        }
        let denominator = 0;
        for (let keyToken = 0; keyToken < segmentLength; keyToken += 1) {
          scores[keyToken] = f32(Math.exp(scores[keyToken]! - maximum));
          denominator = f32(denominator + scores[keyToken]!);
        }
        if (!Number.isFinite(denominator) || denominator <= 0) fail("attention softmax became non-finite");
        for (let dimension = 0; dimension < headDimension; dimension += 1) {
          let weighted = 0;
          for (let keyToken = 0; keyToken < segmentLength; keyToken += 1) {
            weighted = f32(weighted + f32(
              f32(scores[keyToken]! / denominator) * value[((segmentOffset + keyToken) * headCount + head) * headDimension + dimension]!,
            ));
          }
          output[((segmentOffset + queryToken) * headCount + head) * headDimension + dimension] = f32(weighted);
        }
      }
    }
    segmentOffset += segmentLength;
  }
  return output;
}

/** Qwen's vision transformer uses tanh GELU, unlike the post-shuffle merger. */
export function visionGeluPytorchTanh(values: Float32Array): Float32Array {
  finiteArray(values, "GELU input");
  const output = new Float32Array(values.length);
  const coefficient = Math.sqrt(2 / Math.PI);
  for (let index = 0; index < values.length; index += 1) {
    const value = values[index]!;
    output[index] = f32(0.5 * value * (1 + Math.tanh(coefficient * (value + 0.044715 * value ** 3))));
  }
  return output;
}

function erf(value: number): number {
  const absolute = Math.abs(value);
  const t = 1 / (1 + 0.3275911 * absolute);
  // JavaScript has no Math.erf. This stays with the explicit erf GELU formula
  // (rather than substituting tanh GELU) within the declared CPU-oracle tolerance.
  const polynomial = (((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t);
  const approximation = 1 - polynomial * Math.exp(-absolute * absolute);
  return value < 0 ? -approximation : approximation;
}

/** Uses the explicit erf GELU required by Qwen's merger MLP. */
export function visionGeluExact(values: Float32Array): Float32Array {
  finiteArray(values, "exact GELU input");
  const output = new Float32Array(values.length);
  for (let index = 0; index < values.length; index += 1) {
    const value = values[index]!;
    output[index] = f32(0.5 * value * (1 + erf(value / Math.SQRT2)));
  }
  return output;
}

function addResidual(left: Float32Array, right: Float32Array): Float32Array {
  expectedLength("residual", right, left.length);
  finiteArray(left, "residual left");
  finiteArray(right, "residual right");
  const output = new Float32Array(left.length);
  for (let index = 0; index < output.length; index += 1) output[index] = f32(left[index]! + right[index]!);
  return output;
}

function splitTokenMajorQkv(qkv: Float32Array, tokenCount: number, hiddenSize: number): {
  readonly query: Float32Array;
  readonly key: Float32Array;
  readonly value: Float32Array;
} {
  expectedLength("QKV output", qkv, checkedProduct("QKV output", tokenCount, hiddenSize, 3));
  const vectorSize = checkedProduct("QKV vectors", tokenCount, hiddenSize);
  const query = new Float32Array(vectorSize);
  const key = new Float32Array(vectorSize);
  const value = new Float32Array(vectorSize);
  for (let token = 0; token < tokenCount; token += 1) {
    const source = token * hiddenSize * 3;
    const destination = token * hiddenSize;
    // The linear result is token-major [Q, K, V], not three model-wide ranges.
    query.set(qkv.subarray(source, source + hiddenSize), destination);
    key.set(qkv.subarray(source + hiddenSize, source + hiddenSize * 2), destination);
    value.set(qkv.subarray(source + hiddenSize * 2, source + hiddenSize * 3), destination);
  }
  return { query, key, value };
}

/** Runs one reduced Qwen vision block in official residual order for kernel comparisons. */
export function visionTransformerLayer(input: {
  readonly input: Float32Array;
  readonly tokenCount: number;
  readonly hiddenSize: number;
  readonly headCount: number;
  readonly qkvWeight: Uint16Array;
  readonly qkvBias: Float32Array;
  readonly attentionOutputWeight: Uint16Array;
  readonly attentionOutputBias: Float32Array;
  readonly preAttentionWeight: Float32Array;
  readonly preAttentionBias: Float32Array;
  readonly preMlpWeight: Float32Array;
  readonly preMlpBias: Float32Array;
  readonly mlpUpWeight: Uint16Array;
  readonly mlpUpBias: Float32Array;
  readonly mlpDownWeight: Uint16Array;
  readonly mlpDownBias: Float32Array;
  readonly feedForwardSize: number;
  readonly epsilon?: number;
  readonly positions: Int32Array;
  readonly segmentLengths: readonly number[];
}): Float32Array {
  const { tokenCount, hiddenSize, headCount, feedForwardSize } = input;
  const epsilon = input.epsilon ?? LAYER_NORM_EPSILON;
  positiveInteger(tokenCount, "layer token count");
  positiveInteger(hiddenSize, "layer hidden size");
  positiveInteger(headCount, "layer head count");
  positiveInteger(feedForwardSize, "layer feed-forward size");
  if (hiddenSize % headCount !== 0) fail("layer hidden size must divide head count");
  const normalizedAttention = visionLayerNorm({
    input: input.input, tokenCount, hiddenSize, weight: input.preAttentionWeight, bias: input.preAttentionBias, epsilon,
  });
  const qkv = linearBf16({
    input: normalizedAttention, tokenCount, inputWidth: hiddenSize, outputWidth: hiddenSize * 3, weight: input.qkvWeight, bias: input.qkvBias,
  });
  const { query, key, value } = splitTokenMajorQkv(qkv, tokenCount, hiddenSize);
  const rotated = applyVision2dRotary({
    query, key, tokenCount, headCount, headDimension: hiddenSize / headCount, positions: input.positions, theta: 10_000,
  });
  const attention = visionNonCausalAttention({
    query: rotated.query, key: rotated.key, value, tokenCount, headCount, headDimension: hiddenSize / headCount, segmentLengths: input.segmentLengths,
  });
  const attentionOutput = linearBf16({
    input: attention, tokenCount, inputWidth: hiddenSize, outputWidth: hiddenSize,
    weight: input.attentionOutputWeight, bias: input.attentionOutputBias,
  });
  const afterAttention = addResidual(input.input, attentionOutput);
  const normalizedMlp = visionLayerNorm({
    input: afterAttention, tokenCount, hiddenSize, weight: input.preMlpWeight, bias: input.preMlpBias, epsilon,
  });
  const mlpUp = linearBf16({
    input: normalizedMlp, tokenCount, inputWidth: hiddenSize, outputWidth: feedForwardSize, weight: input.mlpUpWeight, bias: input.mlpUpBias,
  });
  const mlpDown = linearBf16({
    input: visionGeluPytorchTanh(mlpUp), tokenCount, inputWidth: feedForwardSize, outputWidth: hiddenSize,
    weight: input.mlpDownWeight, bias: input.mlpDownBias,
  });
  return addResidual(afterAttention, mlpDown);
}

/** Applies Qwen's pre-shuffle norm, 2x2 spatial concatenate, exact GELU, and two BF16 projections. */
export function visionMerger(input: {
  readonly input: Float32Array;
  readonly patchCount: number;
  readonly hiddenSize: number;
  readonly mergeSize: number;
  readonly normalizationWeight: Float32Array;
  readonly normalizationBias: Float32Array;
  readonly epsilon?: number;
  readonly inputWeight: Uint16Array;
  readonly inputBias: Float32Array;
  readonly intermediateSize: number;
  readonly outputWeight: Uint16Array;
  readonly outputBias: Float32Array;
  readonly outputSize: number;
}): Float32Array {
  const { patchCount, hiddenSize, mergeSize, intermediateSize, outputSize } = input;
  const epsilon = input.epsilon ?? LAYER_NORM_EPSILON;
  positiveInteger(patchCount, "merger patch count");
  positiveInteger(hiddenSize, "merger hidden size");
  positiveInteger(mergeSize, "merger merge size");
  positiveInteger(intermediateSize, "merger intermediate size");
  positiveInteger(outputSize, "merger output size");
  const patchesPerOutput = mergeSize * mergeSize;
  if (patchCount % patchesPerOutput !== 0) fail("merger patch count must divide merge blocks");
  const normalized = visionLayerNorm({
    input: input.input, tokenCount: patchCount, hiddenSize,
    weight: input.normalizationWeight, bias: input.normalizationBias, epsilon,
  });
  const outputTokens = patchCount / patchesPerOutput;
  const shuffledWidth = hiddenSize * patchesPerOutput;
  const shuffled = new Float32Array(checkedProduct("merger shuffle", outputTokens, shuffledWidth));
  for (let outputToken = 0; outputToken < outputTokens; outputToken += 1) {
    for (let patch = 0; patch < patchesPerOutput; patch += 1) {
      shuffled.set(
        normalized.subarray((outputToken * patchesPerOutput + patch) * hiddenSize, (outputToken * patchesPerOutput + patch + 1) * hiddenSize),
        outputToken * shuffledWidth + patch * hiddenSize,
      );
    }
  }
  const projected = linearBf16({
    input: shuffled, tokenCount: outputTokens, inputWidth: shuffledWidth, outputWidth: intermediateSize,
    weight: input.inputWeight, bias: input.inputBias,
  });
  return linearBf16({
    input: visionGeluExact(projected), tokenCount: outputTokens, inputWidth: intermediateSize, outputWidth: outputSize,
    weight: input.outputWeight, bias: input.outputBias,
  });
}
