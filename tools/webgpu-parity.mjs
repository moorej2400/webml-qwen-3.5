export function validateParity(id, expected, actual, tolerance = 2e-4) {
  if (actual.length !== expected.length) {
    throw new Error(
      `${id}: CPU length ${expected.length} GPU length ${actual.length}`,
    );
  }
  for (let index = 0; index < expected.length; index += 1) {
    const cpuValue = expected[index];
    const gpuValue = actual[index];
    if (!Number.isFinite(cpuValue) || !Number.isFinite(gpuValue)) {
      throw new Error(
        `${id}: row ${index} has non-finite CPU ${cpuValue} GPU ${gpuValue}`,
      );
    }
    const error = Math.abs(gpuValue - cpuValue);
    if (error > tolerance * (1 + Math.abs(cpuValue))) {
      throw new Error(
        `${id}: row ${index} CPU ${cpuValue} GPU ${gpuValue}`,
      );
    }
  }
}
