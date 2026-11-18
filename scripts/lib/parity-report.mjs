/**
 * Reduce per-coordinate native/TypeScript comparisons into the stable JSON
 * schema consumed by the parity CLI and its Vitest coverage.
 */
export function createParityAccumulator(sample, mode, byteDifferences) {
  let coordinatesChecked = 0;
  let activeMismatches = 0;
  let maxValueError = 0;
  let totalValueError = 0;
  let numericFailures = 0;
  let invalidNumericBounds = 0;
  let maxNumericExcess = 0;
  const classifiedSemanticDifferences = [];
  const unclassifiedByteDifferences = byteDifferences.filter((difference) => {
    const classification =
      typeof difference === "string" ? difference : difference.classification;
    return classification?.startsWith("unclassified") ?? true;
  });

  return {
    add(comparison) {
      coordinatesChecked++;
      const error = Math.abs(comparison.native.value - comparison.typescript.value);
      maxValueError = Math.max(maxValueError, error);
      totalValueError += error;

      if (comparison.classification) {
        classifiedSemanticDifferences.push({
          coordinate: comparison.coordinate,
          classification: comparison.classification,
          native: comparison.native,
          typescript: comparison.typescript,
        });
        return;
      }

      if (comparison.native.active !== comparison.typescript.active) activeMismatches++;
      const validNumericBound =
        Number.isFinite(comparison.numericBound) && comparison.numericBound >= 0;
      const numericExcess = error - comparison.numericBound;
      if (
        mode === "float"
          ? error !== 0
          : !validNumericBound || numericExcess > Number.EPSILON
      ) {
        numericFailures++;
        if (validNumericBound) maxNumericExcess = Math.max(maxNumericExcess, numericExcess);
        else invalidNumericBounds++;
      }
    },
    finish() {
      return {
        sample,
        mode,
        coordinatesChecked,
        activeMismatches,
        maxValueError,
        meanValueError: coordinatesChecked > 0 ? totalValueError / coordinatesChecked : 0,
        numericFailures,
        invalidNumericBounds,
        maxNumericExcess,
        byteDifferences,
        unclassifiedByteDifferences,
        classifiedSemanticDifferences,
        passed:
          activeMismatches === 0 &&
          numericFailures === 0 &&
          unclassifiedByteDifferences.length === 0,
      };
    },
  };
}

export function summarizeParity(sample, mode, comparisons, byteDifferences) {
  const accumulator = createParityAccumulator(sample, mode, byteDifferences);
  for (const comparison of comparisons) accumulator.add(comparison);
  return accumulator.finish();
}

export function overallPassed(reports) {
  return reports.length > 0 && reports.every((report) => report.passed);
}
