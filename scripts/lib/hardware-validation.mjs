export function validateExactState(expected, state) {
  const errors = [];
  for (const [field, value] of Object.entries(expected)) {
    if (state?.[field] !== value)
      errors.push(`${field}: expected ${JSON.stringify(value)}, got ${JSON.stringify(state?.[field])}`);
  }
  return errors;
}

export function exactStateValidator(expected) {
  return (state) => validateExactState(expected, state);
}
