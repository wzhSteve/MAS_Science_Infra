function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

// REST snapshots are JSON: retain unchanged branches so polling does not redraw charts or forms.
export function shareSnapshot<T>(previous: T, next: T): T {
  if (Object.is(previous, next)) return previous;
  if (Array.isArray(previous) && Array.isArray(next)) {
    const values = next.map((value: unknown, index: number) => shareSnapshot(previous[index], value));
    return (previous.length === values.length && values.every((value, index) => value === previous[index])
      ? previous : values) as T;
  }
  if (isRecord(previous) && isRecord(next)) {
    const keys = Object.keys(next);
    const result: Record<string, unknown> = { ...next };
    let equal = Object.keys(previous).length === keys.length;
    for (const key of keys) {
      result[key] = shareSnapshot(previous[key], next[key]);
      if (result[key] !== previous[key] || !Object.prototype.hasOwnProperty.call(previous, key)) equal = false;
    }
    return (equal ? previous : result) as T;
  }
  return next;
}
