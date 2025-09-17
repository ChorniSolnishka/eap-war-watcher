/**
 * Find the highest "screen_id" in a nested payload.
 */
export function extractLastScreenId(payload: any): number | null {
  let max: number | null = null;
  function walk(x: any) {
    if (!x || typeof x !== "object") return;
    if (Array.isArray(x)) return x.forEach(walk);
    for (const [k, v] of Object.entries(x)) {
      if (k === "screen_id" && typeof v === "number") {
        max = max == null ? v : Math.max(max, v);
      } else if (v && typeof v === "object") {
        walk(v);
      }
    }
  }
  try {
    walk(payload);
  } catch {}
  return max;
}
