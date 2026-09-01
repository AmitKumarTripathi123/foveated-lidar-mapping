/**
 * Deterministic numeric formatter that produces identical output across
 * all Node.js server locales (en-IN, en-US, etc.) and client browser environments.
 * Prevents React SSR hydration mismatches.
 */
export function formatInt(num: number): string {
  if (num === null || num === undefined || isNaN(num)) return '0';
  return Math.round(num)
    .toString()
    .replace(/\B(?=(\d{3})+(?!\d))/g, ',');
}

export function formatFloat(num: number, decimals: number = 1): string {
  if (num === null || num === undefined || isNaN(num)) return '0.0';
  return num.toFixed(decimals);
}
