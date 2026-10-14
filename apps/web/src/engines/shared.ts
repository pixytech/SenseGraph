import type { SimilarResult } from "./types";

export const STOP_WORDS = new Set([
  "the", "a", "an", "is", "was", "are", "in", "on", "at", "to", "for", "of", "with",
  "by", "and", "or", "but", "not", "it", "its", "this", "that",
]);

// Drops whitespace-only tokens. For search queries it also drops stop words and
// one character tokens, which add noise when ranking by meaning.
export function cleanResults(results: SimilarResult[], topK: number, search: boolean): SimilarResult[] {
  const out: SimilarResult[] = [];
  const seen = new Set<string>();
  for (const r of results) {
    const t = r.word.trim();
    if (t.length === 0) continue;
    // Case and leading-space variants of the same word count once (best score wins,
    // because results arrive ranked).
    const key = t.toLowerCase();
    if (seen.has(key)) continue;
    seen.add(key);
    if (search && (t.length < 2 || STOP_WORDS.has(t.toLowerCase()))) continue;
    out.push(r);
    if (out.length >= topK) break;
  }
  return out;
}

// Share of the reference top-K that the other list also contains.
export function overlapAtK(a: SimilarResult[], b: SimilarResult[], k: number): number {
  const A = a.slice(0, k);
  const B = new Set(b.slice(0, k).map((r) => r.index));
  if (A.length === 0) return 0;
  let hit = 0;
  for (const r of A) if (B.has(r.index)) hit++;
  return hit / A.length;
}

// Browsers round timer readings inside workers, so very small latencies read as 0.
export const TIMER_FLOOR_MS = 0.1;

export function formatMs(ms: number): string {
  return ms < TIMER_FLOOR_MS ? "<0.1ms" : `${ms.toFixed(1)}ms`;
}

export function formatSpeedup(baselineMs: number, engineMs: number): string {
  if (baselineMs <= 0) return "n/a";
  const floored = engineMs < TIMER_FLOOR_MS;
  const x = baselineMs / Math.max(engineMs, TIMER_FLOOR_MS);
  return `${floored ? ">" : ""}${x < 10 ? x.toFixed(1) : x.toFixed(0)}x`;
}
