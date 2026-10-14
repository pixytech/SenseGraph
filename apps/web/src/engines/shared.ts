import type { SimilarResult } from "./types";

export const STOP_WORDS = new Set([
  "the", "a", "an", "is", "was", "are", "in", "on", "at", "to", "for", "of", "with",
  "by", "and", "or", "but", "not", "it", "its", "this", "that",
]);

// Drops whitespace-only tokens. For search queries it also drops stop words and
// one character tokens, which add noise when ranking by meaning.
export function cleanResults(results: SimilarResult[], topK: number, search: boolean): SimilarResult[] {
  const out: SimilarResult[] = [];
  for (const r of results) {
    const t = r.word.trim();
    if (t.length === 0) continue;
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
