import type { SimilarResult } from "./types";

export const STOP_WORDS = new Set([
  "the", "a", "an", "is", "was", "are", "in", "on", "at", "to", "for", "of", "with",
  "by", "and", "or", "but", "not", "it", "its", "this", "that",
  "from", "into", "as", "be", "been", "were", "has", "have", "had", "do", "does", "did",
  "than", "then", "so", "if", "about", "after", "before", "off", "up", "out",
  "my", "your", "our", "their", "his", "her", "he", "she", "we", "they", "you", "i",
  "will", "would", "can", "could", "should",
]);

// Splits query token ids into those that carry meaning and those that are function words.
// Both engines use only the meaningful ones to build a search query, so the exact scan and
// the graph answer the same question. If every token is a stop word, all of them are used.
export function contentIds(ids: number[], label: (id: number) => string): { used: number[]; ignored: number[] } {
  const used: number[] = [];
  const ignored: number[] = [];
  for (const id of ids) {
    const t = label(id).trim().toLowerCase();
    if (t.length < 2 || STOP_WORDS.has(t)) ignored.push(id);
    else used.push(id);
  }
  return used.length === 0 ? { used: ids, ignored: [] } : { used, ignored };
}

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
