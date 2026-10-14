import type { WorkerMessage, WorkerResponse, SimilarResult } from "./types";
import { createTokenizer, type BPETokenizer } from "../tokenizer/bpe-browser";
import { cleanResults, contentIds } from "./shared";

let embeddings: Float32Array | null = null;
let vocab: string[] = [];
let tokenizer: BPETokenizer | null = null;
let embedDim = 0;
let norms: Float32Array | null = null;

function post(msg: WorkerResponse) {
  self.postMessage(msg);
}

function computeNorms(data: Float32Array, dim: number, count: number): Float32Array {
  const n = new Float32Array(count);
  for (let i = 0; i < count; i++) {
    let sum = 0;
    const off = i * dim;
    for (let d = 0; d < dim; d++) sum += data[off + d] * data[off + d];
    n[i] = Math.sqrt(sum) || 1;
  }
  return n;
}

function dotProduct(a: Float32Array, aOff: number, b: Float32Array, bOff: number, dim: number): number {
  let sum = 0;
  for (let d = 0; d < dim; d++) sum += a[aOff + d] * b[bOff + d];
  return sum;
}

function simulateAttention(queryIndices: number[]): number {
  if (!embeddings || queryIndices.length === 0) return 0;

  const n = queryIndices.length;
  const d = embedDim;

  const Q = new Float32Array(n * d);
  const K = new Float32Array(n * d);
  const V = new Float32Array(n * d);

  for (let i = 0; i < n; i++) {
    const off = queryIndices[i] * d;
    const qOff = i * d;
    for (let dd = 0; dd < d; dd++) {
      Q[qOff + dd] = embeddings![off + dd] * 1.01;
      K[qOff + dd] = embeddings![off + dd] * 0.99;
      V[qOff + dd] = embeddings![off + dd];
    }
  }

  const attnScores = new Float32Array(n * n);
  const scale = 1.0 / Math.sqrt(d);
  let ops = 0;

  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      attnScores[i * n + j] = dotProduct(Q, i * d, K, j * d, d) * scale;
      ops += d;
    }
  }

  for (let i = 0; i < n; i++) {
    let maxVal = -Infinity;
    for (let j = 0; j < n; j++) maxVal = Math.max(maxVal, attnScores[i * n + j]);
    let sumExp = 0;
    for (let j = 0; j < n; j++) {
      attnScores[i * n + j] = Math.exp(attnScores[i * n + j] - maxVal);
      sumExp += attnScores[i * n + j];
    }
    for (let j = 0; j < n; j++) attnScores[i * n + j] /= sumExp;
    ops += n * 3;
  }

  const output = new Float32Array(n * d);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const w = attnScores[i * n + j];
      const oOff = i * d;
      const vOff = j * d;
      for (let dd = 0; dd < d; dd++) output[oOff + dd] += w * V[vOff + dd];
      ops += d;
    }
  }

  return ops;
}

// Score every vocabulary entry against the mean of the normalized query vectors,
// keeping the best topK as it goes (no full sort), after the simulated attention pass.
function fullVocabAttentionSearch(queryIndices: number[], topK: number): { results: SimilarResult[]; ops: number } {
  if (!embeddings || !norms) return { results: [], ops: 0 };

  let totalOps = simulateAttention(queryIndices);
  const d = embedDim;
  const mean = new Float32Array(d);
  for (const idx of queryIndices) {
    const off = idx * d;
    const inv = 1 / norms[idx];
    for (let dd = 0; dd < d; dd++) mean[dd] += embeddings[off + dd] * inv;
  }
  let mNorm = 0;
  for (let dd = 0; dd < d; dd++) mNorm += mean[dd] * mean[dd];
  mNorm = Math.sqrt(mNorm) || 1;

  const exclude = new Set(queryIndices);
  const best: SimilarResult[] = [];
  let floor = -Infinity;
  for (let i = 0; i < vocab.length; i++) {
    if (exclude.has(i)) continue;
    let dot = 0;
    const off = i * d;
    for (let dd = 0; dd < d; dd++) dot += mean[dd] * embeddings[off + dd];
    totalOps += d;
    const sim = dot / (mNorm * norms[i]);
    if (best.length < topK || sim > floor) {
      best.push({ word: "", score: sim, index: i });
      best.sort((a, b) => b.score - a.score);
      if (best.length > topK) best.pop();
      floor = best[best.length - 1].score;
    }
  }
  for (const r of best) r.word = tokenizer!.tokenText(r.index);
  return { results: best, ops: totalOps };
}

function halfToFloat(h: number): number {
  const s = h & 0x8000 ? -1 : 1;
  const e = (h >> 10) & 0x1f;
  const f = h & 0x3ff;
  if (e === 0) return s * Math.pow(2, -14) * (f / 1024);
  if (e === 31) return f ? NaN : s * Infinity;
  return s * Math.pow(2, e - 15) * (1 + f / 1024);
}

function decodeTensor(raw: Uint8Array, dtype: string, count: number): Float32Array {
  const out = new Float32Array(count);
  const dv = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
  if (dtype === "F32") {
    for (let i = 0; i < count; i++) out[i] = dv.getFloat32(i * 4, true);
  } else if (dtype === "F16") {
    for (let i = 0; i < count; i++) out[i] = halfToFloat(dv.getUint16(i * 2, true));
  } else if (dtype === "BF16") {
    const tmp = new DataView(new ArrayBuffer(4));
    for (let i = 0; i < count; i++) {
      tmp.setUint32(0, dv.getUint16(i * 2, true) << 16, false);
      out[i] = tmp.getFloat32(0, false);
    }
  } else {
    throw new Error(`Unsupported tensor type ${dtype}`);
  }
  return out;
}

function decodeBuffer(buf: ArrayBuffer): { data: Float32Array; rows: number; cols: number } {
  const view = new DataView(buf);
  const rows = view.getUint32(0, true);
  const cols = view.getUint32(4, true);
  const code = view.getUint32(8, true);
  const dtype = code === 1 ? "F16" : code === 3 ? "BF16" : "F32";
  return { data: decodeTensor(new Uint8Array(buf, 16), dtype, rows * cols), rows, cols };
}

async function initWithModel(modelId: string, _modelName: string, tokJson: any, buf: ArrayBuffer) {
  try {
    post({ type: "progress", message: "Decoding vectors...", percent: 45 });
    tokenizer = createTokenizer(tokJson);

    const decoded = decodeBuffer(buf);
    embedDim = decoded.cols;
    embeddings = decoded.data;
    const vocabSize = decoded.rows;
    vocab = new Array(vocabSize).fill("");

    post({ type: "progress", message: "Computing norms...", percent: 90 });
    norms = computeNorms(embeddings, embedDim, vocabSize);

    const embMB = (vocabSize * embedDim * 4) / (1024 * 1024);
    const attentionMB = (22 * 4 * embedDim * embedDim * 4) / (1024 * 1024);
    post({
      type: "ready",
      vocabSize,
      modelName: `Baseline (${modelId}, ${embedDim}d, ${vocabSize.toLocaleString()} tokens)`,
      sizeInfo: {
        embeddingsMB: embMB,
        graphMB: 0,
        totalMB: embMB + attentionMB,
        vocabSize,
        embedDim,
        neighborsIndexed: 0,
        attentionLayersMB: attentionMB,
      },
    });
  } catch (err: any) {
    post({ type: "error", message: err.message || String(err) });
  }
}

function encode(text: string): number[] {
  if (!tokenizer) return [];
  return tokenizer.encodeQuery(text).filter((i) => i < vocab.length);
}

function handleRank(type: "similar" | "search", text: string, topK: number) {
  const ids = encode(text);
  if (ids.length === 0) {
    post({ type: "result", data: { type, query: text, results: [], latencyMs: 0, engine: "transformer", error: `No known tokens in "${text}"` } });
    return;
  }
  const t0 = performance.now();
  const text_of = (i: number) => tokenizer!.tokenText(i);
  const { used, ignored } = type === "search" ? contentIds(ids, text_of) : { used: ids, ignored: [] as number[] };
  const { results, ops } = fullVocabAttentionSearch(used, topK + 20);
  const latencyMs = performance.now() - t0;
  const cleaned = cleanResults(results, topK, type === "search", type === "search" ? used.map(text_of) : []);
  post({ type: "result", data: { type, query: text, results: cleaned, latencyMs, operationCount: ops, engine: "transformer", tokens: ids.map(text_of), ignored: ignored.map(text_of) } });
}

function meanUnit(ids: number[]): Float32Array {
  const m = new Float32Array(embedDim);
  for (const id of ids) {
    const inv = 1 / norms![id];
    for (let d = 0; d < embedDim; d++) m[d] += embeddings![id * embedDim + d] * inv;
  }
  return m;
}

function handleCompare(a: string, b: string) {
  const query = `${a} vs ${b}`;
  const ia = encode(a);
  const ib = encode(b);
  if (ia.length === 0 || ib.length === 0) {
    post({ type: "result", data: { type: "compare", query, results: [], latencyMs: 0, engine: "transformer", error: "No known tokens" } });
    return;
  }
  const t0 = performance.now();
  const ops = simulateAttention([...ia, ...ib]);
  const ma = meanUnit(ia);
  const mb = meanUnit(ib);
  let dot = 0, na = 0, nb = 0;
  for (let d = 0; d < embedDim; d++) {
    dot += ma[d] * mb[d];
    na += ma[d] * ma[d];
    nb += mb[d] * mb[d];
  }
  const score = dot / (Math.sqrt(na * nb) || 1);
  const latencyMs = performance.now() - t0;
  post({ type: "result", data: { type: "compare", query, results: [{ word: `${a} <-> ${b}`, score, index: -1 }], latencyMs, operationCount: ops, engine: "transformer", tokens: [...ia, ...ib].map((i) => tokenizer!.tokenText(i)) } });
}

self.onmessage = (e: MessageEvent<WorkerMessage>) => {
  const msg = e.data;
  switch (msg.type) {
    case "init-with-model": initWithModel(msg.modelId, msg.modelName, msg.tokenizerJson, msg.embeddings); break;
    case "similar": handleRank("similar", msg.word, msg.topK); break;
    case "search": handleRank("search", msg.query, msg.topK); break;
    case "compare": handleCompare(msg.word1, msg.word2); break;
  }
};
