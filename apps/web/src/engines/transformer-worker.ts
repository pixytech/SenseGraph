import type { WorkerMessage, WorkerResponse, SimilarResult } from "./types";
import { createTokenizer, type BPETokenizer } from "../tokenizer/bpe-browser";

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

function fullVocabAttentionSearch(queryIndices: number[], topK: number): { results: SimilarResult[]; ops: number } {
  if (!embeddings || !norms) return { results: [], ops: 0 };

  let totalOps = simulateAttention(queryIndices);

  const d = embedDim;
  const queryVec = new Float32Array(d);
  for (const idx of queryIndices) {
    const off = idx * d;
    for (let dd = 0; dd < d; dd++) queryVec[dd] += embeddings![off + dd];
  }
  const scale = 1.0 / queryIndices.length;
  let qNorm = 0;
  for (let dd = 0; dd < d; dd++) {
    queryVec[dd] *= scale;
    qNorm += queryVec[dd] * queryVec[dd];
  }
  qNorm = Math.sqrt(qNorm) || 1;

  const excludeSet = new Set(queryIndices);
  const scores: { word: string; score: number; index: number }[] = [];

  for (let i = 0; i < vocab.length; i++) {
    if (excludeSet.has(i)) continue;
    let dot = 0;
    const off = i * d;
    for (let dd = 0; dd < d; dd++) dot += queryVec[dd] * embeddings![off + dd];
    const sim = dot / (qNorm * norms![i]);
    scores.push({ word: vocab[i], score: sim, index: i });
    totalOps += d;
  }

  scores.sort((a, b) => b.score - a.score);
  return { results: scores.slice(0, topK), ops: totalOps };
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
    const entries = Object.entries(tokJson.model.vocab) as [string, number][];
    entries.sort((a, b) => a[1] - b[1]);
    vocab = entries.map(([word]) => word);

    const decoded = decodeBuffer(buf);
    embedDim = decoded.cols;
    embeddings = decoded.data;
    const vocabSize = Math.min(decoded.rows, vocab.length);

    post({ type: "progress", message: "Computing norms...", percent: 90 });
    norms = computeNorms(embeddings, embedDim, vocabSize);
    vocab = vocab.slice(0, vocabSize);

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
  return tokenizer.encode(text).filter((i) => i < vocab.length);
}

function handleSimilar(word: string, topK: number) {
  const ids = encode(word);
  if (ids.length === 0) {
    post({ type: "result", data: { type: "similar", query: word, results: [], latencyMs: 0, engine: "transformer", error: `Unknown: "${word}"` } });
    return;
  }
  const t0 = performance.now();
  const { results, ops } = fullVocabAttentionSearch(ids, topK);
  const latencyMs = performance.now() - t0;
  post({ type: "result", data: { type: "similar", query: word, results, latencyMs, operationCount: ops, engine: "transformer" } });
}

function handleSearch(query: string, topK: number) {
  const ids = encode(query);
  if (ids.length === 0) {
    post({ type: "result", data: { type: "search", query, results: [], latencyMs: 0, engine: "transformer", error: "No tokens" } });
    return;
  }
  const t0 = performance.now();
  const { results, ops } = fullVocabAttentionSearch(ids, topK + 20);
  const latencyMs = performance.now() - t0;

  const STOP = new Set(["the","a","an","is","was","are","in","on","at","to","for","of","with","by","and","or","but","not","it","its","this","that",",",".","!","?"]);
  const filtered = results.filter(r => r.word.length > 1 && !STOP.has(r.word)).slice(0, topK);
  post({ type: "result", data: { type: "search", query, results: filtered, latencyMs, operationCount: ops, engine: "transformer" } });
}

function handleCompare(word1: string, word2: string) {
  const a = encode(word1);
  const b = encode(word2);
  if (a.length !== 1 || b.length !== 1) {
    post({ type: "result", data: { type: "compare", query: `${word1} vs ${word2}`, results: [], latencyMs: 0, engine: "transformer", error: "Single tokens only" } });
    return;
  }
  const i1 = a[0];
  const i2 = b[0];
  const t0 = performance.now();
  const ops = simulateAttention([i1, i2]);
  let dot = 0;
  for (let d = 0; d < embedDim; d++) dot += embeddings![i1 * embedDim + d] * embeddings![i2 * embedDim + d];
  const score = dot / (norms![i1] * norms![i2]);
  const latencyMs = performance.now() - t0;
  post({ type: "result", data: { type: "compare", query: `${word1} vs ${word2}`, results: [{ word: `${word1} <-> ${word2}`, score, index: -1 }], latencyMs, operationCount: ops, engine: "transformer" } });
}

self.onmessage = (e: MessageEvent<WorkerMessage>) => {
  const msg = e.data;
  switch (msg.type) {
    case "init-with-model": initWithModel(msg.modelId, msg.modelName, msg.tokenizerJson, msg.embeddings); break;
    case "similar": handleSimilar(msg.word, msg.topK); break;
    case "search": handleSearch(msg.query, msg.topK); break;
    case "compare": handleCompare(msg.word1, msg.word2); break;
  }
};
