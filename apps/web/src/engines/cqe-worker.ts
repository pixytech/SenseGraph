import init, {
  create_vector_store,
  load_embeddings_f16,
  load_embeddings_bf16,
  load_embeddings_f32,
  query_similar,
  query_similar_multi,
  cosine_sim,
  vector_store_stats,
  init_neighbor_graph,
  export_neighbor_graph,
  import_neighbor_graph,
  merge_neighbor_range,
  extend_neighbor_graph,
  lookup_neighbors,
  neighbor_graph_size,
} from "../../wasm-pkg/sensegraph_core.js";
import { createTokenizer, type BPETokenizer } from "../tokenizer/bpe-browser";
import type { WorkerMessage, WorkerResponse, SimilarResult, ModelSizeInfo } from "./types";
import type { BuilderResponse } from "./graph-builder-worker";

let storeHandle: number | null = null;
let tokenizer: BPETokenizer | null = null;
let vocabList: string[] = [];
let rawEmbData: ArrayBuffer | null = null;
let cacheTag = "model";
const NEIGHBORS_K = 20;
const BATCH_SIZE = 500;

function getSizeInfo(vocabSize: number, embedDim: number, graphIndexed: number): ModelSizeInfo {
  const embMB = (vocabSize * embedDim * 4) / (1024 * 1024);
  const graphMB = (graphIndexed * NEIGHBORS_K * 8) / (1024 * 1024);
  const attentionMB = (22 * 4 * embedDim * embedDim * 4) / (1024 * 1024);
  return {
    embeddingsMB: embMB,
    graphMB,
    totalMB: embMB + graphMB,
    vocabSize,
    embedDim,
    neighborsIndexed: graphIndexed,
    attentionLayersMB: attentionMB,
  };
}

function post(msg: WorkerResponse) {
  self.postMessage(msg);
}

function graphFile(vocabSize: number, embedDim: number): string {
  return `cqe-graph-v3-${cacheTag}-${vocabSize}-${embedDim}-k${NEIGHBORS_K}.bin`;
}

function progressFile(vocabSize: number, embedDim: number): string {
  return `cqe-progress-${cacheTag}-${vocabSize}-${embedDim}-k${NEIGHBORS_K}.bin`;
}

async function opfsDir(): Promise<FileSystemDirectoryHandle> {
  const root = await navigator.storage.getDirectory();
  return root.getDirectoryHandle("cqe-cache", { create: true });
}

async function opfsRead(name: string): Promise<ArrayBuffer | null> {
  try {
    const dir = await opfsDir();
    const fh = await dir.getFileHandle(name);
    const file = await fh.getFile();
    return await file.arrayBuffer();
  } catch {
    return null;
  }
}

async function opfsWrite(name: string, data: ArrayBuffer): Promise<void> {
  try {
    const dir = await opfsDir();
    const fh = await dir.getFileHandle(name, { create: true });
    const writable = await fh.createWritable();
    await writable.write(data);
    await writable.close();
  } catch (err) {
    console.warn("OPFS write failed:", err);
  }
}

async function opfsDelete(name: string): Promise<void> {
  try {
    const dir = await opfsDir();
    await dir.removeEntry(name);
  } catch {}
}

function loadEmbeddings(embBuf: ArrayBuffer) {
  const embArr = new Uint8Array(embBuf);
  const view = new DataView(embBuf);
  const vocabSize = view.getUint32(0, true);
  const embedDim = view.getUint32(4, true);
  const dtypeCode = view.getUint32(8, true);

  post({ type: "progress", message: `Loading ${vocabSize.toLocaleString()} x ${embedDim}d vectors...`, percent: 45 });
  storeHandle = create_vector_store(embedDim);
  const embData = embArr.slice(16);

  if (dtypeCode === 1) load_embeddings_f16(storeHandle, embData);
  else if (dtypeCode === 3) load_embeddings_bf16(storeHandle, embData);
  else load_embeddings_f32(storeHandle, embData);

  // Convert to F32 format for sub-workers (they always get F32)
  const f32Header = new ArrayBuffer(16);
  const hv = new DataView(f32Header);
  hv.setUint32(0, vocabSize, true);
  hv.setUint32(4, embedDim, true);
  hv.setUint32(8, 2, true); // F32
  hv.setUint32(12, 0, true);

  // Extract normalized vectors from WASM for sub-workers
  // Sub-workers will re-load from this buffer
  rawEmbData = embBuf;
}

async function loadOrBuildGraph(vocabSize: number, embedDim: number) {
  const gFile = graphFile(vocabSize, embedDim);
  const pFile = progressFile(vocabSize, embedDim);

  post({ type: "progress", message: "Checking OPFS cache...", percent: 50 });
  const cached = await opfsRead(gFile);

  if (cached && cached.byteLength > 8) {
    const expectedSize = 8 + vocabSize * NEIGHBORS_K * 8;
    const cachedBytes = cached.byteLength;
    post({ type: "progress", message: `OPFS cache: ${(cachedBytes / 1024 / 1024).toFixed(1)} MB (expected ${(expectedSize / 1024 / 1024).toFixed(1)} MB)`, percent: 55 });

    if (cachedBytes >= expectedSize * 0.9) {
      import_neighbor_graph(storeHandle!, new Uint8Array(cached));
      const gs = neighbor_graph_size(storeHandle!);
      post({ type: "ready", vocabSize, modelName: `CQE (${embedDim}d, ${vocabSize.toLocaleString()} words, ${gs.toLocaleString()} indexed) [OPFS cached]`, sizeInfo: getSizeInfo(vocabSize, embedDim, gs) });
      return;
    }

    post({ type: "progress", message: `Cache incomplete (${(cachedBytes/1024).toFixed(0)}KB / ${(expectedSize/1024).toFixed(0)}KB), rebuilding...`, percent: 50 });
  }

  let resumeFrom = 0;
  const partial = await opfsRead(pFile);
  if (partial && partial.byteLength > 8) {
    post({ type: "progress", message: `Resuming from OPFS checkpoint (${(partial.byteLength / 1024 / 1024).toFixed(1)} MB)...`, percent: 52 });
    import_neighbor_graph(storeHandle!, new Uint8Array(partial));
    resumeFrom = neighbor_graph_size(storeHandle!);
    extend_neighbor_graph(storeHandle!, vocabSize);
    post({ type: "progress", message: `Resuming from word ${resumeFrom.toLocaleString()}/${vocabSize.toLocaleString()}`, percent: 54 });
  } else {
    init_neighbor_graph(storeHandle!, vocabSize, NEIGHBORS_K);
  }

  await buildGraphParallel(vocabSize, embedDim, resumeFrom, gFile, pFile);
}

async function buildGraphParallel(vocabSize: number, embedDim: number, startFrom: number, gFile: string, pFile: string) {
  const numWorkers = Math.min(navigator.hardwareConcurrency || 4, 8);
  const remaining = vocabSize - startFrom;
  const chunkSize = Math.ceil(remaining / numWorkers);

  post({
    type: "progress",
    message: `Spawning ${numWorkers} parallel builders (${navigator.hardwareConcurrency} CPU cores detected)...`,
    percent: 56,
  });

  if (!rawEmbData) {
    post({ type: "error", message: "Embedding data not available for parallel build" });
    return;
  }
  const startTime = performance.now();

  const workerProgress = new Map<number, { done: number; ops: number }>();
  let completedRanges = 0;

  await new Promise<void>((resolve, reject) => {
    for (let w = 0; w < numWorkers; w++) {
      const rangeStart = startFrom + w * chunkSize;
      const rangeEnd = Math.min(rangeStart + chunkSize, vocabSize);
      if (rangeStart >= vocabSize) {
        completedRanges++;
        if (completedRanges >= numWorkers) resolve();
        continue;
      }

      workerProgress.set(rangeStart, { done: 0, ops: 0 });

      const worker = new Worker(
        new URL("./graph-builder-worker.ts", import.meta.url),
        { type: "module" }
      );

      const embCopy = rawEmbData!.slice(0);

      worker.onmessage = (e: MessageEvent<BuilderResponse>) => {
        const msg = e.data;
        if (msg.type === "progress") {
          workerProgress.set(msg.rangeStart, { done: msg.done, ops: msg.ops });

          let totalDone = 0;
          let totalOps = 0;
          for (const p of workerProgress.values()) {
            totalDone += p.done;
            totalOps += p.ops;
          }

          const elapsed = (performance.now() - startTime) / 1000;
          const wordsPerSec = totalDone / Math.max(elapsed, 0.1);
          const eta = (remaining - totalDone) / Math.max(wordsPerSec, 1);
          const pct = 57 + Math.round((totalDone / remaining) * 35);

          post({
            type: "progress",
            message: `Building: ${(startFrom + totalDone).toLocaleString()}/${vocabSize.toLocaleString()} | ${numWorkers} threads | ${wordsPerSec.toFixed(0)} words/s | ${(totalOps / Math.max(elapsed, 0.1) / 1e6).toFixed(0)}M ops/s | ETA: ${formatTime(eta)}`,
            percent: pct,
          });
        } else if (msg.type === "done") {
          merge_neighbor_range(storeHandle!, msg.rangeStart, new Uint8Array(msg.graphData));
          worker.terminate();
          completedRanges++;
          if (completedRanges >= numWorkers) resolve();
        } else if (msg.type === "error") {
          worker.terminate();
          reject(new Error(msg.message));
        }
      };

      worker.postMessage(
        {
          type: "init",
          wasmUrl: `${import.meta.env.BASE_URL}sensegraph_core_bg.wasm`,
          embData: embCopy,
          vocabSize, embedDim, rangeStart, rangeEnd,
          k: NEIGHBORS_K, batchSize: BATCH_SIZE,
        },
        [embCopy]
      );
    }
  });

  const totalTime = (performance.now() - startTime) / 1000;

  post({ type: "progress", message: "Saving to OPFS...", percent: 92 });

  extend_neighbor_graph(storeHandle!, vocabSize);
  const graphData = export_neighbor_graph(storeHandle!);
  const graphBuf = new Uint8Array(graphData).buffer;
  await opfsWrite(gFile, graphBuf);
  await opfsDelete(pFile);

  const savedSize = graphBuf.byteLength;
  const gs = neighbor_graph_size(storeHandle!);
  const si = getSizeInfo(vocabSize, embedDim, gs);
  post({
    type: "ready",
    vocabSize,
    modelName: `CQE (${embedDim}d, ${vocabSize.toLocaleString()} words, ${gs.toLocaleString()} indexed) | Built ${totalTime.toFixed(0)}s, ${numWorkers} threads, saved ${(savedSize/1024/1024).toFixed(1)}MB to OPFS`,
    sizeInfo: si,
  });
}


function formatTime(sec: number): string {
  if (sec < 60) return `${Math.round(sec)}s`;
  const m = Math.floor(sec / 60);
  const s = Math.round(sec % 60);
  return `${m}m${s}s`;
}

async function initWithModel(modelId: string, _modelName: string, tokJson: any, embeddings: ArrayBuffer) {
  try {
    cacheTag = modelId.replace(/[^a-z0-9]+/gi, "-").toLowerCase();
    post({ type: "progress", message: "Loading WASM...", percent: 42 });
    await init({ module_or_path: `${import.meta.env.BASE_URL}sensegraph_core_bg.wasm` });

    tokenizer = createTokenizer(tokJson);
    const entries = Object.entries(tokJson.model.vocab) as [string, number][];
    entries.sort((a, b) => a[1] - b[1]);
    vocabList = entries.map(([w]) => w);

    loadEmbeddings(embeddings);
    const stats = vector_store_stats(storeHandle!);
    await loadOrBuildGraph(stats.count, stats.embed_dim);
  } catch (err: any) {
    post({ type: "error", message: err.message || String(err) });
  }
}

function handleSimilar(word: string, topK: number) {
  if (storeHandle === null || !tokenizer) {
    post({ type: "result", data: { type: "similar", query: word, results: [], latencyMs: 0, engine: "cqe", error: "Engine still loading..." } });
    return;
  }
  try {
    const t0 = performance.now();
    const ids = tokenizer.encode(word);

    if (ids.length === 0) {
      post({ type: "result", data: { type: "similar", query: word, results: [], latencyMs: performance.now() - t0, engine: "cqe", error: `Unknown: "${word}"` } });
      return;
    }

    let results: SimilarResult[];
    if (ids.length === 1) {
      const nbRaw = lookup_neighbors(storeHandle!, ids[0]) as any[];
      if (nbRaw && nbRaw.length > 0) {
        results = nbRaw.slice(0, topK).map((r: any) => ({
          word: vocabList[r.index] || `<${r.index}>`, score: r.score, index: r.index,
        }));
      } else {
        const raw = query_similar(storeHandle!, ids[0], topK) as any[];
        results = raw.map((r: any) => ({
          word: vocabList[r.index] || `<${r.index}>`, score: r.score, index: r.index,
        }));
      }
    } else {
      const raw = query_similar_multi(storeHandle!, new Uint32Array(ids), topK) as any[];
      results = raw.map((r: any) => ({
        word: vocabList[r.index] || `<${r.index}>`, score: r.score, index: r.index,
      }));
    }

    post({ type: "result", data: { type: "similar", query: word, results, latencyMs: performance.now() - t0, engine: "cqe" } });
  } catch (err: any) {
    post({ type: "result", data: { type: "similar", query: word, results: [], latencyMs: 0, engine: "cqe", error: String(err) } });
  }
}

function handleSearch(query: string, topK: number) {
  if (storeHandle === null || !tokenizer) return;
  const t0 = performance.now();
  const ids = tokenizer.encode(query);
  if (ids.length === 0) {
    post({ type: "result", data: { type: "search", query, results: [], latencyMs: 0, engine: "cqe", error: "No tokens" } });
    return;
  }
  const raw = query_similar_multi(storeHandle!, new Uint32Array(ids), topK + 20) as any[];
  const STOP = new Set(["the","a","an","is","was","are","in","on","at","to","for","of","with","by","and","or","but","not","it","its","this","that",",",".","!","?"]);
  const results = raw.map((r: any) => ({ word: vocabList[r.index] || `<${r.index}>`, score: r.score, index: r.index }))
    .filter((r: SimilarResult) => r.word.length > 1 && !STOP.has(r.word)).slice(0, topK);
  post({ type: "result", data: { type: "search", query, results, latencyMs: performance.now() - t0, engine: "cqe" } });
}

function handleCompare(word1: string, word2: string) {
  if (storeHandle === null || !tokenizer) return;
  const t0 = performance.now();
  const ids1 = tokenizer.encode(word1);
  const ids2 = tokenizer.encode(word2);
  if (ids1.length !== 1 || ids2.length !== 1) {
    post({ type: "result", data: { type: "compare", query: `${word1} vs ${word2}`, results: [], latencyMs: 0, engine: "cqe", error: "Single tokens only" } });
    return;
  }
  const score = cosine_sim(storeHandle!, ids1[0], ids2[0]);
  post({ type: "result", data: { type: "compare", query: `${word1} vs ${word2}`, results: [{ word: `${word1} <-> ${word2}`, score, index: -1 }], latencyMs: performance.now() - t0, engine: "cqe" } });
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
