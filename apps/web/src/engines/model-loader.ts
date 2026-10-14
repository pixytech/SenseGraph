import type { ModelLoadInfo } from "./types";

// Downloads a model once so that both engines can be started from the same data.
// Only the tokenizer and the embedding tensor are fetched, using HTTP range
// requests against the safetensors file.

export interface LoadedModel {
  id: string;
  name: string;
  tokenizerJson: any;
  // 16 byte header (rows, cols, dtype code, 0) followed by the raw tensor bytes.
  // dtype code: 1 = F16, 2 = F32, 3 = BF16.
  embeddings: ArrayBuffer;
  rows: number;
  cols: number;
}

export type ProgressFn = (message: string, percent: number) => void;

const HF = "https://huggingface.co";

export async function loadModel(info: ModelLoadInfo, onProgress: ProgressFn): Promise<LoadedModel> {
  if (info.source !== "huggingface" || !info.hfRepo) {
    throw new Error("Select a model from the list and press Load Model");
  }
  const repo = info.hfRepo;
  const file = info.hfFile || "model.safetensors";

  onProgress(`Downloading tokenizer from ${repo}...`, 5);
  const tokRes = await fetch(`${HF}/${repo}/resolve/main/tokenizer.json`);
  if (!tokRes.ok) throw new Error(`Tokenizer download failed: ${tokRes.status}`);
  const tokenizerJson = await tokRes.json();

  const sfUrl = `${HF}/${repo}/resolve/main/${file}`;
  onProgress("Reading safetensors header...", 12);
  const lenRes = await fetch(sfUrl, { headers: { Range: "bytes=0-7" } });
  if (!lenRes.ok) throw new Error(`Header fetch failed: ${lenRes.status}`);
  const lenBytes = new Uint8Array(await lenRes.arrayBuffer());
  const headerLen = Number(new DataView(lenBytes.buffer).getBigUint64(0, true));
  const hdrRes = await fetch(sfUrl, { headers: { Range: `bytes=8-${8 + headerLen - 1}` } });
  if (!hdrRes.ok) throw new Error(`Header data fetch failed: ${hdrRes.status}`);
  const header = JSON.parse(await hdrRes.text());

  const key = Object.keys(header).find(
    (k) => k !== "__metadata__" && (k.includes("embed_tokens") || k.includes("wte") || k.includes("embed"))
  );
  if (!key) {
    const keys = Object.keys(header).filter((k) => k !== "__metadata__").slice(0, 10);
    throw new Error(`No embedding tensor found. Available: ${keys.join(", ")}`);
  }
  const { shape, dtype, data_offsets } = header[key];
  const [rows, cols] = shape as [number, number];
  const [start, end] = data_offsets as [number, number];
  const dtypeCode = dtype === "F16" ? 1 : dtype === "BF16" ? 3 : dtype === "F32" ? 2 : 0;
  if (dtypeCode === 0) throw new Error(`Unsupported tensor type ${dtype}`);

  onProgress(`Downloading ${key} [${rows} x ${cols}] ${dtype} (${((end - start) / 1024 / 1024).toFixed(0)} MB)...`, 20);
  const dataRes = await fetch(sfUrl, {
    headers: { Range: `bytes=${8 + headerLen + start}-${8 + headerLen + end - 1}` },
  });
  if (!dataRes.ok) throw new Error(`Embedding download failed: ${dataRes.status}`);
  const tensor = new Uint8Array(await dataRes.arrayBuffer());

  const out = new Uint8Array(16 + tensor.byteLength);
  const hv = new DataView(out.buffer);
  hv.setUint32(0, rows, true);
  hv.setUint32(4, cols, true);
  hv.setUint32(8, dtypeCode, true);
  hv.setUint32(12, 0, true);
  out.set(tensor, 16);

  onProgress("Model downloaded", 40);
  return { id: info.id, name: repo, tokenizerJson, embeddings: out.buffer, rows, cols };
}
