import * as fs from "fs";
import * as path from "path";
import { execSync } from "child_process";

const MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0";
const HF_BASE = "https://huggingface.co";
const DATA_DIR = path.resolve(import.meta.dirname!, "..", "..", "data");
const EMBEDDING_TENSOR = "model.embed_tokens.weight";

interface SafetensorEntry {
  dtype: string;
  shape: number[];
  data_offsets: [number, number];
}

type SafetensorsHeader = Record<string, SafetensorEntry>;

function curlBytes(url: string, rangeHeader?: string): Buffer {
  const args = ["-sL"];
  if (rangeHeader) {
    args.push("-H", `Range: ${rangeHeader}`);
  }
  args.push(url);
  return execSync(`curl ${args.map((a) => `"${a}"`).join(" ")}`, {
    encoding: "buffer",
    maxBuffer: 200 * 1024 * 1024,
  });
}

function ensureDir(dir: string) {
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }
}

function downloadTokenizer() {
  const tokenizerPath = path.join(DATA_DIR, "tokenizer.json");
  if (fs.existsSync(tokenizerPath)) {
    console.log("  tokenizer.json already exists, skipping download");
    return;
  }

  const url = `${HF_BASE}/${MODEL_ID}/resolve/main/tokenizer.json`;
  console.log(`  Downloading tokenizer.json...`);
  const data = curlBytes(url);
  fs.writeFileSync(tokenizerPath, data);
  console.log(`  Saved tokenizer.json (${(data.byteLength / 1024).toFixed(0)} KB)`);
}

function getSafetensorsHeader(): { header: SafetensorsHeader; headerSize: number } {
  const url = `${HF_BASE}/${MODEL_ID}/resolve/main/model.safetensors`;

  console.log("  Reading safetensors header size...");
  const headerSizeBytes = curlBytes(url, "bytes=0-7");
  const view = new DataView(
    headerSizeBytes.buffer,
    headerSizeBytes.byteOffset,
    8
  );
  const headerLen = Number(view.getBigUint64(0, true));
  console.log(`  Header length: ${headerLen} bytes`);

  console.log("  Reading safetensors header...");
  const headerBytes = curlBytes(url, `bytes=8-${8 + headerLen - 1}`);
  const headerText = headerBytes.toString("utf-8");
  const header = JSON.parse(headerText) as SafetensorsHeader;

  return { header, headerSize: 8 + headerLen };
}

function downloadEmbeddingTensor(header: SafetensorsHeader, headerSize: number) {
  const embeddingsPath = path.join(DATA_DIR, "embeddings.bin");
  if (fs.existsSync(embeddingsPath)) {
    console.log("  embeddings.bin already exists, skipping download");
    return;
  }

  const entry = header[EMBEDDING_TENSOR];
  if (!entry) {
    const keys = Object.keys(header).filter((k) => k !== "__metadata__");
    console.log("  Available tensors:", keys.slice(0, 10).join(", "), "...");
    throw new Error(`Tensor "${EMBEDDING_TENSOR}" not found in safetensors`);
  }

  console.log(`  Embedding tensor: ${entry.dtype}, shape [${entry.shape.join(", ")}]`);
  const [vocabSize, embedDim] = entry.shape;
  const [start, end] = entry.data_offsets;
  const byteLen = end - start;
  console.log(`  Data: ${(byteLen / 1024 / 1024).toFixed(1)} MB at offset ${start}`);

  const fileStart = headerSize + start;
  const fileEnd = headerSize + end - 1;
  const url = `${HF_BASE}/${MODEL_ID}/resolve/main/model.safetensors`;

  console.log(`  Downloading embedding tensor (${(byteLen / 1024 / 1024).toFixed(1)} MB)...`);
  console.log(`  This may take a minute...`);
  const rawData = curlBytes(url, `bytes=${fileStart}-${fileEnd}`);
  console.log(`  Downloaded ${rawData.byteLength} bytes`);

  const dtypeCode = entry.dtype === "F16" ? 1 : entry.dtype === "BF16" ? 3 : 2;

  const headerBuf = Buffer.alloc(16);
  headerBuf.writeUInt32LE(vocabSize, 0);
  headerBuf.writeUInt32LE(embedDim, 4);
  headerBuf.writeUInt32LE(dtypeCode, 8);
  headerBuf.writeUInt32LE(0, 12);

  const out = Buffer.concat([headerBuf, rawData]);
  fs.writeFileSync(embeddingsPath, out);
  console.log(`  Saved embeddings.bin (${(out.byteLength / 1024 / 1024).toFixed(1)} MB)`);
  console.log(`  Vocab: ${vocabSize}, Dim: ${embedDim}, Dtype: ${entry.dtype}`);
}

function main() {
  console.log("SenseGraph Embedding Extractor");
  console.log(`Model: ${MODEL_ID}\n`);

  ensureDir(DATA_DIR);

  console.log("[1/2] Tokenizer");
  downloadTokenizer();

  console.log("\n[2/2] Embeddings");
  const { header, headerSize } = getSafetensorsHeader();
  downloadEmbeddingTensor(header, headerSize);

  console.log("\nDone! Data saved to:", DATA_DIR);
}

main();
