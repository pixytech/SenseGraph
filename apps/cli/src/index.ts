import * as fs from "fs";
import * as path from "path";
import { createEngine, createVectorStore, type VectorStoreHandle } from "./bridge/wasm-loader.js";
import { seedGraph } from "./knowledge/seed.js";
import { startRepl } from "./cli/repl.js";
import { loadTokenizer, type BPETokenizer } from "./tokenizer/bpe.js";

const DATA_DIR = path.resolve(import.meta.dirname!, "..", "..", "..", "data");

function loadEmbeddings(): { vectors: VectorStoreHandle; tokenizer: BPETokenizer } | null {
  const embPath = path.join(DATA_DIR, "embeddings.bin");
  const tokPath = path.join(DATA_DIR, "tokenizer.json");

  if (!fs.existsSync(embPath) || !fs.existsSync(tokPath)) {
    console.log("  No embeddings found. Run: npx tsx scripts/extract-embeddings.ts");
    return null;
  }

  console.log("  Loading tokenizer...");
  const tokenizer = loadTokenizer(tokPath);

  console.log("  Loading embeddings...");
  const raw = fs.readFileSync(embPath);
  const header = Buffer.from(raw.buffer, raw.byteOffset, 16);
  const vocabSize = header.readUInt32LE(0);
  const embedDim = header.readUInt32LE(4);
  const dtypeCode = header.readUInt32LE(8);

  const dtypeNames: Record<number, string> = { 1: "F16", 2: "F32", 3: "BF16" };
  console.log(`  Vocab: ${vocabSize}, Dim: ${embedDim}, Dtype: ${dtypeNames[dtypeCode] ?? "unknown"}`);

  const vectors = createVectorStore(embedDim);
  const embData = new Uint8Array(raw.buffer, raw.byteOffset + 16);

  if (dtypeCode === 1) {
    vectors.loadEmbeddingsF16(embData);
  } else if (dtypeCode === 3) {
    vectors.loadEmbeddingsBF16(embData);
  } else {
    vectors.loadEmbeddingsF32(embData);
  }

  const stats = vectors.stats();
  console.log(`  Loaded ${stats.count} vectors of dimension ${stats.embed_dim}`);

  return { vectors, tokenizer };
}

const engine = createEngine();
seedGraph(engine);

const embResult = loadEmbeddings();
startRepl(engine, embResult?.vectors ?? null, embResult?.tokenizer ?? null);
