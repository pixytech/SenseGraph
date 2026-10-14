import * as fs from "fs";
import * as path from "path";

const DATA_DIR = path.resolve(import.meta.dirname!, "..", "..", "data");

const SAFETENSORS_PATHS = [
  path.join(DATA_DIR, "model.safetensors"),
  path.resolve(import.meta.dirname!, "..", "..", "model.safetensors"),
];

function findSafetensors(): string | null {
  for (const p of SAFETENSORS_PATHS) {
    if (fs.existsSync(p)) return p;
  }
  return null;
}

function main() {
  console.log("SenseGraph Local Embedding Extractor\n");

  if (!fs.existsSync(DATA_DIR)) {
    fs.mkdirSync(DATA_DIR, { recursive: true });
  }

  const embPath = path.join(DATA_DIR, "embeddings.bin");
  if (fs.existsSync(embPath)) {
    console.log(`embeddings.bin already exists at ${embPath}`);
    console.log("Delete it to re-extract.\n");
    return;
  }

  const sfPath = findSafetensors();
  if (!sfPath) {
    console.log("Place model.safetensors in one of these locations:");
    SAFETENSORS_PATHS.forEach((p) => console.log(`  ${p}`));
    console.log("\nDownload from: https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0");
    console.log("(Download model.safetensors and tokenizer.json)\n");
    return;
  }

  console.log(`Found: ${sfPath}`);

  const fd = fs.openSync(sfPath, "r");
  const headerSizeBuf = Buffer.alloc(8);
  fs.readSync(fd, headerSizeBuf, 0, 8, 0);
  const headerLen = Number(headerSizeBuf.readBigUInt64LE(0));
  console.log(`Header length: ${headerLen} bytes`);

  const headerBuf = Buffer.alloc(headerLen);
  fs.readSync(fd, headerBuf, 0, headerLen, 8);
  const header = JSON.parse(headerBuf.toString("utf-8"));

  const tensorName = "model.embed_tokens.weight";
  const entry = header[tensorName];
  if (!entry) {
    const keys = Object.keys(header).filter((k) => k !== "__metadata__");
    console.log("Available tensors:", keys.slice(0, 10).join(", "));
    console.error(`Tensor "${tensorName}" not found!`);
    fs.closeSync(fd);
    return;
  }

  const [vocabSize, embedDim] = entry.shape;
  const [start, end] = entry.data_offsets;
  const byteLen = end - start;
  console.log(`Tensor: ${entry.dtype}, shape [${vocabSize}, ${embedDim}]`);
  console.log(`Data: ${(byteLen / 1024 / 1024).toFixed(1)} MB`);

  const tensorBuf = Buffer.alloc(byteLen);
  fs.readSync(fd, tensorBuf, 0, byteLen, 8 + headerLen + start);
  fs.closeSync(fd);

  const dtypeCode = entry.dtype === "F16" ? 1 : entry.dtype === "BF16" ? 3 : 2;

  const outHeader = Buffer.alloc(16);
  outHeader.writeUInt32LE(vocabSize, 0);
  outHeader.writeUInt32LE(embedDim, 4);
  outHeader.writeUInt32LE(dtypeCode, 8);
  outHeader.writeUInt32LE(0, 12);

  const out = Buffer.concat([outHeader, tensorBuf]);
  fs.writeFileSync(embPath, out);

  const sizeMb = out.byteLength / 1024 / 1024;
  console.log(`\nSaved: ${embPath} (${sizeMb.toFixed(1)} MB)`);
  console.log(`Vocab: ${vocabSize}, Dim: ${embedDim}, Dtype: ${entry.dtype}`);
  console.log("\nDone! Now run: npm start");
}

main();
