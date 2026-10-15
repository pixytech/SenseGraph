import init, {
  create_vector_store,
  drop_vector_store,
  load_embeddings_f16,
  load_embeddings_bf16,
  load_embeddings_f32,
  init_neighbor_graph,
  build_neighbor_batch,
  export_neighbor_graph,
} from "../../wasm-pkg/sensegraph_core.js";

// A builder owns a range of query rows. The rest of the embedding table is streamed to it one
// tile at a time, so it never holds the whole table. For every block of its query rows and every
// tile it builds a small vector store (block rows plus tile rows), asks WASM for the nearest
// neighbors inside that store, and merges them into a running top K per row. After all tiles
// have been seen, each row holds the same top K as an all-pairs search over the full table.

export type BuilderMessage =
  | {
      type: "init";
      wasmUrl: string;
      embedDim: number;
      dtypeCode: number;
      k: number;
      rangeStart: number;
      rangeEnd: number;
      blockRows: number;
      rows: ArrayBuffer;
    }
  | { type: "tile"; tileStart: number; tileEnd: number; rows: ArrayBuffer }
  | { type: "finish" };

export type BuilderResponse =
  | { type: "ready" }
  | { type: "tileDone"; tileStart: number; ops: number }
  | { type: "done"; rangeStart: number; rangeEnd: number; ids: ArrayBuffer; scores: ArrayBuffer }
  | { type: "error"; message: string };

let embedDim = 0;
let dtypeCode = 2;
let k = 20;
let rangeStart = 0;
let rangeEnd = 0;
let blockRows = 512;
let rowBytes = 0;
let queryBytes: Uint8Array = new Uint8Array(0);
let accIds: Uint32Array = new Uint32Array(0);
let accScores: Float32Array = new Float32Array(0);
let accCount: Uint8Array = new Uint8Array(0);

function post(msg: BuilderResponse, transfer: Transferable[] = []) {
  (self as any).postMessage(msg, transfer);
}

function loadInto(handle: number, bytes: Uint8Array) {
  if (dtypeCode === 1) load_embeddings_f16(handle, bytes);
  else if (dtypeCode === 3) load_embeddings_bf16(handle, bytes);
  else load_embeddings_f32(handle, bytes);
}

function insert(localRow: number, gid: number, score: number) {
  const base = localRow * k;
  const count = accCount[localRow];
  for (let j = 0; j < count; j++) if (accIds[base + j] === gid) return;
  if (count < k) {
    accIds[base + count] = gid;
    accScores[base + count] = score;
    accCount[localRow] = count + 1;
    return;
  }
  let minJ = 0;
  for (let j = 1; j < k; j++) if (accScores[base + j] < accScores[base + minJ]) minJ = j;
  if (score > accScores[base + minJ]) {
    accIds[base + minJ] = gid;
    accScores[base + minJ] = score;
  }
}

function processTile(tileStart: number, tileEnd: number, rows: ArrayBuffer): number {
  const tile = new Uint8Array(rows);
  const rangeLen = rangeEnd - rangeStart;
  let ops = 0;

  for (let b = 0; b < rangeLen; b += blockRows) {
    const bStart = rangeStart + b;
    const bEnd = Math.min(bStart + blockRows, rangeEnd);
    const blen = bEnd - bStart;

    // Tile rows that are not already in the block, so the store holds no duplicate vectors.
    const lo = Math.max(tileStart, bStart);
    const hi = Math.min(tileEnd, bEnd);
    const segs: Array<[number, number]> =
      lo < hi ? [[tileStart, lo], [hi, tileEnd]] : [[tileStart, tileEnd]];
    const live = segs.filter(([s, e]) => e > s);
    let other = 0;
    for (const [s, e] of live) other += e - s;
    if (other === 0) continue;

    const total = blen + other;
    const buf = new Uint8Array(total * rowBytes);
    buf.set(queryBytes.subarray(b * rowBytes, (b + blen) * rowBytes), 0);
    const gids = new Uint32Array(total);
    for (let i = 0; i < blen; i++) gids[i] = bStart + i;
    let off = blen * rowBytes;
    let gi = blen;
    for (const [s, e] of live) {
      buf.set(tile.subarray((s - tileStart) * rowBytes, (e - tileStart) * rowBytes), off);
      off += (e - s) * rowBytes;
      for (let g = s; g < e; g++) gids[gi++] = g;
    }

    const handle = create_vector_store(embedDim);
    let out: Uint8Array;
    try {
      loadInto(handle, buf);
      init_neighbor_graph(handle, total, k);
      ops += build_neighbor_batch(handle, 0, blen);
      out = new Uint8Array(export_neighbor_graph(handle));
    } finally {
      drop_vector_store(handle);
    }

    const dv = new DataView(out.buffer, out.byteOffset, out.byteLength);
    const valid = Math.min(k, total - 1);
    for (let i = 0; i < blen; i++) {
      const rowOff = 8 + i * k * 8;
      const queryGid = bStart + i;
      for (let j = 0; j < valid; j++) {
        const gid = gids[dv.getUint32(rowOff + j * 4, true)];
        if (gid === queryGid) continue;
        insert(b + i, gid, dv.getFloat32(rowOff + k * 4 + j * 4, true));
      }
    }
  }
  return ops;
}

function finish() {
  const rangeLen = rangeEnd - rangeStart;
  const ids = new Uint32Array(rangeLen * k);
  const scores = new Float32Array(rangeLen * k);
  const order: number[] = [];
  for (let r = 0; r < rangeLen; r++) {
    const base = r * k;
    const count = accCount[r];
    order.length = 0;
    for (let j = 0; j < count; j++) order.push(j);
    order.sort((a, c) => accScores[base + c] - accScores[base + a]);
    for (let j = 0; j < count; j++) {
      ids[base + j] = accIds[base + order[j]];
      scores[base + j] = accScores[base + order[j]];
    }
  }
  post({ type: "done", rangeStart, rangeEnd, ids: ids.buffer, scores: scores.buffer }, [ids.buffer, scores.buffer]);
}

self.onmessage = async (e: MessageEvent<BuilderMessage>) => {
  const msg = e.data;
  try {
    if (msg.type === "init") {
      await init({ module_or_path: msg.wasmUrl });
      embedDim = msg.embedDim;
      dtypeCode = msg.dtypeCode;
      k = msg.k;
      rangeStart = msg.rangeStart;
      rangeEnd = msg.rangeEnd;
      blockRows = msg.blockRows;
      rowBytes = embedDim * (dtypeCode === 2 ? 4 : 2);
      queryBytes = new Uint8Array(msg.rows);
      const n = rangeEnd - rangeStart;
      accIds = new Uint32Array(n * k);
      accScores = new Float32Array(n * k);
      accCount = new Uint8Array(n);
      post({ type: "ready" });
    } else if (msg.type === "tile") {
      const ops = processTile(msg.tileStart, msg.tileEnd, msg.rows);
      post({ type: "tileDone", tileStart: msg.tileStart, ops });
    } else if (msg.type === "finish") {
      finish();
    }
  } catch (err: any) {
    post({ type: "error", message: err?.message || String(err) });
  }
};
