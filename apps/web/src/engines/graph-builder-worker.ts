import init, {
  create_vector_store,
  load_embeddings_f16,
  load_embeddings_bf16,
  load_embeddings_f32,
  init_neighbor_graph,
  build_neighbor_batch,
  export_neighbor_graph,
} from "../../wasm-pkg/sensegraph_core.js";

export type BuilderMessage = {
  type: "init";
  wasmUrl: string;
  embData: ArrayBuffer;
  vocabSize: number;
  embedDim: number;
  rangeStart: number;
  rangeEnd: number;
  k: number;
  batchSize: number;
};

export type BuilderResponse =
  | { type: "progress"; rangeStart: number; done: number; ops: number }
  | { type: "done"; rangeStart: number; rangeEnd: number; graphData: ArrayBuffer }
  | { type: "error"; message: string };

self.onmessage = async (e: MessageEvent<BuilderMessage>) => {
  const msg = e.data;
  if (msg.type !== "init") return;

  try {
    await init({ module_or_path: msg.wasmUrl });

    const embedDim = new DataView(msg.embData).getUint32(4, true);
    const handle = create_vector_store(embedDim);

    // The raw table is only needed while it is copied into the WASM store. The block scope and
    // the null assignment below drop every reference to it, so the memory can be reclaimed
    // instead of staying resident for the whole build.
    {
      const dtypeCode = new DataView(msg.embData).getUint32(8, true);
      const embBytes = new Uint8Array(msg.embData, 16);
      if (dtypeCode === 1) load_embeddings_f16(handle, embBytes);
      else if (dtypeCode === 3) load_embeddings_bf16(handle, embBytes);
      else load_embeddings_f32(handle, embBytes);
    }
    (msg as { embData: ArrayBuffer | null }).embData = null;

    init_neighbor_graph(handle, msg.rangeEnd, msg.k);

    let totalOps = 0;
    for (let b = msg.rangeStart; b < msg.rangeEnd; b += msg.batchSize) {
      const bEnd = Math.min(b + msg.batchSize, msg.rangeEnd);
      const ops = build_neighbor_batch(handle, b, bEnd);
      totalOps += ops;

      (self as any).postMessage({
        type: "progress",
        rangeStart: msg.rangeStart,
        done: bEnd - msg.rangeStart,
        ops: totalOps,
      } as BuilderResponse);
    }

    const graphData = export_neighbor_graph(handle);
    const buf = new Uint8Array(graphData).buffer;

    (self as any).postMessage(
      { type: "done", rangeStart: msg.rangeStart, rangeEnd: msg.rangeEnd, graphData: buf } as BuilderResponse,
      [buf]
    );
  } catch (err: any) {
    (self as any).postMessage({ type: "error", message: err.message || String(err) } as BuilderResponse);
  }
};
