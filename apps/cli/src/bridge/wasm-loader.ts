import { createRequire } from "module";

const require = createRequire(import.meta.url);

export interface ResolveResult {
  input: string;
  resolutions: Resolution[];
  operation_count: number;
}

export type Resolution =
  | { Literal: Token }
  | { Unambiguous: { token: Token; sense_id: number; sense_label: string } }
  | {
      Disambiguated: {
        token: Token;
        chosen: SenseCandidate;
        alternatives: SenseCandidate[];
      };
    }
  | { Ambiguous: { token: Token; candidates: SenseCandidate[] } };

export interface Token {
  text: string;
  position: number;
  original: string;
}

export interface SenseCandidate {
  sense_id: number;
  sense_label: string;
  score: number;
  matched_indicators: string[];
}

export interface GraphStats {
  nodes: number;
  edges: number;
}

export interface CQEEngine {
  handle: number;
  addWord(label: string): number;
  addSense(wordId: number, senseLabel: string): number;
  addConcept(label: string): number;
  addContextIndicator(senseId: number, indicator: string): void;
  addIndicators(senseId: number, indicators: string[]): void;
  addRelated(nodeA: number, nodeB: number): void;
  resolve(sentence: string): ResolveResult;
  stats(): GraphStats;
  destroy(): void;
}

export interface SimilarResult {
  index: number;
  score: number;
}

export interface VectorStoreStats {
  count: number;
  embed_dim: number;
}

export interface VectorStoreHandle {
  handle: number;
  loadEmbeddingsF16(data: Uint8Array): void;
  loadEmbeddingsBF16(data: Uint8Array): void;
  loadEmbeddingsF32(data: Uint8Array): void;
  querySimilar(tokenId: number, topK: number): SimilarResult[];
  querySimilarMulti(tokenIds: number[], topK: number): SimilarResult[];
  cosineSim(a: number, b: number): number;
  stats(): VectorStoreStats;
  destroy(): void;
}

let wasmModule: any = null;

function getWasm() {
  if (!wasmModule) {
    wasmModule = require("../../wasm-pkg/sensegraph_core.js");
  }
  return wasmModule;
}

export function createVectorStore(embedDim: number): VectorStoreHandle {
  const wasm = getWasm();
  const handle: number = wasm.create_vector_store(embedDim);

  return {
    handle,

    loadEmbeddingsF16(data: Uint8Array): void {
      wasm.load_embeddings_f16(handle, data);
    },

    loadEmbeddingsBF16(data: Uint8Array): void {
      wasm.load_embeddings_bf16(handle, data);
    },

    loadEmbeddingsF32(data: Uint8Array): void {
      wasm.load_embeddings_f32(handle, data);
    },

    querySimilar(tokenId: number, topK: number): SimilarResult[] {
      return wasm.query_similar(handle, tokenId, topK);
    },

    querySimilarMulti(tokenIds: number[], topK: number): SimilarResult[] {
      return wasm.query_similar_multi(handle, new Uint32Array(tokenIds), topK);
    },

    cosineSim(a: number, b: number): number {
      return wasm.cosine_sim(handle, a, b);
    },

    stats(): VectorStoreStats {
      return wasm.vector_store_stats(handle);
    },

    destroy(): void {
      wasm.drop_vector_store(handle);
    },
  };
}

export function createEngine(): CQEEngine {
  const wasm = getWasm();
  const handle: number = wasm.create_graph();

  return {
    handle,

    addWord(label: string): number {
      return wasm.add_word(handle, label);
    },

    addSense(wordId: number, senseLabel: string): number {
      return wasm.add_sense(handle, wordId, senseLabel);
    },

    addConcept(label: string): number {
      return wasm.add_concept(handle, label);
    },

    addContextIndicator(senseId: number, indicator: string): void {
      wasm.add_context_indicator(handle, senseId, indicator);
    },

    addIndicators(senseId: number, indicators: string[]): void {
      for (const ind of indicators) {
        wasm.add_context_indicator(handle, senseId, ind);
      }
    },

    addRelated(nodeA: number, nodeB: number): void {
      wasm.add_related(handle, nodeA, nodeB);
    },

    resolve(sentence: string): ResolveResult {
      return wasm.resolve(handle, sentence);
    },

    stats(): GraphStats {
      return wasm.graph_stats(handle);
    },

    destroy(): void {
      wasm.drop_graph(handle);
    },
  };
}
