export interface SimilarResult {
  word: string;
  score: number;
  index: number;
}

export interface EngineResult {
  type: "similar" | "search" | "compare" | "generate";
  query: string;
  results: SimilarResult[];
  latencyMs: number;
  operationCount?: number;
  memoryMB?: number;
  engine: "transformer" | "cqe";
  error?: string;
}

export interface BenchmarkSample {
  query: string;
  timestamp: number;
  transformer: { latencyMs: number; resultCount: number } | null;
  cqe: { latencyMs: number; resultCount: number } | null;
}

export interface ModelLoadInfo {
  id: string;
  source: "huggingface" | "local";
  hfRepo?: string;
  hfFile?: string;
}

export type WorkerMessage =
  | { type: "init-with-model"; modelId: string; modelName: string; tokenizerJson: any; embeddings: ArrayBuffer }
  | { type: "similar"; word: string; topK: number }
  | { type: "search"; query: string; topK: number }
  | { type: "compare"; word1: string; word2: string }
  | { type: "generate"; prompt: string; maxTokens: number };

export interface ModelSizeInfo {
  embeddingsMB: number;
  graphMB: number;
  totalMB: number;
  vocabSize: number;
  embedDim: number;
  neighborsIndexed: number;
  attentionLayersMB?: number;
}

export type WorkerResponse =
  | { type: "ready"; vocabSize: number; modelName: string; sizeInfo?: ModelSizeInfo }
  | { type: "progress"; message: string; percent?: number }
  | { type: "result"; data: EngineResult }
  | { type: "error"; message: string };
