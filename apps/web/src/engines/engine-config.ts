export interface EngineConfig {
  contextualWSD: boolean;
  positionalWeighting: boolean;
  phraseDetection: boolean;
  retrievalGeneration: boolean;
}

export const DEFAULT_CONFIG: EngineConfig = {
  contextualWSD: true,
  positionalWeighting: true,
  phraseDetection: true,
  retrievalGeneration: true,
};

export const FEATURE_INFO: Record<keyof EngineConfig, { label: string; description: string }> = {
  contextualWSD: {
    label: "Contextual Disambiguation",
    description: "Use embedding neighbors as context indicators to disambiguate words (replaces attention's context-dependent meaning)",
  },
  positionalWeighting: {
    label: "Positional Weighting",
    description: "Weight later words in a query more heavily — they tend to be more specific (replaces attention's positional encoding)",
  },
  phraseDetection: {
    label: "Phrase Detection",
    description: "Detect common multi-word phrases and use compound embeddings instead of naive averaging (replaces attention's cross-token composition)",
  },
  retrievalGeneration: {
    label: "Retrieval Generation",
    description: "Chain retrieved tokens into coherent responses using graph traversal (replaces attention's autoregressive generation)",
  },
};
