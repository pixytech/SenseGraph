export interface ModelInfo {
  id: string;
  name: string;
  // Approximate download: only the embedding rows that are used.
  size: string;
  params: string;
  category: "Code" | "General";
  source: "huggingface" | "local";
  hfRepo: string;
  hfFile: string;
  // Only the first rows of the embedding table are used (the most frequent
  // tokens in BPE vocabularies). Keeps the neighbor graph build practical.
  maxVocab?: number;
  speed: 1 | 2 | 3;
}

export const MODELS: ModelInfo[] = [
  {
    id: "qwen25-coder-05b",
    name: "Qwen2.5-Coder 0.5B",
    size: "~90 MB",
    params: "0.5B",
    category: "Code",
    source: "huggingface",
    hfRepo: "Qwen/Qwen2.5-Coder-0.5B",
    hfFile: "model.safetensors",
    maxVocab: 50000,
    speed: 3,
  },
  {
    id: "qwen25-coder-15b",
    name: "Qwen2.5-Coder 1.5B",
    size: "~155 MB",
    params: "1.5B",
    category: "Code",
    source: "huggingface",
    hfRepo: "Qwen/Qwen2.5-Coder-1.5B",
    hfFile: "model.safetensors",
    maxVocab: 50000,
    speed: 2,
  },
  {
    id: "smollm2-135m",
    name: "SmolLM2 135M",
    size: "~57 MB",
    params: "135M",
    category: "General",
    source: "huggingface",
    hfRepo: "HuggingFaceTB/SmolLM2-135M",
    hfFile: "model.safetensors",
    speed: 3,
  },
  {
    id: "smollm2-360m",
    name: "SmolLM2 360M",
    size: "~95 MB",
    params: "360M",
    category: "General",
    source: "huggingface",
    hfRepo: "HuggingFaceTB/SmolLM2-360M",
    hfFile: "model.safetensors",
    speed: 2,
  },
  {
    id: "qwen25-05b",
    name: "Qwen 2.5 0.5B",
    size: "~90 MB",
    params: "0.5B",
    category: "General",
    source: "huggingface",
    hfRepo: "Qwen/Qwen2.5-0.5B",
    hfFile: "model.safetensors",
    maxVocab: 50000,
    speed: 2,
  },
  {
    id: "tinyllama-1b",
    name: "TinyLlama 1.1B Chat",
    size: "~130 MB",
    params: "1.1B",
    category: "General",
    source: "huggingface",
    hfRepo: "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    hfFile: "model.safetensors",
    speed: 1,
  },
  {
    id: "phi-35-mini",
    name: "Phi 3.5 Mini",
    size: "~200 MB",
    params: "3.8B",
    category: "General",
    source: "huggingface",
    hfRepo: "microsoft/Phi-3.5-mini-instruct",
    hfFile: "model-00001-of-00002.safetensors",
    speed: 1,
  },
];

export function getSpeedIcons(speed: number): string {
  return speed >= 3 ? "fast" : speed === 2 ? "medium" : "slow";
}
