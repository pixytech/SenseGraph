export interface BPETokenizer {
  encode(text: string): number[];
  decode(ids: number[]): string;
  vocabSize(): number;
  tokenToId(token: string): number | undefined;
  idToToken(id: number): string | undefined;
}

interface TokenizerJSON {
  model: {
    vocab: Record<string, number>;
    merges: string[];
  };
}

export function createTokenizer(config: TokenizerJSON): BPETokenizer {
  const vocab = config.model.vocab;
  const reverseVocab = new Map<number, string>();
  for (const [token, id] of Object.entries(vocab)) {
    reverseVocab.set(id, token);
  }

  const isWordLevel = config.model.merges.length === 0;

  function encode(text: string): number[] {
    if (isWordLevel) {
      const ids: number[] = [];
      const words = text
        .toLowerCase()
        .split(/[\s,.!?;:'"()\[\]{}]+/)
        .filter((w) => w.length > 0);
      for (const word of words) {
        const id = vocab[word];
        if (id !== undefined) ids.push(id);
      }
      return ids;
    }
    return [];
  }

  function decode(ids: number[]): string {
    return ids.map((id) => reverseVocab.get(id) || `<${id}>`).join(" ");
  }

  return {
    encode,
    decode,
    vocabSize: () => Object.keys(vocab).length,
    tokenToId: (token: string) => vocab[token],
    idToToken: (id: number) => reverseVocab.get(id),
  };
}
