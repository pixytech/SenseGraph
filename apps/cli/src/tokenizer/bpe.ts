import * as fs from "fs";

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
  added_tokens?: Array<{ id: number; content: string }>;
}

const BYTE_TO_UNICODE = buildByteToUnicode();

function buildByteToUnicode(): Map<number, string> {
  const map = new Map<number, string>();
  const ranges = [
    [33, 126],
    [161, 172],
    [174, 255],
  ];

  let n = 0;
  const bs: number[] = [];
  for (const [start, end] of ranges) {
    for (let i = start; i <= end; i++) {
      bs.push(i);
    }
  }

  const cs = [...bs];
  for (let b = 0; b < 256; b++) {
    if (!bs.includes(b)) {
      bs.push(b);
      cs.push(256 + n);
      n++;
    }
  }

  for (let i = 0; i < bs.length; i++) {
    map.set(bs[i], String.fromCharCode(cs[i]));
  }
  return map;
}

function bytesToUnicode(text: string): string {
  const encoder = new TextEncoder();
  const bytes = encoder.encode(text);
  return Array.from(bytes)
    .map((b) => BYTE_TO_UNICODE.get(b) || String.fromCharCode(b))
    .join("");
}

function getPairs(word: string[]): Set<string> {
  const pairs = new Set<string>();
  for (let i = 0; i < word.length - 1; i++) {
    pairs.add(word[i] + " " + word[i + 1]);
  }
  return pairs;
}

export function loadTokenizer(tokenizerPath: string): BPETokenizer {
  const raw = fs.readFileSync(tokenizerPath, "utf-8");
  const config: TokenizerJSON = JSON.parse(raw);

  const vocab = config.model.vocab;
  const reverseVocab = new Map<number, string>();
  for (const [token, id] of Object.entries(vocab)) {
    reverseVocab.set(id, token);
  }

  if (config.added_tokens) {
    for (const t of config.added_tokens) {
      if (!(t.content in vocab)) {
        vocab[t.content] = t.id;
      }
      reverseVocab.set(t.id, t.content);
    }
  }

  const mergeRanks = new Map<string, number>();
  for (let i = 0; i < config.model.merges.length; i++) {
    mergeRanks.set(config.model.merges[i], i);
  }

  const cache = new Map<string, string[]>();

  function bpe(token: string): string[] {
    if (cache.has(token)) return cache.get(token)!;

    let word = token.split("");
    if (word.length <= 1) {
      cache.set(token, word);
      return word;
    }

    while (true) {
      const pairs = getPairs(word);
      if (pairs.size === 0) break;

      let bestPair: string | null = null;
      let bestRank = Infinity;

      for (const pair of pairs) {
        const rank = mergeRanks.get(pair);
        if (rank !== undefined && rank < bestRank) {
          bestRank = rank;
          bestPair = pair;
        }
      }

      if (bestPair === null) break;

      const [first, second] = bestPair.split(" ");
      const newWord: string[] = [];
      let i = 0;

      while (i < word.length) {
        const j = word.indexOf(first, i);
        if (j === -1) {
          newWord.push(...word.slice(i));
          break;
        }

        newWord.push(...word.slice(i, j));

        if (j < word.length - 1 && word[j] === first && word[j + 1] === second) {
          newWord.push(first + second);
          i = j + 2;
        } else {
          newWord.push(word[j]);
          i = j + 1;
        }
      }

      word = newWord;
      if (word.length === 1) break;
    }

    cache.set(token, word);
    return word;
  }

  const isWordLevel = config.model.merges.length === 0;

  function encode(text: string): number[] {
    if (isWordLevel) {
      return encodeWordLevel(text);
    }
    return encodeBPE(text);
  }

  function encodeWordLevel(text: string): number[] {
    const ids: number[] = [];
    const words = text
      .toLowerCase()
      .split(/[\s,.!?;:'"()\[\]{}]+/)
      .filter((w) => w.length > 0);

    for (const word of words) {
      const id = vocab[word];
      if (id !== undefined) {
        ids.push(id);
      }
    }
    return ids;
  }

  function encodeBPE(text: string): number[] {
    const ids: number[] = [];

    const chunks = text.match(
      /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+/gu
    );
    if (!chunks) return ids;

    for (const chunk of chunks) {
      const encoded = bytesToUnicode(chunk);
      const tokens = bpe(encoded);

      for (const t of tokens) {
        const id = vocab[t];
        if (id !== undefined) {
          ids.push(id);
        }
      }
    }

    return ids;
  }

  function decode(ids: number[]): string {
    if (isWordLevel) {
      return ids.map((id) => reverseVocab.get(id) || `<${id}>`).join(" ");
    }

    const unicodeToBytes = new Map<string, number>();
    for (const [b, u] of BYTE_TO_UNICODE) {
      unicodeToBytes.set(u, b);
    }

    const tokens = ids.map((id) => reverseVocab.get(id) || "");
    const text = tokens.join("");

    const bytes: number[] = [];
    for (const ch of text) {
      const b = unicodeToBytes.get(ch);
      if (b !== undefined) {
        bytes.push(b);
      } else {
        const encoded = new TextEncoder().encode(ch);
        bytes.push(...encoded);
      }
    }

    return new TextDecoder().decode(new Uint8Array(bytes));
  }

  return {
    encode,
    decode,
    vocabSize: () => Object.keys(vocab).length,
    tokenToId: (token: string) => vocab[token],
    idToToken: (id: number) => reverseVocab.get(id),
  };
}
