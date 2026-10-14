export interface BPETokenizer {
  // Tokenize text as written.
  encode(text: string): number[];
  // Tokenize a short query. For byte level BPE models the query is treated as
  // running text, so each word maps to its normal mid-sentence token.
  encodeQuery(text: string): number[];
  // Human readable text for a token id (byte level markers decoded).
  tokenText(id: number): string;
  decode(ids: number[]): string;
  vocabSize(): number;
  tokenToId(token: string): number | undefined;
  idToToken(id: number): string | undefined;
}

export interface TokenizerJSON {
  model: {
    vocab: Record<string, number>;
    merges?: (string | [string, string])[];
  };
}

// GPT-2 style byte to printable unicode mapping, used by byte level BPE
// vocabularies (GPT-2, Qwen, SmolLM and similar).
function buildByteMaps(): { byteToChar: string[]; charToByte: Map<string, number> } {
  const bs: number[] = [];
  for (let i = 33; i <= 126; i++) bs.push(i);
  for (let i = 161; i <= 172; i++) bs.push(i);
  for (let i = 174; i <= 255; i++) bs.push(i);
  const cs = bs.slice();
  let n = 0;
  for (let b = 0; b < 256; b++) {
    if (!bs.includes(b)) {
      bs.push(b);
      cs.push(256 + n);
      n++;
    }
  }
  const byteToChar: string[] = new Array(256);
  const charToByte = new Map<string, number>();
  for (let i = 0; i < bs.length; i++) {
    const ch = String.fromCharCode(cs[i]);
    byteToChar[bs[i]] = ch;
    charToByte.set(ch, bs[i]);
  }
  return { byteToChar, charToByte };
}

const PRETOKENIZE = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;

export function createTokenizer(config: TokenizerJSON): BPETokenizer {
  const vocab = config.model.vocab;
  const reverseVocab = new Map<number, string>();
  for (const [token, id] of Object.entries(vocab)) {
    reverseVocab.set(id, token);
  }

  const merges = config.model.merges ?? [];
  // Three families are supported: byte level BPE (GPT-2, Qwen, SmolLM, ...),
  // SentencePiece style BPE (Llama, TinyLlama, Phi, ...) and plain word lists.
  const SP = "\u2581";
  const isByteLevel = merges.length > 0 && vocab["\u0120"] !== undefined;
  const isSentencePiece = !isByteLevel && merges.length > 0 && vocab[SP] !== undefined;
  const isBpe = isByteLevel || isSentencePiece;

  const { byteToChar, charToByte } = buildByteMaps();
  const encoder = new TextEncoder();
  const decoder = new TextDecoder("utf-8", { fatal: false });

  const ranks = new Map<string, number>();
  merges.forEach((m, i) => {
    const key = typeof m === "string" ? m : `${m[0]} ${m[1]}`;
    if (!ranks.has(key)) ranks.set(key, i);
  });

  const cache = new Map<string, number[]>();

  function bpeWord(word: string): number[] {
    const hit = cache.get(word);
    if (hit) return hit;

    let symbols = Array.from(word);
    while (symbols.length > 1) {
      let bestRank = Infinity;
      let bestIdx = -1;
      for (let i = 0; i < symbols.length - 1; i++) {
        const r = ranks.get(`${symbols[i]} ${symbols[i + 1]}`);
        if (r !== undefined && r < bestRank) {
          bestRank = r;
          bestIdx = i;
        }
      }
      if (bestIdx === -1) break;
      symbols = [
        ...symbols.slice(0, bestIdx),
        symbols[bestIdx] + symbols[bestIdx + 1],
        ...symbols.slice(bestIdx + 2),
      ];
    }

    const ids: number[] = [];
    for (const s of symbols) {
      const id = vocab[s];
      if (id !== undefined) ids.push(id);
    }
    cache.set(word, ids);
    return ids;
  }

  function encodeByteLevel(text: string): number[] {
    const ids: number[] = [];
    const pieces = text.match(PRETOKENIZE) ?? [];
    for (const piece of pieces) {
      let mapped = "";
      for (const b of encoder.encode(piece)) mapped += byteToChar[b];
      ids.push(...bpeWord(mapped));
    }
    return ids;
  }

  function encodeSentencePiece(text: string): number[] {
    const norm = SP + text.replace(/ /g, SP);
    const cached = cache.get(norm);
    if (cached) return cached;
    let symbols = Array.from(norm);
    while (symbols.length > 1) {
      let bestRank = Infinity;
      let bestIdx = -1;
      for (let i = 0; i < symbols.length - 1; i++) {
        const r = ranks.get(`${symbols[i]} ${symbols[i + 1]}`);
        if (r !== undefined && r < bestRank) {
          bestRank = r;
          bestIdx = i;
        }
      }
      if (bestIdx === -1) break;
      symbols = [
        ...symbols.slice(0, bestIdx),
        symbols[bestIdx] + symbols[bestIdx + 1],
        ...symbols.slice(bestIdx + 2),
      ];
    }
    const ids: number[] = [];
    for (const sym of symbols) {
      const id = vocab[sym];
      if (id !== undefined) {
        ids.push(id);
        continue;
      }
      // Byte fallback for symbols that are not in the vocabulary.
      for (const b of encoder.encode(sym)) {
        const hex = `<0x${b.toString(16).toUpperCase().padStart(2, "0")}>`;
        const bid = vocab[hex];
        if (bid !== undefined) ids.push(bid);
      }
    }
    if (norm.length < 200) cache.set(norm, ids);
    return ids;
  }

  function encodeWordLevel(text: string): number[] {
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

  function encode(text: string): number[] {
    if (isByteLevel) return encodeByteLevel(text);
    if (isSentencePiece) return encodeSentencePiece(text);
    return encodeWordLevel(text);
  }

  function encodeQuery(text: string): number[] {
    const t = text.trim();
    if (!t) return [];
    if (isByteLevel) return encodeByteLevel(" " + t);
    // SentencePiece adds the leading marker itself.
    if (isSentencePiece) return encodeSentencePiece(t);
    return encodeWordLevel(t);
  }

  function tokenText(id: number): string {
    const token = reverseVocab.get(id);
    if (token === undefined) return `<${id}>`;
    if (isSentencePiece) {
      const m = /^<0x([0-9A-Fa-f]{2})>$/.exec(token);
      if (m) return decoder.decode(new Uint8Array([parseInt(m[1], 16)]));
      return token.split(SP).join(" ");
    }
    if (!isByteLevel) return token;
    const bytes: number[] = [];
    for (const ch of token) {
      const b = charToByte.get(ch);
      if (b === undefined) return token;
      bytes.push(b);
    }
    return decoder.decode(new Uint8Array(bytes));
  }

  function decode(ids: number[]): string {
    return ids.map(tokenText).join(isBpe ? "" : " ");
  }

  return {
    encode,
    encodeQuery,
    tokenText,
    decode,
    vocabSize: () => Object.keys(vocab).length,
    tokenToId: (token: string) => vocab[token],
    idToToken: (id: number) => reverseVocab.get(id),
  };
}
