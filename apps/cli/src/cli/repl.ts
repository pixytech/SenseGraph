import * as readline from "readline";
import type {
  SenseGraphEngine,
  ResolveResult,
  Resolution,
  VectorStoreHandle,
} from "../bridge/wasm-loader.js";
import type { BPETokenizer } from "../tokenizer/bpe.js";

const RESET = "\x1b[0m";
const BOLD = "\x1b[1m";
const DIM = "\x1b[2m";
const GREEN = "\x1b[32m";
const YELLOW = "\x1b[33m";
const CYAN = "\x1b[36m";
const RED = "\x1b[31m";
const MAGENTA = "\x1b[35m";

function getResolutionKind(
  r: Resolution
): "Literal" | "Unambiguous" | "Disambiguated" | "Ambiguous" {
  if ("Literal" in r) return "Literal";
  if ("Unambiguous" in r) return "Unambiguous";
  if ("Disambiguated" in r) return "Disambiguated";
  return "Ambiguous";
}

function formatResult(result: ResolveResult): string {
  const lines: string[] = [];

  lines.push(`${DIM}Input:${RESET} ${result.input}`);
  lines.push("");

  const tokens = result.resolutions.map((r) => {
    const kind = getResolutionKind(r);
    if (kind === "Literal") {
      const t = (r as { Literal: any }).Literal;
      return `${DIM}${t.original}${RESET}`;
    }
    if (kind === "Unambiguous") {
      const data = (r as { Unambiguous: any }).Unambiguous;
      return `${GREEN}${data.token.original}${RESET}`;
    }
    if (kind === "Disambiguated") {
      const data = (r as { Disambiguated: any }).Disambiguated;
      return `${CYAN}${BOLD}${data.token.original}${RESET}`;
    }
    const data = (r as { Ambiguous: any }).Ambiguous;
    return `${YELLOW}${data.token.original}?${RESET}`;
  });

  lines.push(`${DIM}Tokens:${RESET}  ${tokens.join("  ")}`);
  lines.push("");

  for (const r of result.resolutions) {
    const kind = getResolutionKind(r);

    if (kind === "Disambiguated") {
      const data = (r as { Disambiguated: any }).Disambiguated;
      const matched = data.chosen.matched_indicators.join(", ");
      lines.push(
        `  ${CYAN}${BOLD}${data.token.text}${RESET} ${DIM}→${RESET} ${GREEN}${data.chosen.sense_label}${RESET}  ${DIM}(score: ${data.chosen.score})${RESET}`
      );
      lines.push(`    ${DIM}matched: ${matched}${RESET}`);
      if (data.alternatives.length > 0) {
        for (const alt of data.alternatives) {
          lines.push(
            `    ${DIM}rejected: ${alt.sense_label} (score: ${alt.score})${RESET}`
          );
        }
      }
    } else if (kind === "Ambiguous") {
      const data = (r as { Ambiguous: any }).Ambiguous;
      lines.push(
        `  ${YELLOW}${BOLD}${data.token.text}${RESET} ${DIM}→${RESET} ${RED}AMBIGUOUS${RESET}`
      );
      for (const c of data.candidates) {
        lines.push(
          `    ${DIM}candidate: ${c.sense_label} (score: ${c.score})${RESET}`
        );
      }
    } else if (kind === "Unambiguous") {
      const data = (r as { Unambiguous: any }).Unambiguous;
      lines.push(
        `  ${GREEN}${data.token.text}${RESET} ${DIM}→ ${data.sense_label} (single sense)${RESET}`
      );
    }
  }

  lines.push("");
  lines.push(
    `${DIM}Operations: ${result.operation_count} hash lookups  (vs ~${Math.pow(tokens.length, 2) * 512} multiply-adds for attention)${RESET}`
  );

  return lines.join("\n");
}

export function startRepl(
  engine: SenseGraphEngine,
  vectors: VectorStoreHandle | null,
  tokenizer: BPETokenizer | null
): void {
  const stats = engine.stats();
  console.log(`\n${BOLD}${MAGENTA}SenseGraph v0.2.0${RESET}`);
  console.log(`${DIM}AI inference without attention layers${RESET}\n`);
  console.log(`${DIM}Graph loaded: ${stats.nodes} nodes, ${stats.edges} edges${RESET}`);

  if (vectors && tokenizer) {
    const vs = vectors.stats();
    console.log(
      `${GREEN}Embeddings loaded: ${vs.count} vectors × ${vs.embed_dim} dims (TinyLlama)${RESET}`
    );
    console.log(
      `${DIM}Commands: similar <word>, compare <w1> <w2>, search <query>${RESET}`
    );
  } else {
    console.log(
      `${YELLOW}No embeddings loaded. Run: npx tsx scripts/extract-embeddings.ts${RESET}`
    );
  }

  console.log(
    `${DIM}Type a sentence to resolve word senses. Type "help" for commands.${RESET}\n`
  );

  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
    prompt: `${BOLD}SenseGraph>${RESET} `,
  });

  rl.prompt();

  rl.on("line", (line) => {
    const input = line.trim();

    if (!input) {
      rl.prompt();
      return;
    }

    if (input === "quit" || input === "exit") {
      console.log(`\n${DIM}Goodbye.${RESET}`);
      engine.destroy();
      vectors?.destroy();
      rl.close();
      return;
    }

    if (input === "stats") {
      const s = engine.stats();
      console.log(`\n${DIM}Graph — Nodes: ${s.nodes}, Edges: ${s.edges}${RESET}`);
      if (vectors) {
        const vs = vectors.stats();
        console.log(`${DIM}Vectors — Count: ${vs.count}, Dim: ${vs.embed_dim}${RESET}`);
      }
      console.log("");
      rl.prompt();
      return;
    }

    if (input === "help") {
      console.log(`\n${DIM}Commands:${RESET}`);
      console.log(`  ${DIM}<sentence>         — resolve word senses via graph${RESET}`);
      console.log(`  ${DIM}similar <word>     — nearest neighbors from LLM embeddings${RESET}`);
      console.log(`  ${DIM}compare <w1> <w2>  — cosine similarity between two words${RESET}`);
      console.log(`  ${DIM}search <query>     — semantic search across vocabulary${RESET}`);
      console.log(`  ${DIM}stats              — show graph and vector statistics${RESET}`);
      console.log(`  ${DIM}help               — show this help${RESET}`);
      console.log(`  ${DIM}quit               — exit${RESET}\n`);
      rl.prompt();
      return;
    }

    if (input.startsWith("similar ") || input.startsWith("compare ") || input.startsWith("search ")) {
      if (!vectors || !tokenizer) {
        console.log(`\n${YELLOW}Embeddings not loaded. Run: npx tsx scripts/extract-embeddings.ts${RESET}\n`);
        rl.prompt();
        return;
      }

      if (input.startsWith("similar ")) {
        handleSimilar(input.slice(8).trim(), vectors, tokenizer);
      } else if (input.startsWith("compare ")) {
        const parts = input.slice(8).trim().split(/\s+/);
        if (parts.length >= 2) {
          handleCompare(parts[0], parts[1], vectors, tokenizer);
        } else {
          console.log(`${RED}Usage: compare <word1> <word2>${RESET}\n`);
        }
      } else {
        handleSearch(input.slice(7).trim(), vectors, tokenizer);
      }
      rl.prompt();
      return;
    }

    try {
      const result = engine.resolve(input);
      console.log(`\n${formatResult(result)}\n`);
    } catch (err) {
      console.error(`${RED}Error: ${err}${RESET}\n`);
    }

    rl.prompt();
  });

  rl.on("close", () => {
    process.exit(0);
  });
}

function handleSimilar(
  word: string,
  vectors: VectorStoreHandle,
  tokenizer: BPETokenizer
): void {
  const t0 = performance.now();
  const ids = tokenizer.encode(word);

  if (ids.length === 0) {
    console.log(`${RED}Unknown word: "${word}"${RESET}\n`);
    return;
  }

  const results =
    ids.length === 1
      ? vectors.querySimilar(ids[0], 15)
      : vectors.querySimilarMulti(ids, 15);

  const elapsed = (performance.now() - t0).toFixed(1);

  const tokenStr =
    ids.length === 1
      ? `token #${ids[0]}`
      : `${ids.length} tokens [${ids.join(", ")}]`;

  console.log(
    `\n${DIM}similar "${word}" (${tokenStr})${RESET}  ${DIM}${elapsed}ms${RESET}\n`
  );

  for (let i = 0; i < results.length; i++) {
    const r = results[i];
    const token = tokenizer.idToToken(r.index) || `<${r.index}>`;
    const bar = "█".repeat(Math.round(r.score * 20));
    const scoreStr = r.score.toFixed(4);
    console.log(
      `  ${DIM}${String(i + 1).padStart(2)}.${RESET} ${GREEN}${token.padEnd(20)}${RESET} ${CYAN}${scoreStr}${RESET}  ${DIM}${bar}${RESET}`
    );
  }
  console.log("");
}

function handleCompare(
  word1: string,
  word2: string,
  vectors: VectorStoreHandle,
  tokenizer: BPETokenizer
): void {
  const ids1 = tokenizer.encode(word1);
  const ids2 = tokenizer.encode(word2);

  if (ids1.length === 0 || ids2.length === 0) {
    console.log(`${RED}Unknown word(s)${RESET}\n`);
    return;
  }

  if (ids1.length === 1 && ids2.length === 1) {
    const sim = vectors.cosineSim(ids1[0], ids2[0]);
    const bar = "█".repeat(Math.round(Math.max(0, sim) * 30));
    console.log(
      `\n  ${GREEN}${word1}${RESET} ↔ ${GREEN}${word2}${RESET}  ${CYAN}${BOLD}${sim.toFixed(4)}${RESET}  ${DIM}${bar}${RESET}\n`
    );
  } else {
    console.log(
      `${YELLOW}Multi-token comparison not yet supported. Use single-token words.${RESET}\n`
    );
  }
}

function handleSearch(
  query: string,
  vectors: VectorStoreHandle,
  tokenizer: BPETokenizer
): void {
  const t0 = performance.now();
  const ids = tokenizer.encode(query);

  if (ids.length === 0) {
    console.log(`${RED}Could not tokenize query${RESET}\n`);
    return;
  }

  const results = vectors.querySimilarMulti(ids, 40);
  const elapsed = (performance.now() - t0).toFixed(1);

  const STOP = new Set([
    "the", "a", "an", "is", "was", "are", "were", "be", "been", "being",
    "in", "on", "at", "to", "for", "of", "with", "by", "from", "and",
    "or", "but", "not", "this", "that", "it", "its", "as", "if", "so",
    "than", "which", "who", "whom", "what", "where", "when", "how",
    "all", "each", "both", "few", "more", "most", "some", "any", "no",
    "has", "have", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "am", "he", "she", "they",
    "we", "you", "i", "my", "his", "her", "our", "their", "your", "me",
    ",", ".", "!", "?", ";", ":", "'", "\"", "-", "--",
    "also", "just", "then", "now", "here", "there", "very", "still",
    "already", "even", "while", "since", "about", "after", "before",
    "between", "into", "through", "during", "under", "over", "same",
    "other", "such", "only", "own", "well", "much", "last", "first",
    "part", "because",
  ]);

  const filtered = results.filter((r) => {
    const token = tokenizer.idToToken(r.index) || "";
    return token.length > 1 && !STOP.has(token.toLowerCase());
  }).slice(0, 20);

  console.log(
    `\n${DIM}search "${query}" (${ids.length} tokens)${RESET}  ${DIM}${elapsed}ms${RESET}\n`
  );

  for (let i = 0; i < filtered.length; i++) {
    const r = filtered[i];
    const token = tokenizer.idToToken(r.index) || `<${r.index}>`;
    const scoreStr = r.score.toFixed(4);
    console.log(
      `  ${DIM}${String(i + 1).padStart(2)}.${RESET} ${GREEN}${token.padEnd(20)}${RESET} ${CYAN}${scoreStr}${RESET}`
    );
  }
  console.log("");
}
