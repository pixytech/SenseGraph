# SenseGraph command line

A TypeScript REPL on top of the WASM build of the engine.

```bash
npx nx start cli    # builds the Node.js WASM build, then starts the REPL
npx nx test cli
```

Commands: `similar <word>`, `compare <word1> <word2>`, `search <query>`, or type
a sentence to resolve word senses. Embeddings and the tokenizer are read from
`data/` at the repository root. Create them with `npm run extract` from the repository root, which downloads them from Hugging Face.
