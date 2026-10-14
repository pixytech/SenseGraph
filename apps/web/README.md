# SenseGraph web app

React and Vite app that runs two engines side by side on the same embeddings:
a baseline that simulates attention and scans the vocabulary, and SenseGraph, which
looks neighbors up in a precomputed graph. See the repository README for the
architecture.

```bash
npx nx dev web      # builds the WASM engine, then starts Vite
npx nx build web    # production build in apps/web/dist
```

No data files are needed. Choose a model in the dropdown and the app downloads
its tokenizer and embedding tensor from Hugging Face once, then runs both
engines on that same data.
Set `BASE_PATH` (for example `/SenseGraph/`) when building for a sub-path.
