import { useState, useCallback, useRef, useEffect, useMemo } from "react";
import { useEngine } from "./hooks/useEngine";
import type { EngineResult, BenchmarkSample } from "./engines/types";
import { MODELS, getSpeedIcons } from "./models";
import { loadModel as downloadModel } from "./engines/model-loader";
import "./App.css";

type Tab = "benchmark" | "diagrams";

export default function App() {
  const [tab, setTab] = useState<Tab>("benchmark");
  const [query, setQuery] = useState("");
  const [taskType, setTaskType] = useState<"similar" | "search" | "compare">("similar");
  const [benchRunning, setBenchRunning] = useState(false);
  const [benchSpeed, setBenchSpeed] = useState(2);
  const [samples, setSamples] = useState<BenchmarkSample[]>([]);
  const benchIntervalRef = useRef<number | null>(null);

  const createCQEWorker = useMemo(() => () =>
    new Worker(new URL("./engines/cqe-worker.ts", import.meta.url), { type: "module" }), []);
  const createTransformerWorker = useMemo(() => () =>
    new Worker(new URL("./engines/transformer-worker.ts", import.meta.url), { type: "module" }), []);

  const cqe = useEngine(createCQEWorker);
  const transformer = useEngine(createTransformerWorker);

  const [selectedModel, setSelectedModel] = useState<string>(
    localStorage.getItem("sensegraph-last-model") || "smollm2-135m"
  );
  const pendingBenchRef = useRef<Map<string, BenchmarkSample>>(new Map());

  const loadModel = useCallback(async () => {
    localStorage.setItem("sensegraph-last-model", selectedModel);
    const modelInfo = MODELS.find((m) => m.id === selectedModel) ?? MODELS[0];
    const loadInfo = {
      id: modelInfo.id,
      source: modelInfo.source,
      hfRepo: modelInfo.hfRepo,
      hfFile: modelInfo.hfFile,
    };

    // Download once, then hand the same tokenizer and embeddings to both engines.
    const report = (message: string, percent: number) => {
      transformer.setLoading(message, percent);
      cqe.setLoading(message, percent);
    };
    try {
      report("Starting download...", 1);
      const model = await downloadModel(loadInfo, report);
      const msg = (copy: ArrayBuffer) => ({
        type: "init-with-model" as const,
        modelId: model.id,
        modelName: model.name,
        tokenizerJson: model.tokenizerJson,
        embeddings: copy,
      });
      const forBaseline = model.embeddings.slice(0);
      transformer.start(msg(forBaseline), [forBaseline]);
      cqe.start(msg(model.embeddings), [model.embeddings]);
    } catch (err: any) {
      const message = `Model download failed: ${err.message || String(err)}. Check that huggingface.co is reachable, or try a smaller model.`;
      transformer.setError(message);
      cqe.setError(message);
    }
  }, [transformer, cqe, selectedModel]);

  useEffect(() => {
    cqe.onResult((r: EngineResult) => {
      const sample = pendingBenchRef.current.get(r.query);
      if (sample) {
        sample.cqe = { latencyMs: r.latencyMs, resultCount: r.results.length };
        if (sample.transformer) {
          setSamples((s) => [...s.slice(-199), { ...sample }]);
          pendingBenchRef.current.delete(r.query);
        }
      }
    });
    transformer.onResult((r: EngineResult) => {
      const sample = pendingBenchRef.current.get(r.query);
      if (sample) {
        sample.transformer = { latencyMs: r.latencyMs, resultCount: r.results.length };
        if (sample.cqe) {
          setSamples((s) => [...s.slice(-199), { ...sample }]);
          pendingBenchRef.current.delete(r.query);
        }
      }
    });
  }, [cqe, transformer]);

  const bothReady = cqe.state.status === "ready" && transformer.state.status === "ready";

  const handleSubmit = useCallback(() => {
    if (!query.trim() || !bothReady) return;
    const parts = query.trim().split(/\s+/);
    if (taskType === "similar") {
      cqe.send({ type: "similar", word: parts[0], topK: 15 });
      transformer.send({ type: "similar", word: parts[0], topK: 15 });
    } else if (taskType === "search") {
      cqe.send({ type: "search", query: query.trim(), topK: 15 });
      transformer.send({ type: "search", query: query.trim(), topK: 15 });
    } else if (taskType === "compare" && parts.length >= 2) {
      cqe.send({ type: "compare", word1: parts[0], word2: parts[1] });
      transformer.send({ type: "compare", word1: parts[0], word2: parts[1] });
    }
  }, [query, taskType, cqe, transformer, bothReady]);

  const BENCH_WORDS = [
    "king", "python", "bank", "spring", "computer", "river", "music", "science",
    "earth", "light", "power", "bridge", "cloud", "metal", "forest", "ocean",
    "energy", "market", "stone", "glass", "iron", "gold", "silver", "star",
  ];

  const startBenchmark = useCallback(() => {
    setBenchRunning(true);
    setSamples([]);
    let idx = 0;
    benchIntervalRef.current = window.setInterval(() => {
      const word = BENCH_WORDS[idx % BENCH_WORDS.length];
      pendingBenchRef.current.set(word, {
        query: word, timestamp: Date.now(), transformer: null, cqe: null,
      });
      cqe.send({ type: "similar", word, topK: 10 });
      transformer.send({ type: "similar", word, topK: 10 });
      idx++;
    }, 1000 / benchSpeed);
  }, [benchSpeed, cqe, transformer]);

  const stopBenchmark = useCallback(() => {
    setBenchRunning(false);
    if (benchIntervalRef.current) clearInterval(benchIntervalRef.current);
  }, []);

  const avgLatency = (engine: "cqe" | "transformer") => {
    const vals = samples.filter((s) => s[engine]).map((s) => s[engine]!.latencyMs);
    return vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : 0;
  };

  const p95Latency = (engine: "cqe" | "transformer") => {
    const vals = samples.filter((s) => s[engine]).map((s) => s[engine]!.latencyMs).sort((a, b) => a - b);
    return vals.length ? vals[Math.floor(vals.length * 0.95)] : 0;
  };

  return (
    <div className="app">
      <header className="header">
        <h1>SenseGraph Benchmark</h1>
        <p className="subtitle">Transformer Attention vs Graph-Based Inference — Side by Side</p>
        <nav className="tabs">
          <button className={tab === "benchmark" ? "active" : ""} onClick={() => setTab("benchmark")}>Benchmark</button>
          <button className={tab === "diagrams" ? "active" : ""} onClick={() => setTab("diagrams")}>How It Works</button>
        </nav>
      </header>

      {tab === "benchmark" && (
        <main className="benchmark">
          <section className="setup-section">
            {(cqe.state.status === "idle" && transformer.state.status === "idle") ? (
              <ModelSelector
                selectedModel={selectedModel}
                onSelectModel={setSelectedModel}
                onLoad={loadModel}
              />
            ) : (
              <div className="load-row">
                <button className="load-alt-btn" onClick={() => window.location.reload()}>
                  Change Model
                </button>
              </div>
            )}
            <div className="setup-row">
              <EngineCard title="Transformer + Attention" subtitle="Full attention matrix computation" state={transformer.state} color="transformer" />
              <EngineCard title="CQE Graph Engine" subtitle="Hash lookup + cached neighbors" state={cqe.state} color="cqe" />
            </div>
          </section>

          <section className="query-section">
            <div className="query-row">
              <select value={taskType} onChange={(e) => setTaskType(e.target.value as any)}>
                <option value="similar">Similar Words</option>
                <option value="search">Semantic Search</option>
                <option value="compare">Compare Words</option>
              </select>
              <input
                type="text" value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleSubmit()}
                placeholder={taskType === "compare" ? "word1 word2" : "Enter a word or phrase..."}
                disabled={!bothReady}
              />
              <button onClick={handleSubmit} disabled={!bothReady}>Run</button>
            </div>
          </section>

          <section className="results-row">
            <ResultPanel title="Transformer" result={transformer.state.lastResult} color="var(--transformer)" />
            <ResultPanel title="CQE (Graph)" result={cqe.state.lastResult} color="var(--cqe)" />
          </section>

          <section className="bench-section">
            <h3>Continuous Benchmark</h3>
            <div className="bench-controls">
              {!benchRunning
                ? <button onClick={startBenchmark} disabled={!bothReady}>Start Benchmark</button>
                : <button onClick={stopBenchmark} className="stop">Stop</button>
              }
              <label>
                Speed: {benchSpeed} q/s
                <input type="range" min={1} max={10} value={benchSpeed}
                  onChange={(e) => setBenchSpeed(Number(e.target.value))} disabled={benchRunning} />
              </label>
              <span className="sample-count">{samples.length} samples</span>
            </div>

            {samples.length > 0 && (
              <>
                <div className="stats-grid">
                  <StatCard label="Transformer Avg" value={`${avgLatency("transformer").toFixed(1)}ms`} color="var(--transformer)" />
                  <StatCard label="CQE Avg" value={`${avgLatency("cqe").toFixed(1)}ms`} color="var(--cqe)" />
                  <StatCard label="Transformer P95" value={`${p95Latency("transformer").toFixed(1)}ms`} color="var(--transformer)" />
                  <StatCard label="CQE P95" value={`${p95Latency("cqe").toFixed(1)}ms`} color="var(--cqe)" />
                  <StatCard label="Speedup" value={avgLatency("cqe") > 0 ? `${(avgLatency("transformer") / avgLatency("cqe")).toFixed(0)}x` : "—"} color="var(--accent)" />
                  <StatCard label="Queries" value={`${samples.length}`} color="var(--text-dim)" />
                </div>
                <div className="latency-chart">
                  <h4>Latency Over Time</h4>
                  <LatencyChart samples={samples} />
                </div>
              </>
            )}
          </section>
        </main>
      )}

      {tab === "diagrams" && <DiagramsTab />}
    </div>
  );
}

function EngineCard({ title, subtitle, state, color }: {
  title: string; subtitle: string; state: any; color: string;
}) {
  const si = state.sizeInfo;
  return (
    <div className="setup-card">
      <h3>{title}</h3>
      <p className="model-name">{state.modelName || subtitle}</p>
      {state.status === "idle" && <span className="badge" style={{ opacity: 0.5 }}>Waiting</span>}
      {state.status === "loading" && (
        <div className="progress">
          <div className={`progress-bar ${color}`} style={{ width: `${state.progressPercent}%` }} />
          <span>{state.progress}</span>
        </div>
      )}
      {state.status === "ready" && (
        <>
          <span className="badge ready">Ready ({state.vocabSize.toLocaleString()} words)</span>
          {si && (
            <div className="size-info">
              <span>Embeddings: {si.embeddingsMB.toFixed(1)} MB</span>
              {si.attentionLayersMB > 0 && <span>Attention layers: {si.attentionLayersMB.toFixed(1)} MB</span>}
              {si.graphMB > 0 && <span>Graph index: {si.graphMB.toFixed(1)} MB</span>}
              <span className="size-total">Total: {si.totalMB.toFixed(1)} MB</span>
            </div>
          )}
        </>
      )}
      {state.status === "error" && <span className="badge error">{state.progress}</span>}
    </div>
  );
}

function ResultPanel({ title, result, color }: { title: string; result: EngineResult | null; color: string }) {
  return (
    <div className="result-panel">
      <h3 style={{ borderBottomColor: color }}>{title}</h3>
      {!result ? (
        <p className="empty">Run a query to see results</p>
      ) : (
        <>
          <div className="latency" style={{ color }}>{result.latencyMs.toFixed(1)}ms</div>
          {result.error && <p className="error">{result.error}</p>}
          <div className="result-list">
            {result.results.slice(0, 10).map((r, i) => (
              <div key={i} className="result-item">
                <span className="rank">{i + 1}.</span>
                <span className="word">{r.word}</span>
                <span className="score">{r.score.toFixed(4)}</span>
                <div className="score-bar" style={{ width: `${Math.max(0, r.score) * 100}%`, backgroundColor: color }} />
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  );
}

function StatCard({ label, value, color }: { label: string; value: string; color: string }) {
  return (
    <div className="stat-card" style={{ borderTopColor: color }}>
      <div className="stat-value">{value}</div>
      <div className="stat-label">{label}</div>
    </div>
  );
}

function LatencyChart({ samples }: { samples: BenchmarkSample[] }) {
  const recent = samples.slice(-50);
  const max = Math.max(...recent.map((s) => Math.max(s.transformer?.latencyMs ?? 0, s.cqe?.latencyMs ?? 0)), 1);

  return (
    <div className="chart">
      <div className="chart-area">
        {recent.map((s, i) => (
          <div key={i} className="chart-col">
            {s.transformer && <div className="bar transformer" style={{ height: `${(s.transformer.latencyMs / max) * 100}%` }} title={`T: ${s.transformer.latencyMs.toFixed(1)}ms`} />}
            {s.cqe && <div className="bar cqe" style={{ height: `${(s.cqe.latencyMs / max) * 100}%` }} title={`C: ${s.cqe.latencyMs.toFixed(1)}ms`} />}
          </div>
        ))}
      </div>
      <div className="chart-legend">
        <span className="legend-item"><span className="dot transformer" /> Transformer</span>
        <span className="legend-item"><span className="dot cqe" /> CQE Engine</span>
      </div>
    </div>
  );
}

function DiagramsTab() {
  return (
    <main className="diagrams">
      <div className="diagram-container">
        <section className="diagram-card">
          <h2>How Transformer Attention Works</h2>
          <p style={{ color: "var(--text-dim)", marginBottom: 16, fontSize: 14 }}>
            Each layer computes attention scores between <strong>every pair</strong> of tokens — O(n2 x d) operations.
            With 22 layers, this adds up fast.
          </p>
          <svg viewBox="0 0 800 360" className="arch-diagram">
            <text x="50" y="25" className="label">Input Tokens</text>
            {["The", "king", "sat", "on", "throne"].map((t, i) => (
              <g key={i}>
                <rect x={50 + i * 80} y={35} width={70} height={28} rx={4} className="token-box" />
                <text x={85 + i * 80} y={54} className="token-text">{t}</text>
              </g>
            ))}
            <line x1="245" y1="63" x2="245" y2="90" className="arrow" />
            <text x="50" y="88" className="label">Embedding Layer</text>
            <rect x="50" y="95" width="390" height="24" rx={4} className="layer-box embed" />
            <text x="245" y="111" className="layer-text">Lookup table — O(1) per token</text>
            <line x1="245" y1="119" x2="245" y2="150" className="arrow" />
            <text x="50" y="148" className="label">Self-Attention (x22 layers)</text>
            <rect x="50" y="155" width="390" height="28" rx={4} className="layer-box qkv" />
            <text x="245" y="173" className="layer-text">Q x KT → n x n attention matrix</text>
            <text x="500" y="148" className="label hot">O(n2 x d) per layer!</text>
            {[0,1,2,3,4].map(r => [0,1,2,3,4].map(c => (
              <rect key={`${r}-${c}`} x={500 + c * 26} y={155 + r * 26} width={23} height={23} rx={2}
                className="attn-cell" style={{ opacity: 0.2 + Math.random() * 0.8 }} />
            )))}
            <text x="570" y="300" className="label small">5 x 5 = 25 dot products</text>
            <text x="570" y="315" className="label small">x 2048 dimensions each</text>
            <text x="570" y="330" className="label small">x 22 layers = 1.1M ops</text>
            <line x1="245" y1="183" x2="245" y2="200" className="arrow" />
            <rect x="50" y="200" width="390" height="24" rx={4} className="layer-box ffn" />
            <text x="245" y="216" className="layer-text">Feed-Forward Network</text>
            <line x1="245" y1="224" x2="245" y2="260" className="arrow" />
            <rect x="50" y="260" width="390" height="28" rx={4} className="layer-box output" />
            <text x="245" y="278" className="layer-text">Contextual token embeddings</text>
            <rect x="490" y="260" width="260" height="50" rx={8} className="cost-box bad" />
            <text x="620" y="282" className="cost-text">~1,126,400 operations</text>
            <text x="620" y="298" className="cost-sub">For 5 tokens, single forward pass</text>
          </svg>
        </section>

        <section className="diagram-card">
          <h2>How CQE Graph Engine Works</h2>
          <p style={{ color: "var(--text-dim)", marginBottom: 16, fontSize: 14 }}>
            Skip all 22 attention layers. Just look up the embedding, scan for nearest neighbors.
            Same knowledge, <strong>75x fewer operations</strong>.
          </p>
          <svg viewBox="0 0 800 360" className="arch-diagram">
            <text x="50" y="25" className="label">Input Word</text>
            <rect x="50" y="35" width="90" height="28" rx={4} className="token-box cqe" />
            <text x="95" y="54" className="token-text">king</text>
            <line x1="95" y1="63" x2="95" y2="95" className="arrow cqe-arrow" />
            <text x="50" y="93" className="label">Hash Table Lookup</text>
            <rect x="50" y="100" width="220" height="28" rx={4} className="layer-box cqe-step" />
            <text x="160" y="118" className="layer-text">vocab["king"] → index 691</text>
            <rect x="290" y="100" width="60" height="28" rx={4} className="cost-box good" />
            <text x="320" y="118" className="cost-small">O(1)</text>
            <line x1="160" y1="128" x2="160" y2="160" className="arrow cqe-arrow" />
            <text x="50" y="158" className="label">Vector Fetch</text>
            <rect x="50" y="165" width="220" height="28" rx={4} className="layer-box cqe-step" />
            <text x="160" y="183" className="layer-text">vectors[691] → 300 floats</text>
            <rect x="290" y="165" width="60" height="28" rx={4} className="cost-box good" />
            <text x="320" y="183" className="cost-small">O(1)</text>
            <line x1="160" y1="193" x2="160" y2="225" className="arrow cqe-arrow" />
            <text x="50" y="223" className="label">Cosine Similarity Scan</text>
            <rect x="50" y="230" width="220" height="28" rx={4} className="layer-box cqe-step" />
            <text x="160" y="248" className="layer-text">dot product with 50K vectors</text>
            <rect x="290" y="230" width="60" height="28" rx={4} className="cost-box ok" />
            <text x="320" y="248" className="cost-small">O(V*d)</text>
            <line x1="160" y1="258" x2="160" y2="290" className="arrow cqe-arrow" />
            <text x="50" y="288" className="label">Top-K Results</text>
            {["queen", "prince", "monarch", "throne"].map((w, i) => (
              <g key={i}>
                <rect x={50 + i * 85} y={295} width={78} height={24} rx={4} className="result-box" />
                <text x={89 + i * 85} y={311} className="token-text">{w}</text>
              </g>
            ))}
            <rect x="460" y="60" width="300" height="60" rx={8} className="cost-box good big" />
            <text x="610" y="85" className="cost-text good-text">~15,000 operations</text>
            <text x="610" y="105" className="cost-sub">No attention layers at all</text>
            <rect x="460" y="140" width="300" height="70" rx={8} className="speedup-box" />
            <text x="610" y="170" className="speedup-num">~75x faster</text>
            <text x="610" y="195" className="speedup-sub">15K vs 1.1M operations</text>
            <rect x="460" y="230" width="300" height="85" rx={8} className="insight-box" />
            <text x="610" y="255" className="insight-title">Key Insight</text>
            <text x="610" y="278" className="insight-text">The embedding layer already contains</text>
            <text x="610" y="296" className="insight-text">the model's learned knowledge.</text>
          </svg>
        </section>
      </div>
    </main>
  );
}

function ModelSelector({ selectedModel, onSelectModel, onLoad }: {
  selectedModel: string;
  onSelectModel: (id: string) => void;
  onLoad: () => void;
}) {
  const [open, setOpen] = useState(false);
  const selected = MODELS.find((m) => m.id === selectedModel) || MODELS[0];

  const categories = [...new Set(MODELS.map((m) => m.category))];

  return (
    <div className="model-selector">
      <div className="model-selector-row">
        <div className="model-dropdown-wrap">
          <label className="model-label">MODEL</label>
          <button className="model-dropdown-btn" onClick={() => setOpen(!open)}>
            <span className="model-dropdown-text">
              {selected.name} — {selected.size} {getSpeedIcons(selected.speed)}
            </span>
            <span className="model-dropdown-arrow">{open ? "▲" : "▼"}</span>
          </button>
          {open && (
            <div className="model-dropdown-menu">
              {categories.map((cat) => (
                <div key={cat}>
                  <div className="model-dropdown-category">{cat}</div>
                  {MODELS.filter((m) => m.category === cat).map((m) => (
                    <button
                      key={m.id}
                      className={`model-dropdown-item ${m.id === selectedModel ? "selected" : ""}`}
                      onClick={() => { onSelectModel(m.id); setOpen(false); }}
                    >
                      <span>{m.name} — {m.size} {getSpeedIcons(m.speed)}</span>
                      {m.source === "huggingface" && <span className="model-tag hf">HF</span>}
                    </button>
                  ))}
                </div>
              ))}
            </div>
          )}
        </div>
        <button className="load-all-btn" onClick={onLoad}>
          Load Model
        </button>
      </div>
    </div>
  );
}
