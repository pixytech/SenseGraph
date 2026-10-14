import { useState, useCallback, useRef, useEffect, useMemo } from "react";
import { useEngine } from "./hooks/useEngine";
import type { EngineResult, BenchmarkSample } from "./engines/types";
import { MODELS, getSpeedIcons } from "./models";
import { promptsFor } from "./suite";
import { overlapAtK } from "./engines/shared";
import { loadModel as downloadModel } from "./engines/model-loader";
import "./App.css";

type Tab = "benchmark" | "diagrams";

interface SuiteRow {
  query: string;
  tokens: string[];
  trMs: number;
  sgMs: number;
  speedup: number;
  overlap: number;
  top1: boolean;
  trOut: string[];
  sgOut: string[];
  candidates: number;
  error?: string;
}

export default function App() {
  const [tab, setTab] = useState<Tab>("benchmark");
  const [query, setQuery] = useState("");
  const [taskType, setTaskType] = useState<"similar" | "search" | "compare">("search");
  const [benchRunning, setBenchRunning] = useState(false);
  const [benchSpeed, setBenchSpeed] = useState(2);
  const [samples, setSamples] = useState<BenchmarkSample[]>([]);
  const benchIntervalRef = useRef<number | null>(null);

  const createSenseGraphWorker = useMemo(() => () =>
    new Worker(new URL("./engines/sensegraph-worker.ts", import.meta.url), { type: "module" }), []);
  const createTransformerWorker = useMemo(() => () =>
    new Worker(new URL("./engines/transformer-worker.ts", import.meta.url), { type: "module" }), []);

  const sensegraph = useEngine(createSenseGraphWorker);
  const transformer = useEngine(createTransformerWorker);

  const [selectedModel, setSelectedModel] = useState<string>(
    (() => {
      try {
        const saved = localStorage.getItem("sensegraph-last-model");
        if (saved && MODELS.some((m) => m.id === saved)) return saved;
      } catch {}
      return MODELS[0].id;
    })()
  );
  const prompts = useMemo(() => promptsFor(MODELS.find((m) => m.id === selectedModel)), [selectedModel]);
  const waitersRef = useRef<Map<string, (r: EngineResult) => void>>(new Map());
  const [suiteRows, setSuiteRows] = useState<SuiteRow[]>([]);
  const [suiteRunning, setSuiteRunning] = useState(false);
  const [suiteStatus, setSuiteStatus] = useState("");
  const pendingBenchRef = useRef<Map<string, BenchmarkSample>>(new Map());

  const loadModel = useCallback(async () => {
    try { localStorage.setItem("sensegraph-last-model", selectedModel); } catch {}
    const modelInfo = MODELS.find((m) => m.id === selectedModel) ?? MODELS[0];
    const loadInfo = {
      id: modelInfo.id,
      source: modelInfo.source,
      hfRepo: modelInfo.hfRepo,
      hfFile: modelInfo.hfFile,
      maxRows: modelInfo.maxVocab,
    };

    // Download once, then hand the same tokenizer and embeddings to both engines.
    const report = (message: string, percent: number) => {
      transformer.setLoading(message, percent);
      sensegraph.setLoading(message, percent);
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
      sensegraph.start(msg(model.embeddings), [model.embeddings]);
    } catch (err: any) {
      const message = `Model download failed: ${err.message || String(err)}. Check that huggingface.co is reachable, or try a smaller model.`;
      transformer.setError(message);
      sensegraph.setError(message);
    }
  }, [transformer, sensegraph, selectedModel]);

  useEffect(() => {
    sensegraph.onResult((r: EngineResult) => {
      const waiter = waitersRef.current.get(`${r.engine}|${r.type}|${r.query}`);
      if (waiter) {
        waitersRef.current.delete(`${r.engine}|${r.type}|${r.query}`);
        waiter(r);
        return;
      }
      const sample = pendingBenchRef.current.get(r.query);
      if (sample) {
        sample.sensegraph = { latencyMs: r.latencyMs, resultCount: r.results.length };
        if (sample.transformer) {
          setSamples((s) => [...s.slice(-199), { ...sample }]);
          pendingBenchRef.current.delete(r.query);
        }
      }
    });
    transformer.onResult((r: EngineResult) => {
      const waiter = waitersRef.current.get(`${r.engine}|${r.type}|${r.query}`);
      if (waiter) {
        waitersRef.current.delete(`${r.engine}|${r.type}|${r.query}`);
        waiter(r);
        return;
      }
      const sample = pendingBenchRef.current.get(r.query);
      if (sample) {
        sample.transformer = { latencyMs: r.latencyMs, resultCount: r.results.length };
        if (sample.sensegraph) {
          setSamples((s) => [...s.slice(-199), { ...sample }]);
          pendingBenchRef.current.delete(r.query);
        }
      }
    });
  }, [sensegraph, transformer]);

  const bothReady = sensegraph.state.status === "ready" && transformer.state.status === "ready";

  const handleSubmit = useCallback(() => {
    if (!query.trim() || !bothReady) return;
    const text = query.trim();
    const parts = text.split(/\s+/);
    if (taskType === "similar") {
      sensegraph.send({ type: "similar", word: text, topK: 15 });
      transformer.send({ type: "similar", word: text, topK: 15 });
    } else if (taskType === "search") {
      sensegraph.send({ type: "search", query: text, topK: 15 });
      transformer.send({ type: "search", query: text, topK: 15 });
    } else if (taskType === "compare" && parts.length >= 2) {
      sensegraph.send({ type: "compare", word1: parts[0], word2: parts.slice(1).join(" ") });
      transformer.send({ type: "compare", word1: parts[0], word2: parts.slice(1).join(" ") });
    }
  }, [query, taskType, sensegraph, transformer, bothReady]);

  const runOnce = useCallback((engine: "sensegraph" | "transformer", text: string) =>
    new Promise<EngineResult>((resolve, reject) => {
      const key = `${engine}|search|${text}`;
      const timer = window.setTimeout(() => {
        waitersRef.current.delete(key);
        reject(new Error("timed out"));
      }, 60000);
      waitersRef.current.set(key, (r) => { window.clearTimeout(timer); resolve(r); });
      (engine === "sensegraph" ? sensegraph : transformer).send({ type: "search", query: text, topK: 10 });
    }), [sensegraph, transformer]);

  const runSuite = useCallback(async () => {
    if (!bothReady || suiteRunning) return;
    setSuiteRunning(true);
    setSuiteRows([]);
    const median = (v: number[]) => [...v].sort((a, b) => a - b)[Math.floor(v.length / 2)];
    const REPS = 3;
    try {
      for (let qi = 0; qi < prompts.suite.length; qi++) {
        const text = prompts.suite[qi];
        setSuiteStatus(`Query ${qi + 1} of ${prompts.suite.length}`);
        const tLat: number[] = [];
        const gLat: number[] = [];
        let t: EngineResult | null = null;
        let g: EngineResult | null = null;
        for (let rep = 0; rep < REPS; rep++) {
          t = await runOnce("transformer", text);
          g = await runOnce("sensegraph", text);
          tLat.push(t.latencyMs);
          gLat.push(g.latencyMs);
        }
        const trMs = median(tLat);
        const sgMs = median(gLat);
        const row: SuiteRow = {
          query: text,
          tokens: g?.tokens ?? [],
          trMs,
          sgMs,
          speedup: sgMs > 0 ? trMs / sgMs : 0,
          overlap: t && g ? overlapAtK(t.results, g.results, 10) : 0,
          top1: !!(t && g && t.results[0] && g.results[0] && t.results[0].index === g.results[0].index),
          trOut: (t?.results ?? []).slice(0, 5).map((r) => r.word.trim()),
          sgOut: (g?.results ?? []).slice(0, 5).map((r) => r.word.trim()),
          candidates: g?.candidates ?? 0,
          error: t?.error || g?.error,
        };
        setSuiteRows((rows) => [...rows, row]);
      }
      setSuiteStatus("Done");
    } catch (err: any) {
      setSuiteStatus(`Stopped: ${err.message || String(err)}`);
    } finally {
      setSuiteRunning(false);
    }
  }, [bothReady, suiteRunning, prompts, runOnce]);

  const startBenchmark = useCallback(() => {
    setBenchRunning(true);
    setSamples([]);
    const items: { type: "similar" | "search"; text: string }[] = [
      ...prompts.tokenExamples.map((text) => ({ type: "similar" as const, text })),
      ...prompts.phraseExamples.map((text) => ({ type: "search" as const, text })),
      ...prompts.suite.map((text) => ({ type: "search" as const, text })),
    ];
    let idx = 0;
    benchIntervalRef.current = window.setInterval(() => {
      const item = items[idx % items.length];
      pendingBenchRef.current.set(item.text, {
        query: item.text, timestamp: Date.now(), transformer: null, sensegraph: null,
      });
      if (item.type === "similar") {
        sensegraph.send({ type: "similar", word: item.text, topK: 10 });
        transformer.send({ type: "similar", word: item.text, topK: 10 });
      } else {
        sensegraph.send({ type: "search", query: item.text, topK: 10 });
        transformer.send({ type: "search", query: item.text, topK: 10 });
      }
      idx++;
    }, 1000 / benchSpeed);
  }, [benchSpeed, sensegraph, transformer, prompts]);

  const stopBenchmark = useCallback(() => {
    setBenchRunning(false);
    if (benchIntervalRef.current) clearInterval(benchIntervalRef.current);
  }, []);

  const avgLatency = (engine: "sensegraph" | "transformer") => {
    const vals = samples.filter((s) => s[engine]).map((s) => s[engine]!.latencyMs);
    return vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : 0;
  };

  const p95Latency = (engine: "sensegraph" | "transformer") => {
    const vals = samples.filter((s) => s[engine]).map((s) => s[engine]!.latencyMs).sort((a, b) => a - b);
    return vals.length ? vals[Math.floor(vals.length * 0.95)] : 0;
  };

  return (
    <div className="app">
      <header className="header">
        <h1>SenseGraph Benchmark</h1>
        <p className="subtitle">Transformer attention vs graph-based inference, side by side</p>
        <nav className="tabs">
          <button className={tab === "benchmark" ? "active" : ""} onClick={() => setTab("benchmark")}>Benchmark</button>
          <button className={tab === "diagrams" ? "active" : ""} onClick={() => setTab("diagrams")}>How It Works</button>
        </nav>
      </header>

      {tab === "benchmark" && (
        <main className="benchmark">
          <section className="setup-section">
            {(sensegraph.state.status === "idle" && transformer.state.status === "idle") ? (
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
              <EngineCard title="Baseline" subtitle="Attention pass plus full vocabulary scan" state={transformer.state} color="transformer" />
              <EngineCard title="SenseGraph Engine" subtitle="Precomputed neighbor graph, composed per query" state={sensegraph.state} color="sensegraph" />
            </div>
          </section>

          <section className="query-section">
            <div className="query-row">
              <select value={taskType} onChange={(e) => setTaskType(e.target.value as any)}>
                <option value="search">Intent search (phrase)</option>
                <option value="similar">Related tokens</option>
                <option value="compare">Compare two</option>
              </select>
              <input
                type="text" value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleSubmit()}
                placeholder={
                  taskType === "compare" ? "two terms, for example " + prompts.compareExamples[0]
                  : taskType === "similar" ? prompts.placeholderToken
                  : prompts.placeholderPhrase
                }
                disabled={!bothReady}
              />
              <button onClick={handleSubmit} disabled={!bothReady}>Run</button>
            </div>
            <div className="example-row">
              <span className="example-label">Try:</span>
              {(taskType === "compare" ? prompts.compareExamples : taskType === "similar" ? prompts.tokenExamples : prompts.phraseExamples).map((ex) => (
                <button key={ex} className="example-chip" onClick={() => setQuery(ex)}>{ex}</button>
              ))}
            </div>
          </section>

          <section className="results-row">
            <ResultPanel title="Baseline" result={transformer.state.lastResult} color="var(--transformer)" />
            <ResultPanel title="SenseGraph" result={sensegraph.state.lastResult} color="var(--sensegraph)" />
          </section>

          <AgreementBar baseline={transformer.state.lastResult} sensegraph={sensegraph.state.lastResult} />

          <section className="bench-section">
            <h3>Complex task suite</h3>
            <p className="suite-note">
              {prompts.suite.length} longer multi-token queries run through both engines on the loaded model.
              Each query is timed three times (median). The input tokens and the top results of both engines
              are listed so the outputs can be compared directly.
            </p>
            <div className="bench-controls">
              <button onClick={runSuite} disabled={!bothReady || suiteRunning}>
                {suiteRunning ? "Running..." : "Run suite"}
              </button>
              <span className="sample-count">{suiteStatus}</span>
            </div>
            {suiteRows.length > 0 && <SuiteTable rows={suiteRows} />}
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
                  <StatCard label="SenseGraph Avg" value={`${avgLatency("sensegraph").toFixed(1)}ms`} color="var(--sensegraph)" />
                  <StatCard label="Transformer P95" value={`${p95Latency("transformer").toFixed(1)}ms`} color="var(--transformer)" />
                  <StatCard label="SenseGraph P95" value={`${p95Latency("sensegraph").toFixed(1)}ms`} color="var(--sensegraph)" />
                  <StatCard label="Speedup" value={avgLatency("sensegraph") > 0 ? `${(avgLatency("transformer") / avgLatency("sensegraph")).toFixed(0)}x` : "n/a"} color="var(--accent)" />
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

function AgreementBar({ baseline, sensegraph }: { baseline: EngineResult | null; sensegraph: EngineResult | null }) {
  if (!baseline || !sensegraph || baseline.query !== sensegraph.query || baseline.type !== sensegraph.type) return null;
  if (baseline.error || sensegraph.error) return null;
  const speed = sensegraph.latencyMs > 0 ? baseline.latencyMs / sensegraph.latencyMs : 0;
  if (baseline.type === "compare") {
    const a = baseline.results[0]?.score ?? 0;
    const b = sensegraph.results[0]?.score ?? 0;
    return (
      <section className="agreement">
        <span>Score difference: {Math.abs(a - b).toExponential(1)}</span>
        <span>Speedup: {speed.toFixed(1)}x</span>
      </section>
    );
  }
  const overlap = overlapAtK(baseline.results, sensegraph.results, 10);
  const top1 = baseline.results[0] && sensegraph.results[0] && baseline.results[0].index === sensegraph.results[0].index;
  return (
    <section className="agreement">
      <span>Top-10 overlap: {(overlap * 100).toFixed(0)}%</span>
      <span>Top result: {top1 ? "same" : "different"}</span>
      <span>Speedup: {speed.toFixed(1)}x</span>
    </section>
  );
}

function SuiteTable({ rows }: { rows: SuiteRow[] }) {
  const ok = rows.filter((r) => !r.error);
  const totalTr = ok.reduce((a, r) => a + r.trMs, 0);
  const totalSg = ok.reduce((a, r) => a + r.sgMs, 0);
  const meanOverlap = ok.length ? ok.reduce((a, r) => a + r.overlap, 0) / ok.length : 0;
  const top1Rate = ok.length ? ok.filter((r) => r.top1).length / ok.length : 0;
  return (
    <>
      <div className="stats-grid">
        <StatCard label="Baseline total" value={`${totalTr.toFixed(1)}ms`} color="var(--transformer)" />
        <StatCard label="SenseGraph total" value={`${totalSg.toFixed(1)}ms`} color="var(--sensegraph)" />
        <StatCard label="Speedup" value={totalSg > 0 ? `${(totalTr / totalSg).toFixed(1)}x` : "n/a"} color="var(--accent)" />
        <StatCard label="Top-10 overlap" value={`${(meanOverlap * 100).toFixed(0)}%`} color="var(--text-dim)" />
        <StatCard label="Same top result" value={`${(top1Rate * 100).toFixed(0)}%`} color="var(--text-dim)" />
        <StatCard label="Queries" value={`${rows.length}`} color="var(--text-dim)" />
      </div>
      <div className="suite-list">
        {rows.map((r, i) => (
          <details key={i} className="suite-row">
            <summary>
              <span className="suite-query">{i + 1}. {r.query}</span>
              <span className="suite-nums">
                {r.error ? r.error : `${r.trMs.toFixed(2)}ms vs ${r.sgMs.toFixed(2)}ms, ${r.speedup.toFixed(1)}x, overlap ${(r.overlap * 100).toFixed(0)}%`}
              </span>
            </summary>
            <div className="suite-detail">
              <div><strong>Input tokens ({r.tokens.length})</strong></div>
              <div className="token-row">
                {r.tokens.map((t, k) => <span key={k} className="token-chip">{t.trim() || "(space)"}</span>)}
              </div>
              <div className="suite-outputs">
                <div><strong>Baseline top 5</strong><div>{r.trOut.join(", ")}</div></div>
                <div><strong>SenseGraph top 5</strong><div>{r.sgOut.join(", ")}</div></div>
              </div>
              <div className="suite-meta">{r.candidates.toLocaleString()} candidates scored by SenseGraph instead of the full vocabulary</div>
            </div>
          </details>
        ))}
      </div>
    </>
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
          {result.tokens && result.tokens.length > 0 && (
            <div className="token-row">
              {result.tokens.map((t, i) => <span key={i} className="token-chip">{t.trim() || "(space)"}</span>)}
            </div>
          )}
          <div className="meta-row">
            {result.operationCount !== undefined && <span>{result.operationCount.toLocaleString()} multiply-adds</span>}
            {result.candidates !== undefined && result.candidates > 0 && <span>{result.candidates.toLocaleString()} candidates scored</span>}
          </div>
          <div className="result-list">
            {result.results.slice(0, 10).map((r, i) => (
              <div key={i} className="result-item">
                <span className="rank">{i + 1}.</span>
                <span className="word">{r.word.trim() || "(space)"}</span>
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
  const max = Math.max(...recent.map((s) => Math.max(s.transformer?.latencyMs ?? 0, s.sensegraph?.latencyMs ?? 0)), 1);

  return (
    <div className="chart">
      <div className="chart-area">
        {recent.map((s, i) => (
          <div key={i} className="chart-col">
            {s.transformer && <div className="bar transformer" style={{ height: `${(s.transformer.latencyMs / max) * 100}%` }} title={`T: ${s.transformer.latencyMs.toFixed(1)}ms`} />}
            {s.sensegraph && <div className="bar sensegraph" style={{ height: `${(s.sensegraph.latencyMs / max) * 100}%` }} title={`C: ${s.sensegraph.latencyMs.toFixed(1)}ms`} />}
          </div>
        ))}
      </div>
      <div className="chart-legend">
        <span className="legend-item"><span className="dot transformer" /> Transformer</span>
        <span className="legend-item"><span className="dot sensegraph" /> SenseGraph Engine</span>
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
            Each layer computes attention scores between <strong>every pair</strong> of tokens | O(n2 x d) operations.
            With 22 layers, this adds up fast.
          </p>
          <svg viewBox="0 0 800 360" className="arch-diagram">
            <text x="50" y="25" className="label">Input Tokens</text>
            {["parse", "json", "from", "http", "body"].map((t, i) => (
              <g key={i}>
                <rect x={50 + i * 80} y={35} width={70} height={28} rx={4} className="token-box" />
                <text x={85 + i * 80} y={54} className="token-text">{t}</text>
              </g>
            ))}
            <line x1="245" y1="63" x2="245" y2="90" className="arrow" />
            <text x="50" y="88" className="label">Embedding Layer</text>
            <rect x="50" y="95" width="390" height="24" rx={4} className="layer-box embed" />
            <text x="245" y="111" className="layer-text">Lookup table | O(1) per token</text>
            <line x1="245" y1="119" x2="245" y2="150" className="arrow" />
            <text x="50" y="148" className="label">Self-Attention (x22 layers)</text>
            <rect x="50" y="155" width="390" height="28" rx={4} className="layer-box qkv" />
            <text x="245" y="173" className="layer-text">Q x KT to n x n attention matrix</text>
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
            <text x="620" y="282" className="cost-text">~1.1M multiply-adds (cost model)</text>
            <text x="620" y="298" className="cost-sub">For 5 tokens, single forward pass</text>
          </svg>
        </section>

        <section className="diagram-card">
          <h2>How the SenseGraph Engine Works</h2>
          <p style={{ color: "var(--text-dim)", marginBottom: 16, fontSize: 14 }}>
            No attention layers. Each token already has a precomputed list of its nearest neighbors in the
            embedding space. A multi-token query is answered by combining those lists and scoring only the
            candidates, instead of scanning the whole vocabulary.
          </p>
          <svg viewBox="0 0 800 360" className="arch-diagram">
            <text x="50" y="25" className="label">1. Tokenize with the model's own tokenizer</text>
            {["parse", "json", "from", "http", "body"].map((t, i) => (
              <g key={i}>
                <rect x={50 + i * 80} y={35} width={70} height={28} rx={4} className="token-box sensegraph" />
                <text x={85 + i * 80} y={54} className="token-text">{t}</text>
              </g>
            ))}
            <line x1="245" y1="63" x2="245" y2="95" className="arrow sensegraph-arrow" />
            <text x="50" y="93" className="label">2. Fetch the neighbor list of each token</text>
            <rect x="50" y="100" width="390" height="28" rx={4} className="layer-box sensegraph-step" />
            <text x="245" y="118" className="layer-text">K = 20 neighbors per token, built once and cached</text>
            <line x1="245" y1="128" x2="245" y2="160" className="arrow sensegraph-arrow" />
            <text x="50" y="158" className="label">3. Union the lists into a small candidate set</text>
            <rect x="50" y="165" width="390" height="28" rx={4} className="layer-box sensegraph-step" />
            <text x="245" y="183" className="layer-text">about n x K candidates, not the whole vocabulary</text>
            <line x1="245" y1="193" x2="245" y2="225" className="arrow sensegraph-arrow" />
            <text x="50" y="223" className="label">4. Score candidates against the query mean</text>
            <rect x="50" y="230" width="390" height="28" rx={4} className="layer-box sensegraph-step" />
            <text x="245" y="248" className="layer-text">cosine to the mean of the normalized token vectors</text>
            <line x1="245" y1="258" x2="245" y2="290" className="arrow sensegraph-arrow" />
            <text x="50" y="288" className="label">5. Top-K results</text>
            <rect x="50" y="295" width="390" height="24" rx={4} className="result-box" />
            <text x="245" y="311" className="token-text">same ranking target as the baseline</text>
            <rect x="480" y="60" width="270" height="70" rx={8} className="cost-box good big" />
            <text x="615" y="88" className="cost-text good-text">Work per query</text>
            <text x="615" y="108" className="cost-sub">about n x K x n dot products</text>
            <rect x="480" y="150" width="270" height="70" rx={8} className="speedup-box" />
            <text x="615" y="178" className="speedup-num">Measured live</text>
            <text x="615" y="203" className="speedup-sub">see the complex task suite</text>
            <rect x="480" y="240" width="270" height="85" rx={8} className="insight-box" />
            <text x="615" y="265" className="insight-title">Trade-off</text>
            <text x="615" y="288" className="insight-text">Candidates come from neighbor lists,</text>
            <text x="615" y="306" className="insight-text">so overlap with the exact scan is reported.</text>
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
              {selected.name} | {selected.size} {getSpeedIcons(selected.speed)}
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
                      <span>{m.name} | {m.size} {getSpeedIcons(m.speed)}</span>
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
