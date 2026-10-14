import { useState, useRef, useCallback, useEffect } from "react";
import type { WorkerMessage, WorkerResponse, EngineResult, ModelSizeInfo } from "../engines/types";

interface EngineState {
  status: "idle" | "loading" | "ready" | "error";
  progress: string;
  progressPercent: number;
  modelName: string;
  vocabSize: number;
  sizeInfo: ModelSizeInfo | null;
  lastResult: EngineResult | null;
  latencyHistory: number[];
}

export function useEngine(createWorker: () => Worker) {
  const workerRef = useRef<Worker | null>(null);
  const [state, setState] = useState<EngineState>({
    status: "idle",
    progress: "",
    progressPercent: 0,
    modelName: "",
    vocabSize: 0,
    sizeInfo: null,
    lastResult: null,
    latencyHistory: [],
  });

  const resultCallbackRef = useRef<((result: EngineResult) => void) | null>(null);

  useEffect(() => {
    return () => {
      workerRef.current?.terminate();
    };
  }, []);

  const start = useCallback((msg: WorkerMessage, transfer: Transferable[] = []) => {
    if (workerRef.current) workerRef.current.terminate();

    let worker: Worker;
    try {
      worker = createWorker();
    } catch (err: any) {
      setState((s) => ({ ...s, status: "error", progress: `Worker create failed: ${err.message}` }));
      return;
    }
    workerRef.current = worker;

    setState((s) => ({ ...s, status: "loading", progress: "Starting worker..." }));

    worker.onerror = (e) => {
      setState((s) => ({ ...s, status: "error", progress: `Worker error: ${e.message || "failed to load"}` }));
    };

    worker.onmessage = (e: MessageEvent<WorkerResponse>) => {
      const msg = e.data;
      switch (msg.type) {
        case "progress":
          setState((s) => ({
            ...s,
            progress: msg.message,
            progressPercent: msg.percent ?? s.progressPercent,
          }));
          break;
        case "ready":
          setState((s) => ({
            ...s,
            status: "ready",
            modelName: msg.modelName,
            vocabSize: msg.vocabSize,
            sizeInfo: msg.sizeInfo ?? null,
            progress: "Ready",
            progressPercent: 100,
          }));
          break;
        case "result":
          setState((s) => ({
            ...s,
            lastResult: msg.data,
            latencyHistory: [...s.latencyHistory.slice(-99), msg.data.latencyMs],
          }));
          resultCallbackRef.current?.(msg.data);
          break;
        case "error":
          setState((s) => ({
            ...s,
            status: "error",
            progress: msg.message,
          }));
          break;
      }
    };

    worker.postMessage(msg, transfer);
  }, [createWorker]);

  const setLoading = useCallback((message: string, percent: number) => {
    setState((s) => ({ ...s, status: "loading", progress: message, progressPercent: percent }));
  }, []);

  const setError = useCallback((message: string) => {
    setState((s) => ({ ...s, status: "error", progress: message }));
  }, []);

  const send = useCallback((msg: WorkerMessage) => {
    workerRef.current?.postMessage(msg);
  }, []);

  const onResult = useCallback((cb: (result: EngineResult) => void) => {
    resultCallbackRef.current = cb;
  }, []);

  return { state, start, setLoading, setError, send, onResult };
}
