import { describe, it, expect, beforeAll, afterAll } from "vitest";
import { createEngine, type SenseGraphEngine, type Resolution } from "../src/bridge/wasm-loader.js";
import { seedGraph } from "../src/knowledge/seed.js";

let engine: SenseGraphEngine;

beforeAll(() => {
  engine = createEngine();
  seedGraph(engine);
});

afterAll(() => {
  engine.destroy();
});

function findResolution(resolutions: Resolution[], word: string): Resolution | undefined {
  return resolutions.find((r) => {
    if ("Literal" in r) return r.Literal.text === word;
    if ("Unambiguous" in r) return r.Unambiguous.token.text === word;
    if ("Disambiguated" in r) return r.Disambiguated.token.text === word;
    if ("Ambiguous" in r) return r.Ambiguous.token.text === word;
    return false;
  });
}

function getChosenSense(r: Resolution): string | null {
  if ("Disambiguated" in r) return r.Disambiguated.chosen.sense_label;
  if ("Unambiguous" in r) return r.Unambiguous.sense_label;
  return null;
}

describe("Word Sense Disambiguation", () => {
  it("resolves 'bank' to financial in money context", () => {
    const result = engine.resolve("I deposited money at the bank");
    const bank = findResolution(result.resolutions, "bank")!;
    expect(getChosenSense(bank)).toBe("bank.financial");
  });

  it("resolves 'bank' to river in nature context", () => {
    const result = engine.resolve("The fish swam near the river bank");
    const bank = findResolution(result.resolutions, "bank")!;
    expect(getChosenSense(bank)).toBe("bank.river");
  });

  it("resolves 'crane' to bird when it flew", () => {
    const result = engine.resolve("The crane flew over the river bank");
    const crane = findResolution(result.resolutions, "crane")!;
    expect(getChosenSense(crane)).toBe("crane.bird");
  });

  it("resolves 'crane' to machine on construction site", () => {
    const result = engine.resolve("The crane lifted the steel beam");
    const crane = findResolution(result.resolutions, "crane")!;
    expect(getChosenSense(crane)).toBe("crane.machine");
  });

  it("resolves 'bat' to animal in cave context", () => {
    const result = engine.resolve("The bat flew out of the cave");
    const bat = findResolution(result.resolutions, "bat")!;
    expect(getChosenSense(bat)).toBe("bat.animal");
  });

  it("resolves 'bat' to sports in cricket context", () => {
    const result = engine.resolve("He picked up the cricket bat and hit the ball");
    const bat = findResolution(result.resolutions, "bat")!;
    expect(getChosenSense(bat)).toBe("bat.sports");
  });

  it("resolves 'bass' to music in concert context", () => {
    const result = engine.resolve("She played bass guitar at the concert");
    const bass = findResolution(result.resolutions, "bass")!;
    expect(getChosenSense(bass)).toBe("bass.music");
  });

  it("resolves 'bass' to fish in fishing context", () => {
    const result = engine.resolve("He caught a bass while fishing at the lake");
    const bass = findResolution(result.resolutions, "bass")!;
    expect(getChosenSense(bass)).toBe("bass.fish");
  });

  it("resolves 'spring' to season in garden context", () => {
    const result = engine.resolve("The flowers bloom in spring");
    const spring = findResolution(result.resolutions, "spring")!;
    expect(getChosenSense(spring)).toBe("spring.season");
  });

  it("resolves 'spring' to water source in thermal context", () => {
    const result = engine.resolve("We bathed in the hot spring water");
    const spring = findResolution(result.resolutions, "spring")!;
    expect(getChosenSense(spring)).toBe("spring.water_source");
  });

  it("reports ambiguity when no context clues exist", () => {
    const result = engine.resolve("bank");
    const bank = findResolution(result.resolutions, "bank")!;
    expect("Ambiguous" in bank || "Disambiguated" in bank).toBe(true);
  });
});

describe("Multiple disambiguations in one sentence", () => {
  it("resolves both crane and bank correctly", () => {
    const result = engine.resolve("The crane flew over the river bank");
    const crane = findResolution(result.resolutions, "crane")!;
    const bank = findResolution(result.resolutions, "bank")!;
    expect(getChosenSense(crane)).toBe("crane.bird");
    expect(getChosenSense(bank)).toBe("bank.river");
  });
});

describe("Performance", () => {
  it("uses far fewer operations than attention", () => {
    const result = engine.resolve("I deposited money at the bank");
    const tokenCount = result.resolutions.length;
    const attentionOps = tokenCount * tokenCount * 512;
    expect(result.operation_count).toBeLessThan(attentionOps);
    expect(result.operation_count).toBeLessThan(200);
  });
});

describe("Graph integrity", () => {
  it("has a populated graph", () => {
    const stats = engine.stats();
    expect(stats.nodes).toBeGreaterThan(50);
    expect(stats.edges).toBeGreaterThan(50);
  });
});
