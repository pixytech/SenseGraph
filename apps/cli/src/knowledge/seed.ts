import type { SenseGraphEngine } from "../bridge/wasm-loader.js";

export function seedGraph(engine: SenseGraphEngine): void {
  // --- bank: financial vs. river ---
  const bank = engine.addWord("bank");
  const bankFinancial = engine.addSense(bank, "bank.financial");
  engine.addIndicators(bankFinancial, [
    "money", "deposit", "deposited", "account", "loan", "interest",
    "savings", "withdraw", "withdrawal", "atm", "check", "balance",
    "credit", "debit", "finance", "payment", "mortgage",
  ]);
  const bankRiver = engine.addSense(bank, "bank.river");
  engine.addIndicators(bankRiver, [
    "river", "water", "shore", "fish", "stream", "flood", "mud",
    "erosion", "embankment", "overflow", "creek", "lake",
  ]);

  // --- bat: animal vs. sports ---
  const bat = engine.addWord("bat");
  const batAnimal = engine.addSense(bat, "bat.animal");
  engine.addIndicators(batAnimal, [
    "cave", "wing", "wings", "fly", "flew", "flying", "nocturnal",
    "sonar", "echolocation", "vampire", "hanging", "roost",
  ]);
  const batSports = engine.addSense(bat, "bat.sports");
  engine.addIndicators(batSports, [
    "hit", "swing", "ball", "cricket", "baseball", "innings",
    "batter", "batting", "wooden", "pitch", "home", "run",
  ]);

  // --- spring: season vs. water source vs. coil ---
  const spring = engine.addWord("spring");
  const springSeason = engine.addSense(spring, "spring.season");
  engine.addIndicators(springSeason, [
    "flowers", "bloom", "warm", "march", "april", "may", "season",
    "garden", "pollen", "blossom", "thaw", "equinox",
  ]);
  const springWater = engine.addSense(spring, "spring.water_source");
  engine.addIndicators(springWater, [
    "water", "hot", "mineral", "bubbling", "natural", "thermal",
    "geyser", "underground", "fresh", "source",
  ]);
  const springCoil = engine.addSense(spring, "spring.coil");
  engine.addIndicators(springCoil, [
    "metal", "coil", "bounce", "tension", "compress", "mattress",
    "elastic", "mechanism", "loaded", "steel",
  ]);

  // --- crane: bird vs. machine ---
  const crane = engine.addWord("crane");
  const craneBird = engine.addSense(crane, "crane.bird");
  engine.addIndicators(craneBird, [
    "flew", "fly", "bird", "wing", "wings", "nest", "sky",
    "migrate", "feathers", "flock", "wetland", "marsh",
  ]);
  const craneMachine = engine.addSense(crane, "crane.machine");
  engine.addIndicators(craneMachine, [
    "lifted", "lift", "steel", "construction", "heavy", "beam",
    "build", "site", "tower", "operator", "load", "hoist",
  ]);

  // --- bass: fish vs. music ---
  const bass = engine.addWord("bass");
  const bassFish = engine.addSense(bass, "bass.fish");
  engine.addIndicators(bassFish, [
    "fish", "caught", "fishing", "lake", "pond", "bait", "reel",
    "hook", "angler", "lure", "trophy", "striped",
  ]);
  const bassMusic = engine.addSense(bass, "bass.music");
  engine.addIndicators(bassMusic, [
    "guitar", "played", "music", "note", "notes", "concert",
    "band", "amplifier", "strings", "rhythm", "solo", "deep",
  ]);

  // --- cross-concept relationships ---
  engine.addRelated(bankRiver, springWater);
  engine.addRelated(bassFish, bankRiver);
  engine.addRelated(craneBird, batAnimal);
  engine.addRelated(bassMusic, batSports);
}
