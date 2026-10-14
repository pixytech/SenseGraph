import type { ModelInfo } from "./models";

export interface PromptSet {
  tokenExamples: string[];
  phraseExamples: string[];
  compareExamples: string[];
  // Longer multi token queries used by the complex task suite.
  suite: string[];
  placeholderToken: string;
  placeholderPhrase: string;
}

const CODE: PromptSet = {
  tokenExamples: ["fetch", "async", "iterator", "exception", "regex", "mutex"],
  phraseExamples: [
    "parse json response from http request",
    "sort array of objects by key",
    "read file line by line",
  ],
  compareExamples: ["async await", "array list", "null undefined"],
  suite: [
    "parse json response from http request and handle network errors",
    "sort an array of objects by a nested property in descending order",
    "read a file line by line and close the file handle on error",
    "connect to a database, run a parameterized query and map rows to objects",
    "debounce user input and cancel the previous pending request",
    "recursive depth first traversal of a binary tree",
    "lock a mutex before updating shared state across threads",
    "validate an email address with a regular expression",
    "cache results in a hash map with an expiry time",
    "render a list of components from an array in react",
    "authenticate a user with a token and refresh it when expired",
    "serialize an object to xml and write it to a stream",
  ],
  placeholderToken: "an identifier or keyword, for example fetch",
  placeholderPhrase: "describe what the code should do, for example sort array of objects by key",
};

const GENERAL: PromptSet = {
  tokenExamples: ["river", "science", "market", "forest", "energy", "music"],
  phraseExamples: [
    "stock market falls after interest rate rise",
    "scientists discover a new species in the ocean",
    "ancient castle on a hill above the village",
  ],
  compareExamples: ["river stream", "doctor nurse", "car bicycle"],
  suite: [
    "stock market falls after the central bank raises interest rates",
    "scientists discover a new species of fish in the deep ocean",
    "an ancient castle on a hill above a quiet village",
    "how plants convert sunlight into chemical energy",
    "the football team won the championship after a late goal",
    "heavy rain caused flooding across the river valley",
    "a new vaccine reduces the risk of severe illness",
    "the orchestra performed a symphony in the concert hall",
    "engineers designed a bridge to carry trains over the canyon",
    "the government announced a new policy on renewable energy",
    "students studied mathematics and physics at the university",
    "the chef prepared a meal with fresh vegetables and herbs",
  ],
  placeholderToken: "a single word, for example river",
  placeholderPhrase: "a sentence or phrase, for example stock market falls after interest rate rise",
};

export function promptsFor(model: ModelInfo | undefined): PromptSet {
  return model?.category === "Code" ? CODE : GENERAL;
}
