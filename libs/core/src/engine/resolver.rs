use serde::{Deserialize, Serialize};

use crate::graph::node::NodeId;
use crate::graph::semantic_graph::SemanticGraph;

use super::tokenizer::{is_stop_word, tokenize, Token};

const CONTEXT_WINDOW: usize = 5;
const DIRECT_MATCH_SCORE: f32 = 1.0;
const RELATED_MATCH_SCORE: f32 = 0.5;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SenseCandidate {
    pub sense_id: NodeId,
    pub sense_label: String,
    pub score: f32,
    pub matched_indicators: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Resolution {
    Literal(Token),
    Unambiguous {
        token: Token,
        sense_id: NodeId,
        sense_label: String,
    },
    Disambiguated {
        token: Token,
        chosen: SenseCandidate,
        alternatives: Vec<SenseCandidate>,
    },
    Ambiguous {
        token: Token,
        candidates: Vec<SenseCandidate>,
    },
}

impl Resolution {
    pub fn token(&self) -> &Token {
        match self {
            Resolution::Literal(t) => t,
            Resolution::Unambiguous { token, .. } => token,
            Resolution::Disambiguated { token, .. } => token,
            Resolution::Ambiguous { token, .. } => token,
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ResolveResult {
    pub input: String,
    pub resolutions: Vec<Resolution>,
    pub operation_count: u64,
}

pub fn resolve(sentence: &str, graph: &SemanticGraph) -> ResolveResult {
    let tokens = tokenize(sentence);
    let mut resolutions = Vec::with_capacity(tokens.len());
    let mut op_count: u64 = 0;

    for (i, token) in tokens.iter().enumerate() {
        if is_stop_word(&token.text) {
            resolutions.push(Resolution::Literal(token.clone()));
            continue;
        }

        let senses = graph.get_senses(&token.text);
        op_count += 1;

        match senses.len() {
            0 => {
                resolutions.push(Resolution::Literal(token.clone()));
            }
            1 => {
                let sense_id = senses[0];
                let sense_label = graph
                    .get_node(sense_id)
                    .map(|n| n.label.clone())
                    .unwrap_or_default();
                resolutions.push(Resolution::Unambiguous {
                    token: token.clone(),
                    sense_id,
                    sense_label,
                });
            }
            _ => {
                let context_start = i.saturating_sub(CONTEXT_WINDOW);
                let context_end = (i + CONTEXT_WINDOW + 1).min(tokens.len());
                let context_tokens: Vec<&str> = tokens[context_start..context_end]
                    .iter()
                    .filter(|t| t.position != token.position)
                    .map(|t| t.text.as_str())
                    .collect();

                let mut candidates: Vec<SenseCandidate> = senses
                    .iter()
                    .map(|&sense_id| {
                        let mut score: f32 = 0.0;
                        let mut matched = Vec::new();

                        let indicators = graph.get_context_indicators(sense_id);
                        op_count += 1;

                        for ctx_word in &context_tokens {
                            if indicators.contains(*ctx_word) {
                                score += DIRECT_MATCH_SCORE;
                                matched.push(ctx_word.to_string());
                                op_count += 1;
                            }
                        }

                        let related = graph.get_related(sense_id);
                        op_count += 1;

                        for related_id in related {
                            let rel_indicators = graph.get_context_indicators(related_id);
                            op_count += 1;

                            for ctx_word in &context_tokens {
                                if rel_indicators.contains(*ctx_word) {
                                    score += RELATED_MATCH_SCORE;
                                    let rel_label = graph
                                        .get_node(related_id)
                                        .map(|n| format!("{}→{}", n.label, ctx_word))
                                        .unwrap_or_default();
                                    matched.push(rel_label);
                                    op_count += 1;
                                }
                            }
                        }

                        let sense_label = graph
                            .get_node(sense_id)
                            .map(|n| n.label.clone())
                            .unwrap_or_default();

                        SenseCandidate {
                            sense_id,
                            sense_label,
                            score,
                            matched_indicators: matched,
                        }
                    })
                    .collect();

                candidates.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

                if candidates.len() >= 2
                    && candidates[0].score > 0.0
                    && candidates[0].score > candidates[1].score
                {
                    let chosen = candidates.remove(0);
                    resolutions.push(Resolution::Disambiguated {
                        token: token.clone(),
                        chosen,
                        alternatives: candidates,
                    });
                } else {
                    resolutions.push(Resolution::Ambiguous {
                        token: token.clone(),
                        candidates,
                    });
                }
            }
        }
    }

    ResolveResult {
        input: sentence.to_string(),
        resolutions,
        operation_count: op_count,
    }
}