use wasm_bindgen::prelude::*;

use serde::Serialize;

use crate::engine::resolver;
use crate::graph::semantic_graph::SemanticGraph;
use crate::vectors::similarity;
use crate::vectors::store::VectorStore;

use std::cell::RefCell;
use std::collections::HashMap;

#[derive(Serialize)]
struct GraphStats {
    nodes: usize,
    edges: usize,
}

#[derive(Serialize)]
struct SimilarEntry {
    index: usize,
    score: f32,
}

#[derive(Serialize)]
struct VectorStoreStats {
    count: usize,
    embed_dim: usize,
}

thread_local! {
    static GRAPHS: RefCell<HashMap<u32, SemanticGraph>> = RefCell::new(HashMap::new());
    static VECTOR_STORES: RefCell<HashMap<u32, VectorStore>> = RefCell::new(HashMap::new());
    static NEXT_HANDLE: RefCell<u32> = RefCell::new(0);
}

fn next_handle() -> u32 {
    NEXT_HANDLE.with(|h| {
        let mut h = h.borrow_mut();
        let current = *h;
        *h += 1;
        current
    })
}

fn with_graph<F, R>(handle: u32, f: F) -> Result<R, JsValue>
where
    F: FnOnce(&mut SemanticGraph) -> R,
{
    GRAPHS.with(|graphs| {
        let mut graphs = graphs.borrow_mut();
        match graphs.get_mut(&handle) {
            Some(g) => Ok(f(g)),
            None => Err(JsValue::from_str("Invalid graph handle")),
        }
    })
}

fn with_vectors<F, R>(handle: u32, f: F) -> Result<R, JsValue>
where
    F: FnOnce(&VectorStore) -> R,
{
    VECTOR_STORES.with(|stores| {
        let stores = stores.borrow();
        match stores.get(&handle) {
            Some(s) => Ok(f(s)),
            None => Err(JsValue::from_str("Invalid vector store handle")),
        }
    })
}

fn with_vectors_mut<F, R>(handle: u32, f: F) -> Result<R, JsValue>
where
    F: FnOnce(&mut VectorStore) -> R,
{
    VECTOR_STORES.with(|stores| {
        let mut stores = stores.borrow_mut();
        match stores.get_mut(&handle) {
            Some(s) => Ok(f(s)),
            None => Err(JsValue::from_str("Invalid vector store handle")),
        }
    })
}

// ──────────────────────────────────────────────
// Graph API (unchanged)
// ──────────────────────────────────────────────

#[wasm_bindgen]
pub fn create_graph() -> u32 {
    let handle = next_handle();
    GRAPHS.with(|graphs| {
        graphs.borrow_mut().insert(handle, SemanticGraph::new());
    });
    handle
}

#[wasm_bindgen]
pub fn drop_graph(handle: u32) {
    GRAPHS.with(|graphs| {
        graphs.borrow_mut().remove(&handle);
    });
}

#[wasm_bindgen]
pub fn add_word(handle: u32, label: &str) -> Result<u32, JsValue> {
    with_graph(handle, |g| g.add_word(label))
}

#[wasm_bindgen]
pub fn add_sense(handle: u32, word_id: u32, sense_label: &str) -> Result<u32, JsValue> {
    with_graph(handle, |g| g.add_sense(word_id, sense_label))
}

#[wasm_bindgen]
pub fn add_concept(handle: u32, label: &str) -> Result<u32, JsValue> {
    with_graph(handle, |g| g.add_concept(label))
}

#[wasm_bindgen]
pub fn add_context_indicator(
    handle: u32,
    sense_id: u32,
    indicator: &str,
) -> Result<(), JsValue> {
    with_graph(handle, |g| g.add_context_indicator(sense_id, indicator))
}

#[wasm_bindgen]
pub fn add_related(handle: u32, node_a: u32, node_b: u32) -> Result<(), JsValue> {
    with_graph(handle, |g| g.add_related(node_a, node_b))
}

#[wasm_bindgen]
pub fn resolve(handle: u32, sentence: &str) -> Result<JsValue, JsValue> {
    GRAPHS.with(|graphs| {
        let graphs = graphs.borrow();
        match graphs.get(&handle) {
            Some(g) => {
                let result = resolver::resolve(sentence, g);
                serde_wasm_bindgen::to_value(&result)
                    .map_err(|e| JsValue::from_str(&e.to_string()))
            }
            None => Err(JsValue::from_str("Invalid graph handle")),
        }
    })
}

#[wasm_bindgen]
pub fn graph_stats(handle: u32) -> Result<JsValue, JsValue> {
    GRAPHS.with(|graphs| {
        let graphs = graphs.borrow();
        match graphs.get(&handle) {
            Some(g) => {
                let stats = GraphStats {
                    nodes: g.node_count(),
                    edges: g.edge_count(),
                };
                serde_wasm_bindgen::to_value(&stats)
                    .map_err(|e| JsValue::from_str(&e.to_string()))
            }
            None => Err(JsValue::from_str("Invalid graph handle")),
        }
    })
}

// ──────────────────────────────────────────────
// Vector Store API (new)
// ──────────────────────────────────────────────

#[wasm_bindgen]
pub fn create_vector_store(embed_dim: u32) -> u32 {
    let handle = next_handle();
    VECTOR_STORES.with(|stores| {
        stores
            .borrow_mut()
            .insert(handle, VectorStore::new(embed_dim as usize));
    });
    handle
}

#[wasm_bindgen]
pub fn drop_vector_store(handle: u32) {
    VECTOR_STORES.with(|stores| {
        stores.borrow_mut().remove(&handle);
    });
}

#[wasm_bindgen]
pub fn load_embeddings_f16(handle: u32, data: &[u8]) -> Result<(), JsValue> {
    with_vectors_mut(handle, |s| s.load_f16(data))
}

#[wasm_bindgen]
pub fn load_embeddings_bf16(handle: u32, data: &[u8]) -> Result<(), JsValue> {
    with_vectors_mut(handle, |s| s.load_bf16(data))
}

#[wasm_bindgen]
pub fn load_embeddings_f32(handle: u32, data: &[u8]) -> Result<(), JsValue> {
    with_vectors_mut(handle, |s| s.load_f32(data))
}

#[wasm_bindgen]
pub fn query_similar(handle: u32, vector_index: u32, top_k: u32) -> Result<JsValue, JsValue> {
    with_vectors(handle, |s| {
        let results = similarity::top_k_by_index(s, vector_index as usize, top_k as usize);
        let entries: Vec<SimilarEntry> = results
            .into_iter()
            .map(|(index, score)| SimilarEntry { index, score })
            .collect();
        serde_wasm_bindgen::to_value(&entries).unwrap()
    })
}

#[wasm_bindgen]
pub fn query_similar_multi(handle: u32, indices: &[u32], top_k: u32) -> Result<JsValue, JsValue> {
    with_vectors(handle, |s| {
        let results = similarity::top_k_multi(s, indices, top_k as usize);
        let entries: Vec<SimilarEntry> = results
            .into_iter()
            .map(|(index, score)| SimilarEntry { index, score })
            .collect();
        serde_wasm_bindgen::to_value(&entries).unwrap()
    })
}

#[wasm_bindgen]
pub fn query_similar_weighted(handle: u32, indices: &[u32], weights: &[f32], top_k: u32) -> Result<JsValue, JsValue> {
    with_vectors(handle, |s| {
        let results = similarity::top_k_weighted(s, indices, weights, top_k as usize);
        let entries: Vec<SimilarEntry> = results
            .into_iter()
            .map(|(index, score)| SimilarEntry { index, score })
            .collect();
        serde_wasm_bindgen::to_value(&entries).unwrap()
    })
}

/// Scores a batch of candidates against a weighted group of query tokens in one call.
/// Returns one score per candidate, in the same order as `candidates`.
#[wasm_bindgen]
pub fn score_candidates(handle: u32, candidates: &[u32], query_ids: &[u32], counts: &[f32]) -> Result<Vec<f32>, JsValue> {
    with_vectors(handle, |s| similarity::score_candidates(s, candidates, query_ids, counts))
}

#[wasm_bindgen]
pub fn cosine_sim(handle: u32, a: u32, b: u32) -> Result<f32, JsValue> {
    with_vectors(handle, |s| similarity::cosine_similarity(s, a as usize, b as usize))
}

#[wasm_bindgen]
pub fn vector_store_stats(handle: u32) -> Result<JsValue, JsValue> {
    with_vectors(handle, |s| {
        let stats = VectorStoreStats {
            count: s.count(),
            embed_dim: s.embed_dim(),
        };
        serde_wasm_bindgen::to_value(&stats).unwrap()
    })
}

#[wasm_bindgen]
pub fn init_neighbor_graph(handle: u32, total_count: u32, k: u32) -> Result<(), JsValue> {
    with_vectors_mut(handle, |s| {
        s.init_neighbor_graph(total_count as usize, k as usize);
    })
}

#[wasm_bindgen]
pub fn extend_neighbor_graph(handle: u32, new_target: u32) -> Result<(), JsValue> {
    with_vectors_mut(handle, |s| s.extend_neighbor_graph(new_target as usize))
}

#[wasm_bindgen]
pub fn build_neighbor_batch(handle: u32, batch_start: u32, batch_end: u32) -> Result<u32, JsValue> {
    with_vectors_mut(handle, |s| {
        s.build_neighbor_batch(batch_start as usize, batch_end as usize) as u32
    })
}

#[wasm_bindgen]
pub fn export_neighbor_graph(handle: u32) -> Result<Vec<u8>, JsValue> {
    with_vectors(handle, |s| s.export_neighbor_graph())
}

#[wasm_bindgen]
pub fn import_neighbor_graph(handle: u32, data: &[u8]) -> Result<(), JsValue> {
    with_vectors_mut(handle, |s| s.import_neighbor_graph(data))
}

#[wasm_bindgen]
pub fn merge_neighbor_range(handle: u32, range_start: u32, data: &[u8]) -> Result<(), JsValue> {
    with_vectors_mut(handle, |s| s.merge_neighbor_range(range_start as usize, data))
}

#[wasm_bindgen]
pub fn export_partial_graph(handle: u32, up_to: u32) -> Result<Vec<u8>, JsValue> {
    with_vectors(handle, |s| s.export_partial_graph(up_to as usize))
}

#[wasm_bindgen]
pub fn lookup_neighbors(handle: u32, idx: u32) -> Result<JsValue, JsValue> {
    with_vectors(handle, |s| {
        match s.lookup_neighbors(idx as usize) {
            Some((ids, scores)) => {
                let entries: Vec<SimilarEntry> = ids
                    .iter()
                    .zip(scores.iter())
                    .map(|(&index, &score)| SimilarEntry {
                        index: index as usize,
                        score,
                    })
                    .collect();
                serde_wasm_bindgen::to_value(&entries).unwrap()
            }
            None => serde_wasm_bindgen::to_value::<Vec<SimilarEntry>>(&vec![]).unwrap(),
        }
    })
}

#[wasm_bindgen]
pub fn neighbor_graph_size(handle: u32) -> Result<u32, JsValue> {
    with_vectors(handle, |s| s.neighbor_graph_size() as u32)
}
