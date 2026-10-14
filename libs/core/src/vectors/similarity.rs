use super::store::VectorStore;

pub fn cosine_similarity(store: &VectorStore, a: usize, b: usize) -> f32 {
    let vec_a = store.get_vector(a);
    let vec_b = store.get_vector(b);
    let norm_a = store.get_norm(a);
    let norm_b = store.get_norm(b);

    let dot: f32 = vec_a.iter().zip(vec_b.iter()).map(|(x, y)| x * y).sum();
    dot / (norm_a * norm_b)
}

pub fn cosine_similarity_raw(store: &VectorStore, query: &[f32], query_norm: f32, b: usize) -> f32 {
    let vec_b = store.get_vector(b);
    let norm_b = store.get_norm(b);

    let dot: f32 = query.iter().zip(vec_b.iter()).map(|(x, y)| x * y).sum();
    dot / (query_norm * norm_b)
}

pub fn top_k_by_index(store: &VectorStore, idx: usize, k: usize) -> Vec<(usize, f32)> {
    let query = store.get_vector(idx).to_vec();
    let query_norm = store.get_norm(idx);
    top_k_raw(store, &query, query_norm, k, Some(idx))
}

pub fn top_k_multi(store: &VectorStore, indices: &[u32], k: usize) -> Vec<(usize, f32)> {
    if indices.is_empty() {
        return Vec::new();
    }

    let dim = store.embed_dim();
    let mut avg = vec![0.0f32; dim];

    for &idx in indices {
        let vec = store.get_vector(idx as usize);
        for (a, b) in avg.iter_mut().zip(vec.iter()) {
            *a += b;
        }
    }

    let scale = 1.0 / indices.len() as f32;
    for v in avg.iter_mut() {
        *v *= scale;
    }

    let norm: f32 = avg.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm == 0.0 {
        return Vec::new();
    }

    let exclude: std::collections::HashSet<usize> = indices.iter().map(|&i| i as usize).collect();
    top_k_raw_exclude(store, &avg, norm, k, &exclude)
}

pub fn top_k_weighted(store: &VectorStore, indices: &[u32], weights: &[f32], k: usize) -> Vec<(usize, f32)> {
    if indices.is_empty() {
        return Vec::new();
    }

    let dim = store.embed_dim();
    let mut combined = vec![0.0f32; dim];
    let mut total_weight: f32 = 0.0;

    for (i, &idx) in indices.iter().enumerate() {
        let w = weights.get(i).copied().unwrap_or(1.0);
        let vec = store.get_vector(idx as usize);
        for (c, v) in combined.iter_mut().zip(vec.iter()) {
            *c += v * w;
        }
        total_weight += w;
    }

    if total_weight > 0.0 {
        let scale = 1.0 / total_weight;
        for c in combined.iter_mut() {
            *c *= scale;
        }
    }

    let norm: f32 = combined.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm == 0.0 {
        return Vec::new();
    }

    let exclude: std::collections::HashSet<usize> = indices.iter().map(|&i| i as usize).collect();
    top_k_raw_exclude(store, &combined, norm, k, &exclude)
}

fn top_k_raw(store: &VectorStore, query: &[f32], query_norm: f32, k: usize, skip: Option<usize>) -> Vec<(usize, f32)> {
    let mut results: Vec<(usize, f32)> = Vec::with_capacity(store.count());

    for i in 0..store.count() {
        if skip == Some(i) {
            continue;
        }
        let sim = cosine_similarity_raw(store, query, query_norm, i);
        results.push((i, sim));
    }

    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    results.truncate(k);
    results
}

fn top_k_raw_exclude(
    store: &VectorStore,
    query: &[f32],
    query_norm: f32,
    k: usize,
    exclude: &std::collections::HashSet<usize>,
) -> Vec<(usize, f32)> {
    let mut results: Vec<(usize, f32)> = Vec::with_capacity(store.count());

    for i in 0..store.count() {
        if exclude.contains(&i) {
            continue;
        }
        let sim = cosine_similarity_raw(store, query, query_norm, i);
        results.push((i, sim));
    }

    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    results.truncate(k);
    results
}

/// Scores many candidates against a weighted group of query tokens in one pass.
///
/// For candidate c the score is `sum(count_q * cos(c, q)) / ||sum(count_q * q_hat)||`,
/// which equals the cosine between c and the weighted mean of the normalized query
/// vectors. The weighted query vector is built once, so each candidate costs a single
/// dot product regardless of how many query tokens there are.
pub fn score_candidates(store: &VectorStore, candidates: &[u32], query_ids: &[u32], counts: &[f32]) -> Vec<f32> {
    let dim = store.embed_dim();
    let mut combined = vec![0.0f32; dim];
    for (i, &q) in query_ids.iter().enumerate() {
        let weight = counts.get(i).copied().unwrap_or(1.0) / store.get_norm(q as usize);
        for (acc, v) in combined.iter_mut().zip(store.get_vector(q as usize).iter()) {
            *acc += weight * v;
        }
    }
    let combined_norm = combined.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);

    candidates
        .iter()
        .map(|&c| {
            let v = store.get_vector(c as usize);
            let dot: f32 = combined.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
            dot / (combined_norm * store.get_norm(c as usize))
        })
        .collect()
}
