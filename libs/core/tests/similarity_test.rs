use sensegraph_core::vectors::similarity::{cosine_similarity, score_candidates};
use sensegraph_core::vectors::store::VectorStore;

fn store_from(vectors: &[[f32; 2]]) -> VectorStore {
    let mut raw = Vec::new();
    for v in vectors {
        for x in v {
            raw.extend_from_slice(&x.to_le_bytes());
        }
    }
    let mut store = VectorStore::new(2);
    store.load_f32(&raw);
    store
}

#[test]
fn batched_scores_match_per_candidate_cosines() {
    let store = store_from(&[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.0], [3.0, 1.0]]);
    let query = [0u32, 1u32];
    let counts = [1.0f32, 1.0f32];
    let candidates = [2u32, 3u32, 4u32];

    let batched = score_candidates(&store, &candidates, &query, &counts);

    // Reference: sum(count * cos) divided by the norm of the weighted group.
    let mut group = 0.0f32;
    for (a, ca) in query.iter().zip(counts.iter()) {
        for (b, cb) in query.iter().zip(counts.iter()) {
            group += ca * cb * if a == b { 1.0 } else { cosine_similarity(&store, *a as usize, *b as usize) };
        }
    }
    let norm = group.sqrt();
    for (i, &c) in candidates.iter().enumerate() {
        let mut dot = 0.0f32;
        for (q, n) in query.iter().zip(counts.iter()) {
            dot += n * cosine_similarity(&store, c as usize, *q as usize);
        }
        let expected = dot / norm;
        assert!((batched[i] - expected).abs() < 1e-5, "candidate {c}: {} vs {}", batched[i], expected);
    }
}

#[test]
fn repeated_query_token_counts_double() {
    let store = store_from(&[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]);
    let once = score_candidates(&store, &[2], &[0, 1], &[1.0, 1.0]);
    let weighted = score_candidates(&store, &[2], &[0, 1], &[2.0, 1.0]);
    assert!((once[0] - 1.0).abs() < 1e-5);
    assert!(weighted[0] < once[0], "uneven weights move the query away from the diagonal");
}
