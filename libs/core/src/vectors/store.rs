use super::quantize::{decode_bf16_buffer, decode_f16_buffer};

#[inline(always)]
unsafe fn fast_dot(data: *const f32, a_off: usize, b_off: usize, dim: usize) -> f32 {
    unsafe {
        let a = data.add(a_off);
        let b = data.add(b_off);
        let mut s0: f32 = 0.0;
        let mut s1: f32 = 0.0;
        let mut s2: f32 = 0.0;
        let mut s3: f32 = 0.0;
        let chunks = dim / 4;
        for c in 0..chunks {
            let base = c * 4;
            s0 += *a.add(base) * *b.add(base);
            s1 += *a.add(base + 1) * *b.add(base + 1);
            s2 += *a.add(base + 2) * *b.add(base + 2);
            s3 += *a.add(base + 3) * *b.add(base + 3);
        }
        for dd in (chunks * 4)..dim {
            s0 += *a.add(dd) * *b.add(dd);
        }
        s0 + s1 + s2 + s3
    }
}

pub struct NeighborGraph {
    neighbors: Vec<Vec<u32>>,
    scores: Vec<Vec<f32>>,
    indexed_count: usize,
    target_count: usize,
    k: usize,
}

impl NeighborGraph {
    pub fn lookup(&self, idx: usize) -> Option<(&[u32], &[f32])> {
        if idx < self.indexed_count && !self.neighbors[idx].is_empty() {
            Some((&self.neighbors[idx], &self.scores[idx]))
        } else {
            None
        }
    }

    pub fn indexed_count(&self) -> usize {
        self.indexed_count
    }
}

pub struct VectorStore {
    data: Vec<f32>,
    normalized: Vec<f32>,
    embed_dim: usize,
    count: usize,
    norms: Vec<f32>,
    neighbor_graph: Option<NeighborGraph>,
}

impl VectorStore {
    pub fn new(embed_dim: usize) -> Self {
        Self {
            data: Vec::new(),
            normalized: Vec::new(),
            embed_dim,
            count: 0,
            norms: Vec::new(),
            neighbor_graph: None,
        }
    }

    pub fn load_f16(&mut self, raw: &[u8]) {
        let num_floats = raw.len() / 2;
        let count = num_floats / self.embed_dim;
        let mut data = vec![0.0f32; num_floats];
        decode_f16_buffer(raw, &mut data);
        self.finish_load(data, count);
    }

    pub fn load_bf16(&mut self, raw: &[u8]) {
        let num_floats = raw.len() / 2;
        let count = num_floats / self.embed_dim;
        let mut data = vec![0.0f32; num_floats];
        decode_bf16_buffer(raw, &mut data);
        self.finish_load(data, count);
    }

    pub fn load_f32(&mut self, raw: &[u8]) {
        let num_floats = raw.len() / 4;
        let count = num_floats / self.embed_dim;
        let mut data = vec![0.0f32; num_floats];
        for (i, chunk) in raw.chunks_exact(4).enumerate().take(num_floats) {
            data[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }
        self.finish_load(data, count);
    }

    fn finish_load(&mut self, data: Vec<f32>, count: usize) {
        let d = self.embed_dim;
        let mut norms = Vec::with_capacity(count);
        let mut normalized = vec![0.0f32; data.len()];

        for i in 0..count {
            let start = i * d;
            let end = start + d;
            let norm: f32 = data[start..end].iter().map(|x| x * x).sum::<f32>().sqrt();
            let norm = if norm > 0.0 { norm } else { 1.0 };
            norms.push(norm);
            let inv = 1.0 / norm;
            for dd in start..end {
                normalized[dd] = data[dd] * inv;
            }
        }

        self.data = data;
        self.normalized = normalized;
        self.count = count;
        self.norms = norms;
    }

    pub fn get_vector(&self, index: usize) -> &[f32] {
        let start = index * self.embed_dim;
        &self.data[start..start + self.embed_dim]
    }

    pub fn get_norm(&self, index: usize) -> f32 {
        self.norms[index]
    }

    pub fn count(&self) -> usize {
        self.count
    }

    pub fn embed_dim(&self) -> usize {
        self.embed_dim
    }

    pub fn init_neighbor_graph(&mut self, total_count: usize, k: usize) {
        let n = total_count.min(self.count);
        self.neighbor_graph = Some(NeighborGraph {
            neighbors: vec![Vec::new(); n],
            scores: vec![Vec::new(); n],
            indexed_count: 0,
            target_count: n,
            k,
        });
    }

    pub fn extend_neighbor_graph(&mut self, new_target: usize) {
        if let Some(ref mut graph) = self.neighbor_graph {
            let n = new_target.min(self.count);
            if n > graph.neighbors.len() {
                graph.neighbors.resize(n, Vec::new());
                graph.scores.resize(n, Vec::new());
            }
            graph.target_count = n;
        }
    }

    pub fn build_neighbor_batch(&mut self, batch_start: usize, batch_end: usize) -> usize {
        let d = self.embed_dim;
        let total = self.count;
        let graph = self.neighbor_graph.as_mut().unwrap();
        let k = graph.k;
        let end = batch_end.min(graph.target_count);
        let mut ops: usize = 0;
        let norm_ptr = self.normalized.as_ptr();

        for i in batch_start..end {
            let i_off = i * d;
            let mut min_score: f32 = f32::NEG_INFINITY;
            let mut top_ids = vec![0u32; k];
            let mut top_scores = vec![f32::NEG_INFINITY; k];
            let mut filled = 0usize;
            let mut min_idx = 0usize;

            for j in 0..total {
                if i == j { continue; }
                let j_off = j * d;

                let dot = unsafe { fast_dot(norm_ptr, i_off, j_off, d) };
                ops += d;

                if filled < k {
                    top_ids[filled] = j as u32;
                    top_scores[filled] = dot;
                    filled += 1;
                    if filled == k {
                        min_idx = 0;
                        for m in 1..k {
                            if top_scores[m] < top_scores[min_idx] { min_idx = m; }
                        }
                        min_score = top_scores[min_idx];
                    }
                } else if dot > min_score {
                    top_ids[min_idx] = j as u32;
                    top_scores[min_idx] = dot;
                    min_idx = 0;
                    for m in 1..k {
                        if top_scores[m] < top_scores[min_idx] { min_idx = m; }
                    }
                    min_score = top_scores[min_idx];
                }
            }

            let mut indices: Vec<usize> = (0..filled).collect();
            indices.sort_by(|&a, &b| top_scores[b].partial_cmp(&top_scores[a]).unwrap());

            graph.neighbors[i] = indices.iter().map(|&idx| top_ids[idx]).collect();
            graph.scores[i] = indices.iter().map(|&idx| top_scores[idx]).collect();
        }

        graph.indexed_count = end;
        ops
    }

    pub fn lookup_neighbors(&self, idx: usize) -> Option<(&[u32], &[f32])> {
        self.neighbor_graph.as_ref()?.lookup(idx)
    }

    pub fn neighbor_graph_size(&self) -> usize {
        self.neighbor_graph.as_ref().map_or(0, |g| g.indexed_count())
    }

    pub fn export_neighbor_graph(&self) -> Vec<u8> {
        let graph = match &self.neighbor_graph {
            Some(g) => g,
            None => return Vec::new(),
        };
        let n = graph.indexed_count as u32;
        let k = graph.k;
        let mut out = Vec::with_capacity(8 + (n as usize) * k * 8);
        out.extend_from_slice(&n.to_le_bytes());
        out.extend_from_slice(&(k as u32).to_le_bytes());
        for i in 0..graph.indexed_count {
            for j in 0..k {
                let idx = graph.neighbors[i].get(j).copied().unwrap_or(0);
                out.extend_from_slice(&idx.to_le_bytes());
            }
            for j in 0..k {
                let score = graph.scores[i].get(j).copied().unwrap_or(0.0);
                out.extend_from_slice(&score.to_le_bytes());
            }
        }
        out
    }

    pub fn import_neighbor_graph(&mut self, data: &[u8]) {
        if data.len() < 8 { return; }
        let n = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
        let k = u32::from_le_bytes([data[4], data[5], data[6], data[7]]) as usize;
        let mut offset = 8;
        let mut neighbors = Vec::with_capacity(n);
        let mut scores = Vec::with_capacity(n);
        for _ in 0..n {
            let mut ids = Vec::with_capacity(k);
            for _ in 0..k {
                if offset + 4 > data.len() { break; }
                let idx = u32::from_le_bytes([data[offset], data[offset+1], data[offset+2], data[offset+3]]);
                ids.push(idx);
                offset += 4;
            }
            let mut sc = Vec::with_capacity(k);
            for _ in 0..k {
                if offset + 4 > data.len() { break; }
                let s = f32::from_le_bytes([data[offset], data[offset+1], data[offset+2], data[offset+3]]);
                sc.push(s);
                offset += 4;
            }
            neighbors.push(ids);
            scores.push(sc);
        }
        self.neighbor_graph = Some(NeighborGraph {
            neighbors,
            scores,
            indexed_count: n,
            target_count: n,
            k,
        });
    }

    pub fn merge_neighbor_range(&mut self, range_start: usize, data: &[u8]) {
        if data.len() < 8 { return; }
        let n = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
        let k = u32::from_le_bytes([data[4], data[5], data[6], data[7]]) as usize;
        let mut offset = 8;

        let graph = match &mut self.neighbor_graph {
            Some(g) => g,
            None => return,
        };

        for rel_i in 0..n {
            let abs_i = rel_i;
            if abs_i >= graph.neighbors.len() { break; }
            if abs_i < range_start { offset += k * 8; continue; }

            let mut ids = Vec::with_capacity(k);
            for _ in 0..k {
                if offset + 4 > data.len() { break; }
                ids.push(u32::from_le_bytes([data[offset], data[offset+1], data[offset+2], data[offset+3]]));
                offset += 4;
            }
            let mut scores = Vec::with_capacity(k);
            for _ in 0..k {
                if offset + 4 > data.len() { break; }
                scores.push(f32::from_le_bytes([data[offset], data[offset+1], data[offset+2], data[offset+3]]));
                offset += 4;
            }

            if !ids.is_empty() {
                graph.neighbors[abs_i] = ids;
                graph.scores[abs_i] = scores;
                if abs_i >= graph.indexed_count {
                    graph.indexed_count = abs_i + 1;
                }
            }
        }
    }

    pub fn export_partial_graph(&self, up_to: usize) -> Vec<u8> {
        let graph = match &self.neighbor_graph {
            Some(g) => g,
            None => return Vec::new(),
        };
        let n = up_to.min(graph.indexed_count) as u32;
        let k = graph.k as u32;
        let mut out = Vec::with_capacity(8 + (n as usize) * (k as usize) * 8);
        out.extend_from_slice(&n.to_le_bytes());
        out.extend_from_slice(&k.to_le_bytes());
        for i in 0..n as usize {
            for &idx in &graph.neighbors[i] {
                out.extend_from_slice(&idx.to_le_bytes());
            }
            for &score in &graph.scores[i] {
                out.extend_from_slice(&score.to_le_bytes());
            }
        }
        out
    }
}
