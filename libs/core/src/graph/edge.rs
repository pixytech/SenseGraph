use serde::{Deserialize, Serialize};

use super::node::NodeId;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EdgeType {
    HasSense,
    ContextIndicator,
    RelatedTo,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Edge {
    pub from: NodeId,
    pub to: NodeId,
    pub edge_type: EdgeType,
    pub weight: f32,
}

impl Edge {
    pub fn new(from: NodeId, to: NodeId, edge_type: EdgeType) -> Self {
        Self {
            from,
            to,
            edge_type,
            weight: 1.0,
        }
    }

    pub fn weighted(from: NodeId, to: NodeId, edge_type: EdgeType, weight: f32) -> Self {
        Self {
            from,
            to,
            edge_type,
            weight,
        }
    }
}
