use serde::{Deserialize, Serialize};

pub type NodeId = u32;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeType {
    Word,
    Sense,
    Concept,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    pub id: NodeId,
    pub node_type: NodeType,
    pub label: String,
}

impl Node {
    pub fn word(id: NodeId, label: &str) -> Self {
        Self {
            id,
            node_type: NodeType::Word,
            label: label.to_lowercase(),
        }
    }

    pub fn sense(id: NodeId, label: &str) -> Self {
        Self {
            id,
            node_type: NodeType::Sense,
            label: label.to_string(),
        }
    }

    pub fn concept(id: NodeId, label: &str) -> Self {
        Self {
            id,
            node_type: NodeType::Concept,
            label: label.to_lowercase(),
        }
    }
}
