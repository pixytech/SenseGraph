use std::collections::{HashMap, HashSet};

use super::edge::{Edge, EdgeType};
use super::node::{Node, NodeId, NodeType};

pub struct SemanticGraph {
    nodes: HashMap<NodeId, Node>,
    next_id: NodeId,
    label_index: HashMap<String, Vec<NodeId>>,
    adjacency: HashMap<NodeId, Vec<Edge>>,
    reverse_adj: HashMap<NodeId, Vec<Edge>>,
    word_senses: HashMap<String, Vec<NodeId>>,
    sense_indicators: HashMap<NodeId, HashSet<String>>,
}

impl SemanticGraph {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            next_id: 0,
            label_index: HashMap::new(),
            adjacency: HashMap::new(),
            reverse_adj: HashMap::new(),
            word_senses: HashMap::new(),
            sense_indicators: HashMap::new(),
        }
    }

    pub fn add_word(&mut self, label: &str) -> NodeId {
        let node = Node::word(self.next_id, label);
        self.insert_node(node)
    }

    pub fn add_sense(&mut self, word_id: NodeId, sense_label: &str) -> NodeId {
        let sense_node = Node::sense(self.next_id, sense_label);
        let sense_id = self.insert_node(sense_node);

        self.add_edge(Edge::new(word_id, sense_id, EdgeType::HasSense));

        if let Some(word_node) = self.nodes.get(&word_id) {
            self.word_senses
                .entry(word_node.label.clone())
                .or_default()
                .push(sense_id);
        }

        sense_id
    }

    pub fn add_concept(&mut self, label: &str) -> NodeId {
        let node = Node::concept(self.next_id, label);
        self.insert_node(node)
    }

    pub fn add_context_indicator(&mut self, sense_id: NodeId, indicator_word: &str) {
        let indicator = indicator_word.to_lowercase();
        self.sense_indicators
            .entry(sense_id)
            .or_default()
            .insert(indicator.clone());

        let indicator_node_id = self.find_or_create_word(&indicator);
        self.add_edge(Edge::new(
            indicator_node_id,
            sense_id,
            EdgeType::ContextIndicator,
        ));
    }

    pub fn add_related(&mut self, node_a: NodeId, node_b: NodeId) {
        self.add_edge(Edge::new(node_a, node_b, EdgeType::RelatedTo));
        self.add_edge(Edge::new(node_b, node_a, EdgeType::RelatedTo));
    }

    pub fn get_senses(&self, word: &str) -> Vec<NodeId> {
        let normalized = word.to_lowercase();
        self.word_senses
            .get(&normalized)
            .cloned()
            .unwrap_or_default()
    }

    pub fn get_context_indicators(&self, sense_id: NodeId) -> &HashSet<String> {
        static EMPTY: std::sync::LazyLock<HashSet<String>> =
            std::sync::LazyLock::new(HashSet::new);
        self.sense_indicators.get(&sense_id).unwrap_or(&EMPTY)
    }

    pub fn get_related(&self, node_id: NodeId) -> Vec<NodeId> {
        self.adjacency
            .get(&node_id)
            .map(|edges| {
                edges
                    .iter()
                    .filter(|e| e.edge_type == EdgeType::RelatedTo)
                    .map(|e| e.to)
                    .collect()
            })
            .unwrap_or_default()
    }

    pub fn get_node(&self, id: NodeId) -> Option<&Node> {
        self.nodes.get(&id)
    }

    pub fn find_nodes_by_label(&self, label: &str) -> Vec<&Node> {
        let normalized = label.to_lowercase();
        self.label_index
            .get(&normalized)
            .map(|ids| ids.iter().filter_map(|id| self.nodes.get(id)).collect())
            .unwrap_or_default()
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn edge_count(&self) -> usize {
        self.adjacency.values().map(|edges| edges.len()).sum()
    }

    fn insert_node(&mut self, node: Node) -> NodeId {
        let id = node.id;
        self.label_index
            .entry(node.label.to_lowercase())
            .or_default()
            .push(id);
        self.nodes.insert(id, node);
        self.next_id += 1;
        id
    }

    fn add_edge(&mut self, edge: Edge) {
        self.reverse_adj
            .entry(edge.to)
            .or_default()
            .push(edge.clone());
        self.adjacency.entry(edge.from).or_default().push(edge);
    }

    fn find_or_create_word(&mut self, label: &str) -> NodeId {
        let normalized = label.to_lowercase();
        if let Some(ids) = self.label_index.get(&normalized) {
            for &id in ids {
                if let Some(node) = self.nodes.get(&id) {
                    if node.node_type == NodeType::Word {
                        return id;
                    }
                }
            }
        }
        self.add_word(&normalized)
    }
}
