use sensegraph_core::graph::semantic_graph::SemanticGraph;

#[test]
fn test_add_word_and_senses() {
    let mut g = SemanticGraph::new();
    let bank = g.add_word("bank");
    let fin = g.add_sense(bank, "bank.financial");
    let river = g.add_sense(bank, "bank.river");

    let senses = g.get_senses("bank");
    assert_eq!(senses.len(), 2);
    assert!(senses.contains(&fin));
    assert!(senses.contains(&river));
}

#[test]
fn test_context_indicators() {
    let mut g = SemanticGraph::new();
    let bank = g.add_word("bank");
    let fin = g.add_sense(bank, "bank.financial");

    g.add_context_indicator(fin, "money");
    g.add_context_indicator(fin, "deposit");

    let indicators = g.get_context_indicators(fin);
    assert!(indicators.contains("money"));
    assert!(indicators.contains("deposit"));
    assert!(!indicators.contains("river"));
}

#[test]
fn test_related_concepts() {
    let mut g = SemanticGraph::new();
    let bank = g.add_word("bank");
    let river_sense = g.add_sense(bank, "bank.river");
    let water = g.add_concept("water");

    g.add_related(river_sense, water);

    let related = g.get_related(river_sense);
    assert!(related.contains(&water));
    let reverse_related = g.get_related(water);
    assert!(reverse_related.contains(&river_sense));
}

#[test]
fn test_case_insensitive_lookup() {
    let mut g = SemanticGraph::new();
    let bank = g.add_word("Bank");
    g.add_sense(bank, "bank.financial");

    assert_eq!(g.get_senses("bank").len(), 1);
    assert_eq!(g.get_senses("BANK").len(), 1);
    assert_eq!(g.get_senses("Bank").len(), 1);
}
