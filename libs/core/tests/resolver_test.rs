use sensegraph_core::engine::resolver::{resolve, Resolution};
use sensegraph_core::graph::semantic_graph::SemanticGraph;

fn build_test_graph() -> SemanticGraph {
    let mut g = SemanticGraph::new();

    let bank = g.add_word("bank");
    let bank_fin = g.add_sense(bank, "bank.financial");
    for word in &[
        "money", "deposit", "deposited", "account", "loan", "interest", "savings", "withdraw",
    ] {
        g.add_context_indicator(bank_fin, word);
    }
    let bank_river = g.add_sense(bank, "bank.river");
    for word in &["river", "water", "shore", "fish", "stream", "flood", "mud"] {
        g.add_context_indicator(bank_river, word);
    }

    let crane = g.add_word("crane");
    let crane_bird = g.add_sense(crane, "crane.bird");
    for word in &["flew", "fly", "bird", "wing", "nest", "sky", "migrate"] {
        g.add_context_indicator(crane_bird, word);
    }
    let crane_machine = g.add_sense(crane, "crane.machine");
    for word in &[
        "lifted", "lift", "steel", "construction", "heavy", "beam", "build", "site",
    ] {
        g.add_context_indicator(crane_machine, word);
    }

    g
}

#[test]
fn test_resolve_bank_financial() {
    let graph = build_test_graph();
    let result = resolve("I deposited money at the bank", &graph);

    let bank_resolution = result
        .resolutions
        .iter()
        .find(|r| r.token().text == "bank")
        .unwrap();

    match bank_resolution {
        Resolution::Disambiguated { chosen, .. } => {
            assert_eq!(chosen.sense_label, "bank.financial");
            assert!(chosen.score >= 2.0);
        }
        other => panic!("Expected Disambiguated, got {:?}", other),
    }
}

#[test]
fn test_resolve_bank_river() {
    let graph = build_test_graph();
    let result = resolve("The crane flew over the river bank", &graph);

    let bank_resolution = result
        .resolutions
        .iter()
        .find(|r| r.token().text == "bank")
        .unwrap();

    match bank_resolution {
        Resolution::Disambiguated { chosen, .. } => {
            assert_eq!(chosen.sense_label, "bank.river");
        }
        other => panic!("Expected Disambiguated, got {:?}", other),
    }
}

#[test]
fn test_resolve_crane_bird() {
    let graph = build_test_graph();
    let result = resolve("The crane flew over the river bank", &graph);

    let crane_resolution = result
        .resolutions
        .iter()
        .find(|r| r.token().text == "crane")
        .unwrap();

    match crane_resolution {
        Resolution::Disambiguated { chosen, .. } => {
            assert_eq!(chosen.sense_label, "crane.bird");
        }
        other => panic!("Expected Disambiguated, got {:?}", other),
    }
}

#[test]
fn test_resolve_crane_machine() {
    let graph = build_test_graph();
    let result = resolve("The crane lifted the steel beam", &graph);

    let crane_resolution = result
        .resolutions
        .iter()
        .find(|r| r.token().text == "crane")
        .unwrap();

    match crane_resolution {
        Resolution::Disambiguated { chosen, .. } => {
            assert_eq!(chosen.sense_label, "crane.machine");
        }
        other => panic!("Expected Disambiguated, got {:?}", other),
    }
}

#[test]
fn test_ambiguous_no_context() {
    let graph = build_test_graph();
    let result = resolve("bank", &graph);

    let bank_resolution = result
        .resolutions
        .iter()
        .find(|r| r.token().text == "bank")
        .unwrap();

    matches!(bank_resolution, Resolution::Ambiguous { .. });
}

#[test]
fn test_operation_count() {
    let graph = build_test_graph();
    let result = resolve("I deposited money at the bank", &graph);
    assert!(
        result.operation_count < 200,
        "Should use far fewer operations than attention: got {}",
        result.operation_count
    );
}
