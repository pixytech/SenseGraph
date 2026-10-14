use sensegraph_core::engine::tokenizer::{is_stop_word, tokenize};

#[test]
fn test_tokenize_simple() {
    let tokens = tokenize("I deposited money at the bank");
    assert_eq!(tokens.len(), 6);
    assert_eq!(tokens[0].text, "i");
    assert_eq!(tokens[5].text, "bank");
}

#[test]
fn test_tokenize_punctuation() {
    let tokens = tokenize("The bank, near the river, was flooded.");
    let texts: Vec<&str> = tokens.iter().map(|t| t.text.as_str()).collect();
    assert!(texts.contains(&"bank"));
    assert!(texts.contains(&"river"));
    assert!(texts.contains(&"flooded"));
}

#[test]
fn test_stop_words() {
    assert!(is_stop_word("the"));
    assert!(is_stop_word("The"));
    assert!(!is_stop_word("bank"));
}
