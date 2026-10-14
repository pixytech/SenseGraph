use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Token {
    pub text: String,
    pub position: usize,
    pub original: String,
}

static STOP_WORDS: &[&str] = &[
    "a", "an", "the", "is", "was", "were", "are", "am", "been", "be", "at", "in", "on", "to",
    "of", "for", "and", "or", "but", "not", "with", "by", "from", "it", "its", "this", "that",
    "i", "he", "she", "they", "we", "you", "my", "his", "her", "our", "their",
];

pub fn tokenize(sentence: &str) -> Vec<Token> {
    sentence
        .split(|c: char| c.is_whitespace() || c == ',' || c == '.' || c == '!' || c == '?')
        .filter(|s| !s.is_empty())
        .enumerate()
        .map(|(i, word)| Token {
            text: word.to_lowercase(),
            position: i,
            original: word.to_string(),
        })
        .collect()
}

pub fn is_stop_word(word: &str) -> bool {
    STOP_WORDS.contains(&word.to_lowercase().as_str())
}