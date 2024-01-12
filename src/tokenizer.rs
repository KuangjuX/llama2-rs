use std::vec::Vec;

pub struct TokenIndex {
    pub str: Box<u8>,
    pub id: i32,
}

pub struct Tokenizer {
    pub vocab: Vec<String>,
    pub vocab_size: usize,
    pub vocab_scores: Vec<f32>,
    pub sorted_vocab: Vec<TokenIndex>,
    max_token_length: u32,
    byte_pieces: [u8; 512],
}

impl Tokenizer {
    pub fn new(tokenizer_path: String, vocab_size: i32) -> Self {
        todo!()
    }
}
