use std::vec::Vec;

/// Transformer configuration
pub struct TransformerConfig {
    /// transformer dimension
    dim: i32,
    /// for ffn layers
    hidden_dim: i32,
    /// number of layers
    num_layers: i32,
    /// number of query heads
    num_heads: i32,
    /// number of key/value heads (can be < query heads because of multiquery)
    num_kv_heads: i32,
    /// vovabulary size, usually 256
    vocab_size: i32,
    /// max sequence length
    seq_len: i32,
}

/// Transformer weights
pub struct TransformerWeights {
    /// token embedding table
    token_embedding: Vec<f32>,
    /// weights for rmsnorms
    /// (layer, dim) rmsnorms weights
    rms_att_weights: Vec<f32>,
    /// weights for ffn layers
    rms_ffn_weights: Vec<f32>,
    /// weights for matmuls, note dim == n_heads * head_size
    // (layer, dim, n_heads * head_size)
    wq: Vec<f32>,
    // (layer, dim, n_kv_heads * head_size)
    wk: Vec<f32>,
    // (layer, dim, n_kv_heads * head_size)
    wv: Vec<f32>,
    // (layer, dim, n_heads * head_size)
    wo: Vec<f32>,
    // (layer, hidden_dim, dim)
    w1: Vec<f32>,
    // (layer, hidden_dim, dim)
    w2: Vec<f32>,
    // (layer, hidden_dim, dim)
    w3: Vec<f32>,
    /// final rmsnorm
    rms_final_weight: Vec<f32>,
    /// (optional) classifier weights for the logits, on the last layer
    wcls: Vec<f32>,
}
