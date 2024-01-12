use std::vec::Vec;

/// Transformer configuration
pub struct TransformerConfig {
    /// transformer dimension
    pub dim: i32,
    /// for ffn layers
    pub hidden_dim: i32,
    /// number of layers
    pub num_layers: i32,
    /// number of query heads
    pub num_heads: i32,
    /// number of key/value heads (can be < query heads because of multiquery)
    pub num_kv_heads: i32,
    /// vovabulary size, usually 256
    pub vocab_size: i32,
    /// max sequence length
    pub seq_len: i32,
}

/// Transformer weights
pub struct TransformerWeights {
    /// token embedding table
    pub token_embedding: Vec<f32>,
    /// weights for rmsnorms
    /// (layer, dim) rmsnorms weights
    pub rms_att_weights: Vec<f32>,
    /// weights for ffn layers
    pub rms_ffn_weights: Vec<f32>,
    /// weights for matmuls, note dim == n_heads * head_size
    // (layer, dim, n_heads * head_size)
    pub wq: Vec<f32>,
    // (layer, dim, n_kv_heads * head_size)
    pub wk: Vec<f32>,
    // (layer, dim, n_kv_heads * head_size)
    pub wv: Vec<f32>,
    // (layer, dim, n_heads * head_size)
    pub wo: Vec<f32>,
    // (layer, hidden_dim, dim)
    pub w1: Vec<f32>,
    // (layer, hidden_dim, dim)
    pub w2: Vec<f32>,
    // (layer, hidden_dim, dim)
    pub w3: Vec<f32>,
    /// final rmsnorm
    pub rms_final_weight: Vec<f32>,
    /// (optional) classifier weights for the logits, on the last layer
    pub wcls: Vec<f32>,
}

/// Transformer model
pub struct Transformer {
    /// transformer configuration
    pub config: TransformerConfig,
    pub weights: TransformerWeights,
}

impl Transformer {
    pub fn new(checkpoint_path: String) -> Self {
        todo!()
    }
}
