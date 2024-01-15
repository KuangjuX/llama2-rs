use std::fs::File;
use std::io::{Read, Seek};
use std::mem::size_of;
use std::os::fd::AsRawFd;
use std::ptr::{self, NonNull};

use libc::mmap;
use log::{debug, info};

/// Transformer configuration
#[repr(C)]
#[derive(Debug, Default)]
pub struct TransformerConfig {
    /// transformer dimension
    pub dim: u32,
    /// for ffn layers
    pub hidden_dim: u32,
    /// number of layers
    pub num_layers: u32,
    /// number of query heads
    pub num_heads: u32,
    /// number of key/value heads (can be < query heads because of multiquery)
    pub num_kv_heads: u32,
    /// vovabulary size, usually 256
    pub vocab_size: u32,
    /// max sequence length
    pub seq_len: u32,
}

/// Transformer weights
pub struct TransformerWeights {
    /// token embedding table
    pub token_embedding: NonNull<f32>,
    /// weights for rmsnorms
    /// (layer, dim) rmsnorms weights
    pub rms_att_weights: NonNull<f32>,
    /// weights for ffn layers
    pub rms_ffn_weights: NonNull<f32>,
    /// weights for matmuls, note dim == n_heads * head_size
    // (layer, dim, n_heads * head_size)
    pub wq: NonNull<f32>,
    // (layer, dim, n_kv_heads * head_size)
    pub wk: NonNull<f32>,
    // (layer, dim, n_kv_heads * head_size)
    pub wv: NonNull<f32>,
    // (layer, dim, n_heads * head_size)
    pub wo: NonNull<f32>,
    // (layer, hidden_dim, dim)
    pub w1: NonNull<f32>,
    // (layer, hidden_dim, dim)
    pub w2: NonNull<f32>,
    // (layer, hidden_dim, dim)
    pub w3: NonNull<f32>,
    /// final rmsnorm
    pub rms_final_weight: NonNull<f32>,
    /// (optional) classifier weights for the logits, on the last layer
    pub wcls: NonNull<f32>,
}

impl Default for TransformerWeights {
    fn default() -> Self {
        Self {
            token_embedding: NonNull::dangling(),
            rms_att_weights: NonNull::dangling(),
            rms_ffn_weights: NonNull::dangling(),
            wq: NonNull::dangling(),
            wk: NonNull::dangling(),
            wv: NonNull::dangling(),
            wo: NonNull::dangling(),
            w1: NonNull::dangling(),
            w2: NonNull::dangling(),
            w3: NonNull::dangling(),
            rms_final_weight: NonNull::dangling(),
            wcls: NonNull::dangling(),
        }
    }
}

pub struct RunState {
    /// activation at current time stamp.
    pub x: NonNull<f32>,
    /// same, but inside a residual branch (dim,)
    pub xb: NonNull<f32>,
    /// an additional buffer just for convenience (dim,)
    pub xb2: NonNull<f32>,
    /// buffer for hidden dimension in the ffn (hidden_dim,)
    pub hb: NonNull<f32>,
    /// buffer for hidden dimension in the ffn (hidden_dim,)
    pub hb2: NonNull<f32>,
    /// query (dim,)
    pub q: NonNull<f32>,
    /// key (dim,)
    pub k: NonNull<f32>,
    /// value (dim,)
    pub v: NonNull<f32>,
    /// buffer for scores/attention values (n_heads, seq_len)
    pub att: NonNull<f32>,
    /// output logits
    pub logits: NonNull<f32>,
    // (layer, seq_len, dim)
    pub key_cache: NonNull<f32>,
    // (layer, seq_len, dim)
    pub value_cache: NonNull<f32>,
}

impl Default for RunState {
    fn default() -> Self {
        Self {
            x: NonNull::dangling(),
            xb: NonNull::dangling(),
            xb2: NonNull::dangling(),
            hb: NonNull::dangling(),
            hb2: NonNull::dangling(),
            q: NonNull::dangling(),
            k: NonNull::dangling(),
            v: NonNull::dangling(),
            att: NonNull::dangling(),
            logits: NonNull::dangling(),
            key_cache: NonNull::dangling(),
            value_cache: NonNull::dangling(),
        }
    }
}

/// Transformer model
pub struct Transformer {
    /// transformer configuration
    pub config: TransformerConfig,
    /// the weights of the model
    pub weights: TransformerWeights,
    /// buffers for the "wave" of activations in the forward pass
    pub state: RunState,
    // file descriptor for memory mapping.
    pub fd: usize,
    // memory mapped data.
    pub data: NonNull<u8>,
    // size of the checkpoint file in bytes.
    pub file_size: usize,
}

impl Default for Transformer {
    fn default() -> Self {
        Self {
            config: TransformerConfig::default(),
            weights: TransformerWeights::default(),
            state: RunState::default(),
            fd: 0,
            data: NonNull::dangling(),
            file_size: 0,
        }
    }
}

impl Transformer {
    pub fn new(checkpoint_path: String) -> Self {
        let mut transformer = Self::read_checkpoint(checkpoint_path);
        transformer.alloc_run_state();
        transformer
    }

    fn read_checkpoint(checkpoint_path: String) -> Self {
        let mut transformer = Transformer::default();
        let mut file = File::open(checkpoint_path).unwrap();
        // read config header
        file.read_exact(unsafe {
            std::slice::from_raw_parts_mut(
                &mut transformer.config as *mut _ as *mut u8,
                size_of::<TransformerConfig>(),
            )
        })
        .unwrap();
        info!("config: {:?}", transformer.config);

        // negative vocab size is hacky way of signaling unshared weights. bit yikes.
        let shared_weights = transformer.config.vocab_size > 0;
        // transformer.config.vocab_size = transformer.config.vocab_size.abs();
        // figure out the file size
        transformer.file_size = file.seek(std::io::SeekFrom::End(0)).unwrap() as usize;
        debug!("file size: {:#x}", transformer.file_size);

        let fd = file.as_raw_fd();
        debug!("fd: {}", fd);
        let data = unsafe {
            mmap(
                ptr::null_mut(),
                transformer.file_size,
                libc::PROT_READ,
                libc::MAP_PRIVATE,
                fd,
                0,
            )
        };
        debug!("data: {:#x}", data as usize);

        if data.is_null() {
            panic!("failed to mmap file");
        }
        transformer.data = NonNull::new(data as *mut u8).unwrap();
        unsafe {
            transformer.mmap_weights(transformer.data, shared_weights);
        }

        transformer
    }

    unsafe fn mmap_weights(&mut self, ptr: NonNull<u8>, shared_weights: bool) {
        let mut ptr = ptr.as_ptr() as *const f32;
        let config = &self.config;
        let weights = &mut self.weights;
        let head_size = config.dim / config.num_heads;
        let n_layers = config.num_layers;
        let dim = config.dim;
        let hidden_dim = config.hidden_dim;

        weights.token_embedding = NonNull::new(ptr as *mut f32).unwrap();

        ptr = ptr.add((config.vocab_size * dim) as usize);
        weights.rms_att_weights = NonNull::new(ptr as *mut f32).unwrap();

        ptr = ptr.add((config.num_layers * dim) as usize);
        weights.wq = NonNull::new(ptr as *mut f32).unwrap();

        ptr = ptr.add((n_layers * dim * (config.num_heads * head_size)) as usize);
        weights.wk = NonNull::new(ptr as *mut f32).unwrap();

        ptr = ptr.add((n_layers * dim * (config.num_kv_heads * head_size)) as usize);
        weights.wv = NonNull::new(ptr as *mut f32).unwrap();

        ptr = ptr.add((n_layers * dim * (config.num_kv_heads * head_size)) as usize);
        weights.wo = NonNull::new(ptr as *mut f32).unwrap();

        ptr = ptr.add((n_layers * dim * (config.num_heads * head_size)) as usize);
        weights.rms_ffn_weights = NonNull::new(ptr as *mut f32).unwrap();

        ptr = ptr.add((n_layers * dim) as usize);
        weights.w1 = NonNull::new(ptr as *mut f32).unwrap();

        ptr = ptr.add((n_layers * dim * hidden_dim) as usize);
        weights.w2 = NonNull::new(ptr as *mut f32).unwrap();

        ptr = ptr.add((n_layers * hidden_dim * dim) as usize);
        weights.w3 = NonNull::new(ptr as *mut f32).unwrap();

        ptr = ptr.add((n_layers * dim * hidden_dim) as usize);
        weights.rms_final_weight = NonNull::new(ptr as *mut f32).unwrap();

        ptr = ptr.add(dim as usize);
        ptr = ptr.add((config.seq_len * head_size) as usize / 2);
        ptr = ptr.add((config.seq_len * head_size) as usize / 2);

        weights.wcls = if shared_weights {
            weights.token_embedding
        } else {
            NonNull::new(ptr as *mut f32).unwrap()
        };
    }

    fn alloc_run_state(&mut self) {
        let config = &self.config;
        let state = &mut self.state;
        let dim = config.dim as usize;
        let hidden_dim = config.hidden_dim as usize;
        let n_layers = config.num_layers as usize;
        let seq_num = config.seq_len as usize;
        let n_heads = config.num_heads as usize;
        let kv_dim = (dim * config.num_kv_heads as usize) / n_heads;

        state.x = NonNull::new(vec![0.0; dim].as_mut_ptr()).unwrap();
        state.xb = NonNull::new(vec![0.0; dim].as_mut_ptr()).unwrap();
        state.xb2 = NonNull::new(vec![0.0; dim].as_mut_ptr()).unwrap();
        state.hb = NonNull::new(vec![0.0; hidden_dim].as_mut_ptr()).unwrap();
        state.hb2 = NonNull::new(vec![0.0; hidden_dim].as_mut_ptr()).unwrap();
        state.q = NonNull::new(vec![0.0; dim].as_mut_ptr()).unwrap();
        state.key_cache =
            NonNull::new(vec![0.0; n_layers * seq_num * kv_dim].as_mut_ptr()).unwrap();
        state.value_cache =
            NonNull::new(vec![0.0; n_layers * seq_num * kv_dim].as_mut_ptr()).unwrap();
        state.att = NonNull::new(vec![0.0; n_heads * seq_num].as_mut_ptr()).unwrap();
        state.logits = NonNull::new(vec![0.0; config.vocab_size as usize].as_mut_ptr()).unwrap();
    }
}
