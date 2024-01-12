pub struct ProbeIndex {
    pub prob: f32,
    index: i32,
}

pub struct Sampler {
    pub vocab_size: i32,
    pub probindex: Vec<ProbeIndex>,
    pub temperature: f32,
    pub topp: f32,
    pub rng_state: u64,
}

impl Sampler {
    pub fn new(vocab_size: i32, temperature: f32, topp: f32, rng_seed: u64) -> Self {
        todo!()
    }
}
