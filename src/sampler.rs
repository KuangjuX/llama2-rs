pub struct ProbeIndex {
    pub prob: f32,
    pub index: u32,
}

pub struct Sampler {
    pub vocab_size: u32,
    pub prob_index: Vec<ProbeIndex>,
    pub temperature: f32,
    pub topp: f32,
    pub rng_state: u64,
}

impl Sampler {
    pub fn new(vocab_size: u32, temperature: f32, topp: f32, rng_seed: u64) -> Self {
        Self {
            vocab_size,
            prob_index: Vec::with_capacity(vocab_size as usize),
            temperature,
            topp,
            rng_state: rng_seed,
        }
    }

    pub fn random_u32(mut state: u64) -> u32 {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        (state.wrapping_mul(0x2545F4914F6CDD1D) >> 32) as u32
    }

    pub fn random_f32(mut state: u64) -> f32 {
        (Self::random_u32(state) >> 8) as f32 / 16777216.0
    }
}
