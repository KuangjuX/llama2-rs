// #![deny(warnings)]
#![allow(clippy::iter_nth_zero)]

use std::{env, process::exit};

use log::{info, log_enabled};
use sampler::Sampler;
use tokenizer::Tokenizer;
use transformer::Transformer;

mod sampler;
mod tokenizer;
mod transformer;

extern crate env_logger;
extern crate log;

fn usage_helper() -> ! {
    exit(-1);
}

fn generate(
    transformer: Transformer,
    tokenizer: Tokenizer,
    sampler: Sampler,
    prompt: &[u8],
    steps: i32,
) {
    todo!()
}

fn chat(
    transformer: Transformer,
    tokenizer: Tokenizer,
    sampler: Sampler,
    cli_user_prompt: &[u8],
    cli_system_prompt: &[u8],
    steps: i32,
) {
    todo!()
}

fn main() {
    env_logger::init();
    let argv = env::args().collect::<Vec<String>>();
    let argc = argv.len();

    let mut steps = 0;
    let mut tokenizer_path = String::from("tokenizer.bin");
    let mut temperature = 1.0;
    let mut topp = 0.9;
    let mut steps = 256;
    let mut rng_seed = 0;
    let mut mode = "generate";
    let mut prompt = b"";
    let mut system_prompt = b"";
    let checkpoint_path = if argc >= 2 {
        argv[1].clone()
    } else {
        usage_helper()
    };
    info!("checkpoint_path: {}", checkpoint_path);

    for arg in argv.iter().skip(2) {
        if arg.chars().nth(0).unwrap() != '-' {
            usage_helper()
        }
    }

    let transformer = Transformer::new(checkpoint_path);

    // if steps == 0 || steps > transformer.config.seq_len {
    //     steps = transformer.config.seq_len;
    // }

    // // build the Tokenizer via the model .bin file.
    // let tokenizer = Tokenizer::new(tokenizer_path, transformer.config.vocab_size);

    // let sampler = Sampler::new(transformer.config.vocab_size, temperature, topp, rng_seed);

    // if mode == "generate" {
    //     generate(transformer, tokenizer, sampler, prompt, steps);
    // } else if mode == "chat" {
    //     chat(
    //         transformer,
    //         tokenizer,
    //         sampler,
    //         prompt,
    //         system_prompt,
    //         steps,
    //     );
    // } else {
    //     println!("mode not supported");
    //     usage_helper();
    // }
}
