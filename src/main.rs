// #![deny(warnings)]
#![allow(clippy::iter_nth_zero)]

use std::{env, process::exit};

use log::{debug, info};
use sampler::Sampler;
use tokenizer::Tokenizer;
use transformer::Transformer;

mod kernels;
mod sampler;
mod tokenizer;
mod transformer;

extern crate env_logger;
extern crate log;

fn usage_helper() -> ! {
    exit(-1);
}

fn generate(
    transformer: &mut Transformer,
    tokenizer: &Tokenizer,
    sampler: &mut Sampler,
    prompt: &str,
    steps: u32,
) {
    let num_prompt_tokens = 0;
    // +3 for '\0', ?BOS, ?EOS
    let prompt_tokens = vec![0i32; prompt.len() + 3];

    // TODO: encode
    if num_prompt_tokens < 1 {
        panic!("Something is wrong, expected at least 1 prompt token");
    }

    // start the main loop.
    // used to time our code, only initialized after first iteration.
    let mut start = 0;
    // will store the next token in the sequence
    let mut next = 0;
    // kick off with the first token in the prompt.
    let mut roken = prompt_tokens[0];
    // position in the sequence
    let mut pos = 0;

    loop {
        if pos >= steps {
            break;
        }
    }
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

const USAGE_HELP: &str = "\
Usage: cargo run <checkpoint> [OPTIONS]
Options:
     --tokenizer-path <string>
     --temperature <float>
     --top-p <float>
     --steps <int>
     --prompt <string>
     --rng-seed <int>
";

fn main() {
    env_logger::init();
    let mut argv = env::args();
    argv.next().unwrap();

    struct Args {
        checkpoint_path: String,
        tokenizer_path: String,
        temperature: f32,
        topp: f32,
        steps: u32,
        rng_seed: u64,
        prompt: String,
        mode: String,
    }

    let mut args = Args {
        checkpoint_path: argv.next().expect(USAGE_HELP),
        tokenizer_path: String::from("tokenizer.bin"),
        temperature: 1.0,
        topp: 0.9,
        steps: 256,
        rng_seed: 0,
        prompt: String::from(""),
        mode: String::from("generate"),
    };

    loop {
        match argv.next() {
            Some(s) if s == "--tokenizer-path" => {
                args.tokenizer_path = argv.next().map(String::from).expect(USAGE_HELP);
            }
            Some(s) if s == "--temperature" => {
                args.temperature = argv.next().expect(USAGE_HELP).parse().unwrap();
            }
            Some(s) if s == "--top-p" => {
                args.topp = argv.next().expect(USAGE_HELP).parse().unwrap();
            }
            Some(s) if s == "--steps" => {
                args.steps = argv.next().expect(USAGE_HELP).parse().unwrap();
            }
            Some(s) if s == "--prompt" => {
                args.prompt = argv.next().expect(USAGE_HELP);
            }
            Some(s) if s == "--rng-seed" => {
                args.rng_seed = argv.next().expect(USAGE_HELP).parse().unwrap();
            }
            None => break,
            _ => panic!("{USAGE_HELP}"),
        }
    }

    info!("checkpoint_path: {}", args.checkpoint_path);

    let mut transformer = Transformer::new(args.checkpoint_path);

    if args.steps == 0 || args.steps > transformer.config.seq_len {
        args.steps = transformer.config.seq_len;
    }
    debug!("steps: {}", args.steps);

    // build the Tokenizer via the model .bin file.
    let tokenizer = Tokenizer::new(args.tokenizer_path, transformer.config.vocab_size as usize);
    let mut sampler = Sampler::new(
        transformer.config.vocab_size,
        args.temperature,
        args.topp,
        args.rng_seed,
    );

    if args.mode == "generate" {
        generate(
            &mut transformer,
            &tokenizer,
            &mut sampler,
            &args.prompt,
            args.steps,
        );
    } else if args.mode == "chat" {
        todo!()
    } else {
        println!("mode not supported");
        usage_helper();
    }
}
