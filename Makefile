LOG ?= DEBUG
DATA ?= stories15M.bin

.PHONY: run
run:
	@RUST_LOG=$(LOG) cargo run $(DATA)