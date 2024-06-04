# Changelog

All changes we make to the assignment code or PDF will be documented in this file.

## [unreleased] - yyyy-mm-dd

### Added

### Changed

- code: fix AlpacaEval auto-evaluator to use local Llama 3 70B instruct.
- code: add missing `evaluate_safety.py` script to `scripts`
- code: add Llama 3 tokenizer as a fixture.
- code: fix DPO loss test
- handout: fix typo in code example for writing AlpacaEval outputs.
- handout: provide more instructions on interpreting AlpacaEval annotations file.
- handout: give better default DPO hyperparameters
- handout: clarify prompt to use for the DPO loss (AlpacaEval prompt) and mention EOS token

### Fixed

## 0.0.2 - 2024-05-30

### Added

- code: add MMLU, GSM8K, AlpacaEval, and SimpleSafetyTests data to `./data`.

### Changed

### Fixed


## 0.0.1 - 2024-05-30

### Added

- handout: explicitly set CUDA_HOME in FlashAttention-2 installation instructions. 
- code: explicitly set CUDA_HOME in FlashAttention-2 installation instructions.

### Changed

### Fixed


## 0.0.0 - 2024-05-30

Initial release
