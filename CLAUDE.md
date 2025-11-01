# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository implements **In-context Autoencoder (ICAE)**, a context compression technique for Large Language Models published at ICLR 2024. ICAE compresses long contexts into fixed-size memory tokens that can be used by the decoder to reconstruct or continue the context.

### Key Concept
ICAE uses a dual-model architecture:
- **Encoder**: Compresses input text into fixed-size memory tokens
- **Decoder**: Uses compressed memory tokens to reconstruct text or generate continuations

## Repository Structure

The codebase has two major versions:

### Version 2 (icae_v2/) - Current, Recommended
- Based on **Mistral-7B** models
- Supports **multi-span concatenation** (Figure 6 in paper)
- Max input length: **5120 tokens**
- Uses official PEFT package
- Requires `transformers >= 4.36.2`

### Version 1 (icae_v1/) - Legacy
- Based on **Llama-2-7b-chat**
- Uses customized PEFT package (in `code/icae_v1/peft/`)
- Requires `transformers == 4.31.0`
- Install custom PEFT: `pip install -e code/icae_v1/peft/`

## Core Architecture (V2)

### Main Model Class: `ICAE` (modeling_icae_multi_span.py:96)
- Inherits from `torch.nn.Module`
- Contains two model instances during training:
  - `self.icae`: Encoder with LoRA adapters (trainable)
  - `self.decoder`: Frozen decoder for gradient checkpointing
- During inference: Only `self.icae` is used (adapter disabled for decoding)

### Token ID Layout
The model extends the vocabulary with special tokens:
```
[0, vocab_size-1]              : Original vocabulary
vocab_size-1                   : PAD token
[vocab_size, vocab_size+mem_size) : Memory tokens (default: 128 tokens)
vocab_size+mem_size+0          : AE token (autoencoding mode)
vocab_size+mem_size+1          : LM token (language modeling mode)
vocab_size+mem_size+2          : FT token (fine-tuning mode)
```

### Compression Strategy
Input text is segmented and compressed:
1. `compute_num_segments()`: Calculates number of segments based on `mean_compression_rate` (default: 4)
2. Each segment is processed through encoder with memory tokens appended
3. Memory token representations collected to form compressed context
4. Decoder uses compressed memory tokens + prompt to generate output

## Commands

### Running Inference (V2)

**Pretrained model:**
```bash
cd code/icae_v2
bash pretrained_inference_script.sh <path_to_icae_model>
# Download model: wget "https://huggingface.co/sggetao/icae/resolve/main/mistral_7b_pretrained_icae.safetensors"
```

**Instruction fine-tuned model:**
```bash
cd code/icae_v2
bash fine_tuned_inference_script.sh <path_to_icae_model>
# Download model: wget "https://huggingface.co/sggetao/icae/resolve/main/mistral_7b_ft_icae.safetensors"
```

### Training (V2)

**Pretraining:**
```bash
cd code/icae_v2
python pretrain.py \
  --model_name_or_path mistralai/Mistral-7B-Instruct-v0.2 \
  --model_max_length 5120 \
  --fixed_mem_size 128 \
  --lora_r 512 \
  --mean_compression_rate 4 \
  --bf16 \
  --output_dir <output_path> \
  [other training args]
```

**Instruction Fine-tuning:**
```bash
cd code/icae_v2
python instruction_finetune.py \
  --model_name_or_path mistralai/Mistral-7B-Instruct-v0.2 \
  --restore_from <pretrained_icae_checkpoint> \
  --model_max_length 5120 \
  --fixed_mem_size 128 \
  --lora_r 512 \
  --mean_compression_rate 4 \
  --bf16 \
  --output_dir <output_path> \
  [other training args]
```

### Key Training Parameters
- `--fixed_mem_size`: Number of memory tokens (must be power of 2, default: 128)
- `--lora_r`: LoRA rank (default: 512 for V2)
- `--mean_compression_rate`: Target compression ratio (default: 4)
- `--model_max_length`: Max sequence length (default: 5120 for V2)
- `--lm_ratio`: Ratio of language modeling vs autoencoding training (default: 0.0)
- `--min_tokens_for_lm`: Minimum tokens needed for LM objective (default: 64)
- `--leave_tokens_for_lm`: Tokens without loss in LM objective (default: 8)
- `--restore_from`: Path to checkpoint for continued training

## Critical Training Requirements

1. **NEVER use fp16 for training** - Always use `--bf16` (bfloat16). fp16 causes instability.

2. **Batch size = 1** - V2 training code only supports batch_size=1. Increasing batch size does not improve throughput based on experiments.

3. **fixed_mem_size must be power of 2** - Enforced by assertion in pretrain.py:27

4. **Flash Attention 2** - Models use `use_flash_attention_2=True` for efficiency

5. **Gradient Checkpointing** - Enabled for decoder during training to save memory

## Data Format

### Pretraining (pretrain.py)
Dataset should be JSON with "text" field:
```json
{"text": "long document content..."}
```

### Instruction Fine-tuning (instruction_finetune.py)
Dataset should be JSON with three fields:
```json
{
  "input": "context to compress",
  "prompt": "user question",
  "answer": "expected response"
}
```

## Model Resources

- **Models**: https://huggingface.co/sggetao/icae/tree/main
- **Dataset**: https://huggingface.co/datasets/sggetao/PwC

## Important Implementation Details

### Multi-span Concatenation (V2)
V2 models support concatenating compressed representations from multiple text spans (Figure 6 in paper). The `_compress()` method (modeling_icae_multi_span.py:228) handles segmentation and compression.

### Training vs Inference Mode
- **Training** (`train=True`): Uses separate encoder (`self.icae`) and decoder (`self.decoder`)
- **Inference** (`train=False`): Uses single model with `disable_adapter()` context for decoding

### Memory Token Embeddings
Special embedding layer (`self.memory_token_embed`) handles:
- Memory tokens in the vocab extension
- Special tokens (AE, LM, FT)
- Separate from base model embeddings for flexibility

### Loss Computation
- Autoencoding: Full reconstruction loss on target sequence
- Language modeling: Loss starts after `leave_tokens_for_lm` tokens to allow context establishment
- Uses CrossEntropyLoss with ignore_index=-100

## File Organization Patterns

- `modeling_icae_*.py`: Core model architecture
- `*_inference.py`: Inference scripts
- `pretrain.py` / `instruction_finetune.py`: Training entry points
- `training_utils.py`: Tokenization and data collation utilities
- `*_script.sh`: Bash scripts with default hyperparameters
