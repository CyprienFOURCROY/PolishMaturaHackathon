# Bielik essay SFT — how to run (for Bartek)

Everything needed is in this folder + the two data files. ~10 min on a 5090.

```bash
# 0. env: torch, transformers>=4.44, trl>=0.9, datasets (your trening.py env is fine)
cd essay_v2/train

# 1. train (full fine-tune, bf16, ~1000 examples, 2 epochs). If VRAM is tight add --lora.
python train_sft.py --train train.jsonl --val val.jsonl --out bielik-essay-sft

# 2. convert + quantise with your llama.cpp checkout (Bielik is Qwen2.5 architecture, standard path)
python /path/to/llama.cpp/convert_hf_to_gguf.py bielik-essay-sft --outtype f16 --outfile bielik-essay-sft.F16.gguf
/path/to/llama.cpp/build/bin/llama-quantize bielik-essay-sft.F16.gguf bielik-essay-sft.Q8_0.gguf Q8_0

# 3. upload bielik-essay-sft.Q8_0.gguf (~1.7 GB) — HF private repo or Drive — and send the link.
#    Also keep bielik-essay-sft.F16.gguf (3.2 GB) if easy; the Q8 is what we run.
```

What to expect: train loss starts ~1.3–1.6 and ends ~0.6–0.9; eval loss should not rise in epoch 2. If it does,
rerun with `--epochs 1`. Runtime on a 5090: a few minutes.

Data: `train.jsonl` / `val.jsonl` = chat examples (system / user / assistant), three task shapes: full essay,
single body paragraph, intro+conclusion. Targets were written by GPT-6 Sol from `kanon.jsonl` facts and kept only
if a blind CKE-rubric judge scored them ≥ 10/15 with no factual error flagged.
