"""Full SFT of Bielik-1.5B-v3.0-Instruct on essay chat examples. Self-contained; one command.

    python train_sft.py --train train.jsonl --val val.jsonl --out bielik-essay-sft [--epochs 2] [--lr 1e-5]

Input: JSONL with {"messages":[{"role":"system"|"user"|"assistant","content":...}]} (chat format).
Output: merged full model in --out (safetensors + tokenizer), ready for llama.cpp convert_hf_to_gguf.py.
Needs: torch, transformers>=4.44, trl>=0.9, datasets. Tested design for one 24-48 GB GPU (bf16, batch 4 x accum 4).
Loss is computed on assistant tokens only (TRL completion-only via assistant_only_loss when the chat template
supports it; falls back to full-sequence loss with a warning)."""
import argparse, json, os, sys, time
from datasets import load_dataset

p = argparse.ArgumentParser()
p.add_argument("--model", default="speakleash/Bielik-1.5B-v3.0-Instruct")
p.add_argument("--train", required=True); p.add_argument("--val", default=None)
p.add_argument("--out", default="bielik-essay-sft")
p.add_argument("--epochs", type=float, default=2.0); p.add_argument("--lr", type=float, default=1e-5)
p.add_argument("--max-len", type=int, default=2048); p.add_argument("--bs", type=int, default=4); p.add_argument("--accum", type=int, default=4)
p.add_argument("--lora", action="store_true", help="LoRA instead of full fine-tune (smaller GPUs); merged on save")
a = p.parse_args()

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig

tok = AutoTokenizer.from_pretrained(a.model)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa")

data = load_dataset("json", data_files={"train": a.train, **({"val": a.val} if a.val else {})})
keep = lambda ex: {"messages": ex["messages"]}
train_ds = data["train"].map(keep, remove_columns=[c for c in data["train"].column_names if c != "messages"])
val_ds = data["val"].map(keep, remove_columns=[c for c in data["val"].column_names if c != "messages"]) if a.val else None
print(f"train examples: {len(train_ds)}" + (f" | val: {len(val_ds)}" if val_ds else ""))

cfg_kwargs = dict(
    output_dir=a.out + "-ckpt", num_train_epochs=a.epochs, learning_rate=a.lr, lr_scheduler_type="cosine", warmup_ratio=0.05,
    per_device_train_batch_size=a.bs, gradient_accumulation_steps=a.accum, bf16=True, gradient_checkpointing=True,
    logging_steps=5, save_strategy="no", eval_strategy="epoch" if val_ds else "no", report_to=[],
    max_length=a.max_len, packing=False,
)
# --- TRL version tolerance: keep only kwargs this SFTConfig knows, map renamed ones ---
import inspect
sig = set(inspect.signature(SFTConfig.__init__).parameters)
renames = {"max_length": "max_seq_length", "eval_strategy": "evaluation_strategy"}
for new, old in renames.items():
    if new not in sig and old in sig and new in cfg_kwargs:
        cfg_kwargs[old] = cfg_kwargs.pop(new)
if "assistant_only_loss" in sig:
    cfg_kwargs["assistant_only_loss"] = True          # loss on assistant turns only
else:
    print("WARNING: this TRL has no assistant_only_loss; training on full sequences", file=sys.stderr)
dropped = [k for k in cfg_kwargs if k not in sig]
for k in dropped:
    print(f"WARNING: SFTConfig has no '{k}', dropping", file=sys.stderr); cfg_kwargs.pop(k)
cfg = SFTConfig(**cfg_kwargs)

peft_config = None
if a.lora:
    from peft import LoraConfig
    peft_config = LoraConfig(r=32, lora_alpha=64, lora_dropout=0.05, target_modules="all-linear", task_type="CAUSAL_LM")

tsig = set(inspect.signature(SFTTrainer.__init__).parameters)
tok_kw = {"processing_class": tok} if "processing_class" in tsig else {"tokenizer": tok}
trainer = SFTTrainer(model=model, args=cfg, train_dataset=train_ds, eval_dataset=val_ds, peft_config=peft_config, **tok_kw)
t = time.time()
trainer.train()
print(f"trained in {(time.time()-t)/60:.1f} min")
if val_ds:
    print("final eval:", trainer.evaluate())

m = trainer.model
if a.lora:
    m = m.merge_and_unload()
m.save_pretrained(a.out, safe_serialization=True)
tok.save_pretrained(a.out)
json.dump(vars(a), open(os.path.join(a.out, "train_args.json"), "w"), indent=1)
print(f"saved merged model to {a.out}/  ->  next: python convert_hf_to_gguf.py {a.out} --outtype q8_0 --outfile bielik-essay-sft.Q8_0.gguf")
