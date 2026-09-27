#!/usr/bin/env bash
# Co to jest: trening obu uczniów eseju na esejach nauczyciela (eksperyment 27.09).
# Po co: jedno polecenie od danych do plików gotowych do oceny: S1 Bielik-1.5B-v3 pełny SFT → Q8_0,
#        S2 Qwen3.5-4B LoRA → adapter GGUF do bazy UD-IQ3_XXS (adapter nie liczy się do rozmiaru, D7).
# Co zrobić: bash scripts/trenuj_uczniow_eseju.sh <runda> [s1|s2|s3|oba|wszyscy]   np. bash scripts/trenuj_uczniow_eseju.sh v1
#   dane: data/sft/train-esej-nauczyciel.jsonl (scripts/dane_eseju_sft.py) kopiowane do data/sft/train-esej-<runda>.jsonl
#   S1: data/modele/bielik-1.5b-esej-<runda>/bielik-1.5b-esej-<runda>-Q8_0.gguf
#   S2: data/trening/qwen3.5-4b-lora-esej-<runda>/adapter-F16.gguf (llama-server -m <UD-IQ3_XXS> --lora <adapter>)
#   S3: data/trening/qwen3.5-2b-lora-esej-<runda>/adapter-F16.gguf (llama-server -m <Qwen3.5-2B-Q4_K_M> --lora <adapter>)
set -euo pipefail
cd "$(dirname "$0")/.."
RUNDA=${1:?runda, np. v1}; KTO=${2:-oba}
LOG=review/esej-sft-2026-09-27/trening; mkdir -p "$LOG"
DANE=data/sft/train-esej-$RUNDA.jsonl
cp "${ZBIOR:-data/sft/train-esej-nauczyciel.jsonl}" "$DANE"   # ZBIOR: inny zbiór (runda kb), MAX_LEN: 4096 przy materiale
echo "$(date +%H:%M:%S) runda $RUNDA: $(wc -l < "$DANE") przykładów" | tee -a "$LOG/runda-$RUNDA.log"
QUANT=data/bin/llama-cuda/llama-quantize

if [[ $KTO == s1 || $KTO == oba || $KTO == wszyscy ]]; then
  N=bielik-1.5b-esej-$RUNDA
  uv run python -m matura.trening trenuj --baza data/hf/Bielik-1.5B-v3.0-Instruct --nazwa "$N" --dane "$DANE" \
    --epoki "${S1_EPOKI:-3}" --lr "${S1_LR:-2e-5}" --batch 8 --max-len "${MAX_LEN:-3072}" > "$LOG/$N.log" 2>&1
  uv run python -m matura.trening eksport --nazwa "$N" --kwanty Q8_0 --quantize "$QUANT" >> "$LOG/$N.log" 2>&1
  mkdir -p "data/modele/$N"; ln -sf "../../trening/$N/gguf/$N-Q8_0.gguf" "data/modele/$N/$N-Q8_0.gguf"
  grep -o "'eval_loss': '[0-9.]*'" "$LOG/$N.log" | tr '\n' ' ' | sed "s/^/$(date +%H:%M:%S) S1 $N eval_loss: /" | tee -a "$LOG/runda-$RUNDA.log"; echo
fi

if [[ $KTO == s2 || $KTO == oba || $KTO == wszyscy ]]; then
  N=qwen3.5-4b-lora-esej-$RUNDA
  uv run python -m matura.trening trenuj --baza data/hf/Qwen3.5-4B --nazwa "$N" --dane "$DANE" --lora "${S2_R:-16}" \
    --epoki "${S2_EPOKI:-3}" --lr "${S2_LR:-2e-4}" --batch 4 --max-len "${MAX_LEN:-3072}" > "$LOG/$N.log" 2>&1
  uv run python data/llama.cpp/convert_lora_to_gguf.py --base data/hf/Qwen3.5-4B --outtype f16 \
    --outfile "data/trening/$N/adapter-F16.gguf" "data/trening/$N/adapter" >> "$LOG/$N.log" 2>&1
  grep -o "'eval_loss': '[0-9.]*'" "$LOG/$N.log" | tr '\n' ' ' | sed "s/^/$(date +%H:%M:%S) S2 $N eval_loss: /" | tee -a "$LOG/runda-$RUNDA.log"; echo
fi
if [[ $KTO == s3 || $KTO == oba || $KTO == wszyscy ]]; then
  # S3: Qwen3.5-2B (już w zestawie jako opisywacz Q4_K_M) + LoRA: bez wzrostu rozmiaru i bez zmiany bazy przyrostu
  N=qwen3.5-2b-lora-esej-$RUNDA
  uv run python -m matura.trening trenuj --baza data/hf/Qwen3.5-2B --nazwa "$N" --dane "$DANE" --lora "${S3_R:-16}" \
    --epoki "${S3_EPOKI:-3}" --lr "${S3_LR:-2e-4}" --batch 4 --max-len "${MAX_LEN:-3072}" > "$LOG/$N.log" 2>&1
  uv run python data/llama.cpp/convert_lora_to_gguf.py --base data/hf/Qwen3.5-2B --outtype f16 \
    --outfile "data/trening/$N/adapter-F16.gguf" "data/trening/$N/adapter" >> "$LOG/$N.log" 2>&1
  grep -o "'eval_loss': '[0-9.]*'" "$LOG/$N.log" | tr '\n' ' ' | sed "s/^/$(date +%H:%M:%S) S3 $N eval_loss: /" | tee -a "$LOG/runda-$RUNDA.log"; echo
fi
echo "$(date +%H:%M:%S) runda $RUNDA gotowa" | tee -a "$LOG/runda-$RUNDA.log"
