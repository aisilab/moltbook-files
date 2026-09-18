#!/bin/bash
# Wait for one training run to finish, then evaluate it on the same GPU.
# Usage: post_train.sh <gpu> <low|med>
set -u
GPU=$1
RUN=$2
D=/work/moltbook
A=$D/wiki_$RUN/adapter

cd $D
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=$GPU

until [ -f "$A/adapter_model.safetensors" ] && ! pgrep -f "wikipedia_control_config_$RUN" > /dev/null; do
  sleep 120
done

if ! grep -q "saved adapter" $D/wiki_$RUN.log; then
  echo "TRAINING FAILED for $RUN" >> $D/eval_$RUN.log
  exit 1
fi

{
  echo "=== TruthfulQA $RUN ==="
  lm_eval --model hf \
    --model_args "pretrained=Qwen/Qwen2.5-14B-Instruct,peft=$A,dtype=bfloat16" \
    --tasks truthfulqa_mc1,truthfulqa_mc2 --batch_size 8 \
    --output_path $D/tqa_$RUN

  echo "=== Generation $RUN ==="
  python gen_eval.py $A $D/gen_wiki_$RUN.json

  echo "EVAL DONE $RUN"
} >> $D/eval_$RUN.log 2>&1
