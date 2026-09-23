#!/bin/bash
# Control experiments for the rebuttal: Wikipedia at three adaptation levels and a second Moltbook
# seed. Trains on four GPUs, pushes each adapter to the Hub as soon as it exists, and evaluates it
# together with every published adapter, so all rows of Table 2 come from one machine.
#
# Needs HF_TOKEN (write access to filter-with-espresso), WANDB_API_KEY and SCORER_API_KEY.
# Usage:  ./run_controls.sh <work_dir>
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
D=${1:-$PWD}
mkdir -p "$D" && cd "$D"
export WANDB_PROJECT=moltbook-wikipedia-control SCORER_BASE_URL=https://ai.cloud.sdu.dk/v1
B=Qwen/Qwen2.5-14B-Instruct
ORG=filter-with-espresso/Qwen2.5-14B-Instruct
declare -A REPO=([molt_low_s1]=$ORG-moltbook-finetune-v3-seed1 [wiki_low]=$ORG-wikipedia-control-low
                 [wiki_med]=$ORG-wikipedia-control-med [wiki_high]=$ORG-wikipedia-control-high)
RUNS="molt_low_s1 wiki_low wiki_med wiki_high"  # shortest first

log () { echo "[$(date '+%m-%d %H:%M')] $*" >> pipeline.log; }
mc1 () { python -c "import glob,json;f=sorted(glob.glob('tqa_$1/**/*.json',recursive=True))[-1];print(round(json.load(open(f))['results']['truthfulqa_mc1']['acc,none']*100,2))"; }
tail_loss () { python -c "import json;h=[x['loss'] for x in json.load(open('$1/log_history.json')) if 'loss' in x];print(round(sum(h[-5:])/len(h[-5:]),3))"; }
tqa () {  # gpu, name, model_args
  CUDA_VISIBLE_DEVICES=$1 lm_eval --model hf --model_args "$3" \
    --tasks truthfulqa_mc1,truthfulqa_mc2 --batch_size 8 --output_path tqa_$2 >> tqa_$2.log 2>&1
}
gen_score_push () {  # gpu, run
  CUDA_VISIBLE_DEVICES=$1 python "$HERE/gen_eval.py" $2/adapter gen_$2.json >> eval_$2.log 2>&1
  python "$HERE/score_eval.py" gen_$2.json scored_$2.json >> eval_$2.log 2>&1
  mkdir -p evalout_$2 && cp -r tqa_$2 gen_$2.json scored_$2.json evalout_$2/
  python "$HERE/push.py" evalout_$2 ${REPO[$2]} eval >> push.log 2>&1
}

# 1. data: the Wikipedia sample must come out the same (datasets==4.3.0), plus the Moltbook posts
python "$HERE/build_wiki.py" wikipedia_control.parquet > build.log 2>&1
grep -q "IDENTICAL SAMPLE" build.log || { log "STOP: Wikipedia sample does not match"; exit 1; }
python -c "
from datasets import load_dataset
df = load_dataset('aisilab/moltbook-files', split='train').to_pandas()
df[['title', 'content']].to_parquet('moltbook_posts.parquet', index=False)"

# 2. harness check: two known rows must reproduce (a difference above 1 point means a different setup)
tqa 0 base "pretrained=$B,dtype=bfloat16" &
tqa 1 moltlow "pretrained=$B,peft=$ORG-moltbook-finetune-v3,dtype=bfloat16" &
wait
BASE=$(mc1 base); MOLT=$(mc1 moltlow)
log "harness: base MC1 $BASE (expect 51.29), moltlow MC1 $MOLT (expect 36.60)"
[ "$(python -c "print(int(abs($BASE-51.29)<1.0 and abs($MOLT-36.60)<1.0))")" = 1 ] || { log "STOP: harness does not match"; exit 1; }

# 3. train each run on all four GPUs; push and evaluate it before starting the next one
for r in $RUNS; do
  log "training $r"
  torchrun --nproc_per_node 4 "$HERE/train_control.py" "$HERE/cfg_$r.json" $r > $r.log 2>&1
  [ -f $r/adapter/adapter_model.safetensors ] || { log "FAILED $r, see $r.log"; continue; }
  L=$(tail_loss $r)
  python -c "exit(0 if $L < 3 else 1)" || { log "DIVERGED $r (final loss $L), not pushed"; continue; }
  python "$HERE/push.py" $r/adapter ${REPO[$r]} >> push.log 2>&1
  tqa 0 $r "pretrained=$B,peft=$D/$r/adapter,dtype=bfloat16" &
  gen_score_push 1 $r &
  wait
  log "done $r: loss $L, MC1 $(mc1 $r), pushed ${REPO[$r]}"
done

# 4. re-measure the remaining published adapters on this machine
tqa 0 moltmed "pretrained=$B,peft=$ORG-moltbook-finetune-v5,dtype=bfloat16" &
tqa 1 molthigh "pretrained=$B,peft=$ORG-moltbook-finetune-v9,dtype=bfloat16" &
tqa 2 reddlow "pretrained=$B,peft=$ORG-reddit-baseline-v2-low,dtype=bfloat16" &
tqa 3 reddmed "pretrained=$B,peft=$ORG-reddit-baseline-v1,dtype=bfloat16" &
wait
tqa 0 reddhigh "pretrained=$B,peft=$ORG-reddit-baseline-v3-high,dtype=bfloat16"
for n in moltmed molthigh reddlow reddmed reddhigh; do log "published $n: MC1 $(mc1 $n)"; done
log "ALL DONE"
