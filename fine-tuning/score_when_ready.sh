#!/bin/bash
# Wait for a run's generations, then score them with GLM-5.3 on the SDU endpoint.
# Usage: score_when_ready.sh <low|med>
set -u
RUN=$1
D=/work/moltbook

cd $D
source .venv/bin/activate
export SCORER_API_KEY=$(cat $D/.scorer_key)
export SCORER_BASE_URL=https://ai.cloud.sdu.dk/v1

until [ -s "$D/gen_wiki_$RUN.json" ]; do
  sleep 120
done

python score_eval.py "$D/gen_wiki_$RUN.json" "$D/scored_wiki_$RUN.json" >> "$D/score_$RUN.log" 2>&1
echo "SCORING DONE $RUN" >> "$D/score_$RUN.log"
