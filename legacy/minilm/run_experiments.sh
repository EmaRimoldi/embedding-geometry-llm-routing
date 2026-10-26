#!/bin/bash
# minilm/ study: submit the experiments behind Sections 1-2 of the paper (Claims 1-13, Checks 1-2) for
#   rb-minilm   RouterBench 0-shot (standard 34,661 rows)            x MiniLM-L12
#   lrb-minilm  LLMRouterBench performance-cost, rows <= 512 tokens x MiniLM-L12
#   lrb-gte     the same rows x GTE (the reference on identical rows)
# Jobs are grouped into sequential command files (minilm/seq.sbatch) to stay under the 256-job submit limit.
# Usage: bash minilm/run_experiments.sh [embed_job_id]   (DRY=1 prints the command files and submits nothing)
set -euo pipefail
E4=__E4_ROOT__; cd "$E4"
DEP=""; [ -n "${1:-}" ] && DEP="--dependency=afterok:$1"
declare -A CFG=([rb-minilm]="routerbench-0shot computed/minilm-l12@prompt" [lrb-minilm]="llmrouterbench-performance-cost-le512 computed/minilm-l12@prompt" [lrb-gte]="llmrouterbench-performance-cost-le512 computed/gte-qwen2-7b@prompt")
for cfg in rb-minilm lrb-minilm lrb-gte; do
  set -- ${CFG[$cfg]}; panel=$1; rep=$2; repname=${rep#computed/}; repname=${repname%@*}
  RES=$E4/minilm/results/${panel#llmrouterbench-}__$repname; J=$RES/jobs; mkdir -p "$J"
  ENV="ALL,E4_PANEL=$panel,E4_REP=$rep,E4_RESULTS_ROOT=$RES"; RB=$([ "$cfg" = rb-minilm ] && echo 1 || echo 0)
  A=()
  sub() { if [ "${DRY:-0}" = 1 ]; then echo "--- $1 ($2)"; cat "$2"; else A+=("$(sbatch --parsable $DEP --job-name="m-$cfg-$1" --export="$ENV,CMDFILE=$2" minilm/seq.sbatch)"); fi; }
  { for s in exp01_kmeans7_vs_families exp02_kmeans7_balanced exp04_mmlu_topic exp05_knn_family_neighbours exp06_knn_family_neighbours_balanced exp07_kmeans7_bagofwords_control; do echo "\$PY scripts/$s.py"; done
    [ $RB = 1 ] && echo "\$PY scripts/exp21_cross_format_math.py"; } > "$J/A1_geometry.txt"; sub A1geom "$J/A1_geometry.txt"
  echo "\$PY scripts/exp08_basic_model_tests.py" > "$J/A2_exp08.txt"; sub A2exp08 "$J/A2_exp08.txt"
  { for s in exp10_outcome_locality exp11_outcome_locality_split exp12_outcome_locality_topic; do echo "\$PY scripts/$s.py"; done; } > "$J/A3_locality.txt"; sub A3loc "$J/A3_locality.txt"
  { echo "\$PY scripts/exp14_15_locality_lambda.py --cell family"; echo "\$PY scripts/exp14_15_locality_lambda.py --cell topic"; [ $RB = 1 ] && echo "\$PY scripts/exp16_hellaswag_check.py"; } > "$J/A4_cells.txt"; sub A4cells "$J/A4_cells.txt"
  for seed in 0 1 2 3 4; do
    echo "\$PY scripts/exp20_topic_recovery.py --seed $seed" > "$J/A5_exp20_s$seed.txt"; sub exp20s$seed "$J/A5_exp20_s$seed.txt"
    { for l in $(seq 0 14); do echo "\$PY scripts/exp09_mlp_fit.py --seed $seed --lam-index $l"; done; } > "$J/A6_exp09_s$seed.txt"; sub exp09s$seed "$J/A6_exp09_s$seed.txt"
    if [ $RB = 1 ]; then { for l in $(seq 0 14); do for a in 0 1 2; do echo "\$PY scripts/exp13_mlp_reg_fit.py --seed $seed --lam-index $l --alpha-index $a"; done; done; } > "$J/A7_exp13_s$seed.txt"; sub exp13s$seed "$J/A7_exp13_s$seed.txt"; fi
  done
  if [ $RB = 1 ]; then
    { for probe in hard_examples format_invariance technique_twins deformatting; do mkdir -p "$RES/$probe"; echo "E4_OUT=$RES/$probe \$PY results/$probe/analyse.py"; done
      echo "E4_OUT=$RES/format_invariance \$PY results/format_invariance/twin_vs_panel.py"; echo "E4_OUT=$RES/technique_twins \$PY results/technique_twins/mates_vs_panel.py"; echo "E4_OUT=$RES/deformatting \$PY results/deformatting/family_confusion.py"; } > "$J/A8_probes.txt"; sub A8probes "$J/A8_probes.txt"
  fi
  { for s in exp08_choice_distribution exp17_topic_significance exp19_decision_agreement exp28_choice_by_lambda exp09_mlp_aggregate; do echo "\$PY scripts/$s.py"; done; echo "\$PY scripts/exp20_topic_recovery.py --aggregate"
    [ $RB = 1 ] && echo "\$PY scripts/exp13_mlp_reg_aggregate.py"; [ $RB = 0 ] && echo "\$PY scripts/exp33_llm_topics.py"; true; } > "$J/B_after.txt"
  if [ "${DRY:-0}" = 1 ]; then echo "--- B"; cat "$J/B_after.txt"; else
    sbatch --parsable --dependency=afterok:$(IFS=:; echo "${A[*]}") --job-name="m-$cfg-B" --export="$ENV,CMDFILE=$J/B_after.txt" minilm/seq.sbatch >/dev/null
    echo "$cfg: ${#A[@]} stage-A jobs + 1 stage-B job -> $RES"; fi
done
[ "${DRY:-0}" = 1 ] || squeue -u "$USER" -h -o "%j %T" | grep "^m-" | awk '{print $2}' | sort | uniq -c
