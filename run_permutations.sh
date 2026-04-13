#!/usr/bin/env bash
# Run from repo root, e.g. ubuntu@host:~/state_lora$ ./run_permutations.sh
#
# Defaults (override with env vars):
#   GENE_EMBEDDINGS_FILE  ->  <repo>/data/all_embeddings.pt  (ESM-2 dict .pt for A=1 runs)
# Put Replogle .h5ad under e.g. ~/data/replogle and set [datasets] replogle in each TOML to that path.
#
# One-time on Ubuntu:
#   mkdir -p ~/state_lora/data ~/data/replogle
#   # copy or link your ESM-2 gene embeddings to:
#   #   ~/state_lora/data/all_embeddings.pt
#   # set each TOML [datasets] replogle = "/home/ubuntu/data/replogle" (or your path)

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${ROOT_DIR}/results"
TOML_DIR="${ROOT_DIR}/state_preprint_toml_files/replogle_tomls"
CELL_LINES=("hepg2" "jurkat" "k562" "rpe1")

# Placeholder-friendly defaults for a typical ~/state_lora checkout on Ubuntu
: "${GENE_EMBEDDINGS_FILE:=${ROOT_DIR}/data/all_embeddings.pt}"

mkdir -p "${ROOT_DIR}/data"
mkdir -p "${RESULTS_DIR}"

aggregate_run_metrics() {
  local run_dir="$1"
  local run_name="$2"
  local a="$3"
  local b="$4"
  local c="$5"
  local d="$6"

  python - <<'PY' "${run_dir}" "${run_name}" "${a}" "${b}" "${c}" "${d}"
import json
import sys
from pathlib import Path

import pandas as pd

run_dir = Path(sys.argv[1])
run_name = sys.argv[2]
a, b, c, d = map(int, sys.argv[3:7])

rows = []
for eval_dir in sorted(run_dir.glob("train_runs/*/eval_final.ckpt")):
    cell_line = eval_dir.parent.name
    metric_values = {}
    for csv_path in sorted(eval_dir.rglob("*.csv")):
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        if df.empty:
            continue
        numeric_df = df.select_dtypes(include=["number"])
        for col in numeric_df.columns:
            vals = numeric_df[col].dropna().tolist()
            if not vals:
                continue
            metric_values.setdefault(col, []).extend(float(v) for v in vals)

    row = {
        "run_name": run_name,
        "A": a,
        "B": b,
        "C": c,
        "D": d,
        "cell_line": cell_line,
    }
    for metric, values in metric_values.items():
        row[metric] = sum(values) / len(values)
    rows.append(row)

out_path = run_dir / "metrics.csv"
if rows:
    df_out = pd.DataFrame(rows)
    fixed = ["run_name", "A", "B", "C", "D", "cell_line"]
    metric_cols = sorted(c for c in df_out.columns if c not in fixed)
    df_out = df_out[fixed + metric_cols]
    df_out.to_csv(out_path, index=False)
else:
    pd.DataFrame(columns=["run_name", "A", "B", "C", "D", "cell_line"]).to_csv(out_path, index=False)
PY
}

for A in 0 1; do
  for B in 0 1; do
    for C in 0 1; do
      for D in 0 1; do
        RUN_NAME="A${A}_B${B}_C${C}_D${D}"
        RUN_DIR="${RESULTS_DIR}/${RUN_NAME}"

        if [[ -d "${RUN_DIR}" ]]; then
          echo "Skipping ${RUN_NAME}: ${RUN_DIR} already exists."
          continue
        fi

        mkdir -p "${RUN_DIR}/train_runs"
        LOG_FILE="${RUN_DIR}/train.log"

        {
          echo "============================================================"
          echo "Starting run ${RUN_NAME}"
          echo "============================================================"
          date
        } | tee -a "${LOG_FILE}"

        for CELL_LINE in "${CELL_LINES[@]}"; do
          TOML_PATH="${TOML_DIR}/${CELL_LINE}.toml"
          TRAIN_OUTPUT_DIR="${RUN_DIR}/train_runs"
          TRAIN_NAME="${CELL_LINE}"
          TRAIN_RUN_DIR="${TRAIN_OUTPUT_DIR}/${TRAIN_NAME}"

          if [[ ! -f "${TOML_PATH}" ]]; then
            echo "Missing TOML file: ${TOML_PATH}" | tee -a "${LOG_FILE}"
            exit 1
          fi

          TRAIN_CMD=(
            uv run state tx train
            "data.kwargs.toml_config_path=${TOML_PATH}"
            "data.kwargs.embed_key=X_hvg"
            "data.kwargs.pert_col=gene"
            "data.kwargs.cell_type_key=cell_type"
            "model=state"
            "model.kwargs.transformer_backbone_key=GPT2"
            "model.kwargs.hidden_dim=128"
            "model.kwargs.cell_set_len=32"
            "model.kwargs.n_encoder_layers=4"
            "model.kwargs.n_decoder_layers=4"
            "model.kwargs.batch_encoder=False"
            "model.kwargs.perturbation_mode=$([[ ${A} -eq 1 ]] && echo gene_embedding || echo one_hot)"
            "model.kwargs.use_lowrank_branch=$([[ ${B} -eq 1 ]] && echo True || echo False)"
            "model.kwargs.use_gated_hybrid=$([[ ${C} -eq 1 ]] && echo True || echo False)"
            "model.kwargs.use_soft_efficacy_weights=$([[ ${D} -eq 1 ]] && echo True || echo False)"
            "model.kwargs.lowrank_dim=32"
            "output_dir=${TRAIN_OUTPUT_DIR}"
            "name=${TRAIN_NAME}"
          )

          if [[ ${A} -eq 1 ]]; then
            if [[ ! -f "${GENE_EMBEDDINGS_FILE}" ]]; then
              echo "A=1 needs gene embeddings at: ${GENE_EMBEDDINGS_FILE}" | tee -a "${LOG_FILE}"
              echo "Copy your ESM-2 dict .pt there, or: export GENE_EMBEDDINGS_FILE=/path/to/all_embeddings.pt" | tee -a "${LOG_FILE}"
              exit 1
            fi
            TRAIN_CMD+=(
              "data.kwargs.pert_rep=embedding"
              "data.kwargs.perturbation_features_file=${GENE_EMBEDDINGS_FILE}"
            )
          fi

          {
            echo
            echo "[${RUN_NAME}] Training ${CELL_LINE}"
            printf 'Command: %q ' "${TRAIN_CMD[@]}"
            echo
          } | tee -a "${LOG_FILE}"

          "${TRAIN_CMD[@]}" >>"${LOG_FILE}" 2>&1

          PREDICT_CMD=(
            uv run state tx predict
            --output-dir "${TRAIN_RUN_DIR}"
            --checkpoint final.ckpt
          )

          {
            echo "[${RUN_NAME}] Evaluating ${CELL_LINE}"
            printf 'Command: %q ' "${PREDICT_CMD[@]}"
            echo
          } | tee -a "${LOG_FILE}"

          "${PREDICT_CMD[@]}" >>"${LOG_FILE}" 2>&1
        done

        aggregate_run_metrics "${RUN_DIR}" "${RUN_NAME}" "${A}" "${B}" "${C}" "${D}"
      done
    done
  done
done

python - <<'PY' "${RESULTS_DIR}"
import re
import sys
from pathlib import Path

import pandas as pd

results_dir = Path(sys.argv[1])
metric_paths = sorted(results_dir.glob("A*_B*_C*_D*/metrics.csv"))
frames = []
for path in metric_paths:
    try:
        df = pd.read_csv(path)
    except Exception:
        continue
    if df.empty:
        continue
    frames.append(df)

if not frames:
    out = results_dir / "summary.csv"
    pd.DataFrame(columns=["run_name", "A", "B", "C", "D", "cell_line"]).to_csv(out, index=False)
    print(f"No metrics found. Wrote empty summary to {out}")
    sys.exit(0)

summary = pd.concat(frames, ignore_index=True)

metric_col = None
priority = [
    "effect_size_spearman",
    "effect_size_spearman_correlation",
    "spearman_effect_size",
    "de_effect_size_spearman",
]
for c in priority:
    if c in summary.columns:
        metric_col = c
        break
if metric_col is None:
    for c in summary.columns:
        lc = c.lower()
        if "effect" in lc and "spearman" in lc:
            metric_col = c
            break

if metric_col is not None:
    summary = summary.sort_values(metric_col, ascending=False, kind="mergesort")

fixed = ["run_name", "A", "B", "C", "D", "cell_line"]
metric_cols = [c for c in summary.columns if c not in fixed]
summary = summary[fixed + metric_cols]

summary_path = results_dir / "summary.csv"
summary.to_csv(summary_path, index=False)

print(summary.to_string(index=False))
print(f"\nSaved summary to {summary_path}")
PY
