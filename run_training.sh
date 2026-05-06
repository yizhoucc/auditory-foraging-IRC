#!/bin/bash
#SBATCH --job-name=af_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2080Ti:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=6:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#
# Usage:
#   On a Slurm cluster:
#     sbatch run_training.sh              # submit as batch job
#   Locally (no Slurm):
#     bash run_training.sh                # run directly
#
# Trains all notebooks in parallel (8 at a time).
# Models are saved to ycstore/.
# Logs go to logs/<script_name>.log.

set -e

REPO="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO"
mkdir -p ycstore logs

# Activate venv if it exists
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -n "$VIRTUAL_ENV" ]; then
    : # already in a venv
else
    echo "Warning: no venv found. Using system python."
fi

echo "Python: $(python3 --version)"
echo "Repo: $REPO"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'CPU only')"

# Collect all training scripts
SCRIPTS=()
for f in notebooks/*.py yctest/*.py; do
    [ -f "$f" ] && SCRIPTS+=("$f")
done

echo "Found ${#SCRIPTS[@]} training scripts"
echo "Starting at $(date)"

MAX_PARALLEL=${MAX_PARALLEL:-8}

for script in "${SCRIPTS[@]}"; do
    name=$(basename "$script" .py)
    echo "[$(date +%H:%M:%S)] Starting: $script"
    python3 "$script" > "logs/${name}.log" 2>&1 &

    # Limit parallel processes
    while [ "$(jobs -r | wc -l)" -ge "$MAX_PARALLEL" ]; do
        sleep 5
    done
done

wait
echo ""
echo "All training complete at $(date)"

# Summary
DONE=$(grep -rl "Done:" logs/*.log 2>/dev/null | wc -l)
FAIL=0
for f in logs/*.log; do
    if grep -q "Traceback" "$f" 2>/dev/null && ! grep -q "Done:" "$f" 2>/dev/null; then
        FAIL=$((FAIL + 1))
        echo "FAILED: $(basename "$f" .log)"
    fi
done
echo "Results: $DONE succeeded, $FAIL failed out of ${#SCRIPTS[@]}"

# Pack models
if [ -d ycstore ] && ls ycstore/*.zip 1>/dev/null 2>&1; then
    ZIPS=$(ls ycstore/*.zip | wc -l)
    tar czf ycstore.tar.gz ycstore/
    echo "Packed $ZIPS models into ycstore.tar.gz ($(du -sh ycstore.tar.gz | cut -f1))"
fi
