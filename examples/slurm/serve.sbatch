#!/bin/bash
# Template only: fill in the <...> placeholders for your cluster. Starts the plain server or the edit engine as the
# job's main step and keeps the allocation for the tools (run them as extra steps, see below).
#
#   sbatch --export=ALL,MODE=engine examples/slurm/serve.sbatch     # MODE=plain (default) or engine
#
#SBATCH --job-name=dsv41
#SBATCH --partition=<gpu partition>
#SBATCH --account=<account>
#SBATCH --qos=<qos>
#SBATCH --nodes=1
#SBATCH --gres=gpu:<b200 or h200>:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=700G
#SBATCH --time=<hh:mm:ss>
#SBATCH --output=<log dir>/dsv41_%j.out
set -uo pipefail
cd "<path to this repo>"
case "${MODE:-plain}" in
  plain) exec bash scripts/serve_plain.sh ;;
  engine) exec bash scripts/serve_engine.sh ;;
  *) echo "MODE must be plain or engine" >&2; exit 2 ;;
esac
# Tools inside the same allocation, from a login shell:
#   srun --jobid <jobid> --overlap -N1 -n1 -c 4 bash -lc 'cd <repo> && . scripts/common.sh && python tools/client.py installed'
# Or launch the server itself as an overlap step of an interactive allocation:
#   srun --jobid <jobid> --overlap -N1 -n1 -c 32 bash scripts/serve_engine.sh
