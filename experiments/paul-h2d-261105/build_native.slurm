#!/usr/bin/env bash
#SBATCH --job-name=mahonem-paul-h2d-native-build
#SBATCH --partition=nvidia-part
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=768G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
set -euo pipefail
export JNI_REVISION=${JNI_REVISION:?}
/cm/local/apps/slurm/current/bin/srun --kill-on-bad-exit=1 \
  --container-image=nvcr.io#nvidia/cuda:13.1.0-devel-ubuntu24.04 \
  --container-mounts=/home/mahonem:/home/mahonem,/home/prestouser:/home/prestouser \
  --container-remap-root --container-writable \
  bash /home/mahonem/paul_hybrid_multithreaded_261105/src/cudf-spark-jni/experiments/paul-h2d-261105/build_native.sh
