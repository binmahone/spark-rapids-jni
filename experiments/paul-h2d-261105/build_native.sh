#!/usr/bin/env bash
set -euo pipefail
readonly own=/home/mahonem/paul_hybrid_multithreaded_261105
readonly legacy=/home/prestouser/mahonem/wild_spark_rapids
readonly src=$own/src/cudf-spark-jni
readonly logs=$own/logs/job-${SLURM_JOB_ID:?}
mkdir -p "$logs" "$own/tools" "$own/maven-repository"
df -hT / /home/mahonem > "$logs/capacity-before.txt"
test "$(df -PB1 /home/mahonem | awk 'NR==2 {print $4}')" -gt 322122547200
test "$(df -PB1 / | awk 'NR==2 {print $4}')" -gt 21474836480
export DEBIAN_FRONTEND=noninteractive
timeout 180 apt-get update -qq -o Acquire::Retries=2 -o Acquire::http::Timeout=30
timeout 240 apt-get install -y -qq --no-install-recommends \
  openjdk-17-jdk python3 git ninja-build curl unzip zip patch \
  zlib1g-dev libnuma-dev libboost-filesystem-dev build-essential
test "$(git -C "$src" rev-parse HEAD)" = "${JNI_REVISION:?}"
test -z "$(git -C "$src" status --short)"
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-arm64
export PATH="$legacy/tools/maven/bin:$JAVA_HOME/bin:$PATH"
readonly cmake=$own/tools/cmake-4.0.3-linux-aarch64
if [[ ! -x "$cmake/bin/cmake" ]]; then
  cd "$own/tools"
  curl -fL --retry 2 -o cmake-4.0.3-linux-aarch64.tar.gz \
    https://github.com/Kitware/CMake/releases/download/v4.0.3/cmake-4.0.3-linux-aarch64.tar.gz
  curl -fL --retry 2 -o cmake-4.0.3-SHA-256.txt \
    https://github.com/Kitware/CMake/releases/download/v4.0.3/cmake-4.0.3-SHA-256.txt
  python3 - <<'PY'
import hashlib
from pathlib import Path
name = 'cmake-4.0.3-linux-aarch64.tar.gz'
expected = next(line.split()[0] for line in Path('cmake-4.0.3-SHA-256.txt').read_text().splitlines()
                if line.split()[-1].lstrip('*') == name)
assert hashlib.file_digest(Path(name).open('rb'), 'sha256').hexdigest() == expected
PY
  tar -xzf cmake-4.0.3-linux-aarch64.tar.gz
  rm cmake-4.0.3-linux-aarch64.tar.gz
fi
export PATH="$cmake/bin:$PATH"
export MAVEN_OPTS='-Xmx16g -XX:ReservedCodeCacheSize=512m'
cd "$src"
git submodule update --init --recursive --depth 1 --filter=blob:none
test "$(git -C thirdparty/cudf rev-parse HEAD)" = d984f0a215980763781ae1b26be3816875c18cd9
{
  git rev-parse HEAD
  git submodule status
  gcc --version
  nvcc --version
  cmake --version
  mvn --version
} > "$logs/build-environment.txt" 2>&1
monitor_capacity() {
  echo 'epoch,home_available_bytes,root_available_bytes' > "$logs/capacity.csv"
  while true; do
    echo "$(date +%s),$(df -PB1 /home/mahonem | awk 'NR==2 {print $4}'),$(df -PB1 / | awk 'NR==2 {print $4}')" >> "$logs/capacity.csv"
    sleep 15
  done
}
monitor_capacity &
readonly monitor_pid=$!
trap 'kill "$monitor_pid" 2>/dev/null || true' EXIT
mvn -B -ntp -Dmaven.repo.local="$own/maven-repository" -Parm64 \
  -Dcuda.version=cuda13 -DUSE_GDS=OFF -DBUILD_FAULTINJ=OFF -DBUILD_PROFILER=OFF \
  -DBUILD_TESTS=OFF -DBUILD_BENCHMARKS=OFF -DBUILD_CUDF_TESTS=OFF \
  -DBUILD_CUDF_BENCHMARKS=OFF -DCPP_PARALLEL_LEVEL=48 \
  -DCMAKE_CUDA_ARCHITECTURES=100-real -Dlibcudf.build.configure=false \
  -Dtest=PackedHostToDeviceCopyTest -DfailIfNoTests=true install \
  2>&1 | tee "$logs/jni-build-and-tests.log"
readonly jar=$own/maven-repository/com/nvidia/cudf-spark-jni/26.12.0-SNAPSHOT/cudf-spark-jni-26.12.0-SNAPSHOT-cuda13-arm64.jar
test -s "$jar"
sha256sum "$jar" > "$logs/jni-jar.sha256"
mkdir -p "$own/artifacts"
artifact_sha=$(sha256sum "$jar" | awk '{print $1}')
cp --reflink=never --no-clobber "$jar" "$own/artifacts/$artifact_sha.jar"
df -hT / /home/mahonem > "$logs/capacity-after.txt"
