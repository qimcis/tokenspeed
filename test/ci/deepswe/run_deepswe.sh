#!/usr/bin/env bash
set -euo pipefail

DEEPSWE_REPOSITORY=https://github.com/datacurve-ai/deep-swe.git
DEEPSWE_REVISION=0b9fabbb63b9104d678fe965e1632f2dd9eaa2ea
KIMI_CODE_VERSION=0.29.0

task_count=${DEEPSWE_TASK_COUNT:-10}
sample_seed=${DEEPSWE_SAMPLE_SEED:-0}
concurrency=${DEEPSWE_CONCURRENCY:-4}
minimum_score=${DEEPSWE_MINIMUM_SCORE:-0.0}
model=${DEEPSWE_MODEL:-kimi/kimi-k3}
max_context=${DEEPSWE_MAX_CONTEXT_SIZE:-1048576}
max_completion=${DEEPSWE_MAX_COMPLETION_TOKENS:-32768}

[[ "$task_count" =~ ^[0-9]+$ ]] && (( task_count >= 1 && task_count <= 113 )) || {
  echo "DEEPSWE_TASK_COUNT must be between 1 and 113" >&2
  exit 2
}
[[ "$sample_seed" =~ ^-?[0-9]+$ ]] || {
  echo "DEEPSWE_SAMPLE_SEED must be an integer" >&2
  exit 2
}
[[ "$concurrency" =~ ^[0-9]+$ ]] && (( concurrency >= 1 && concurrency <= 16 )) || {
  echo "DEEPSWE_CONCURRENCY must be between 1 and 16" >&2
  exit 2
}
if [[ -n "${DEEPSWE_BASE_URL:-}" ]]; then
  base_url=${DEEPSWE_BASE_URL}
else
  [[ "${POD_IP:-}" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]] || {
    echo "POD_IP or DEEPSWE_BASE_URL is required" >&2
    exit 2
  }
  base_url=http://${POD_IP}/v1
fi
[[ "$base_url" =~ ^http://[a-zA-Z0-9._-]+(:80)?/v1$ ]] || {
  echo "DEEPSWE_BASE_URL must be an HTTP /v1 endpoint on port 80" >&2
  exit 2
}
[[ "$max_context" =~ ^[0-9]+$ ]] && (( max_context > 0 )) || {
  echo "DEEPSWE_MAX_CONTEXT_SIZE must be positive" >&2
  exit 2
}
[[ "$max_completion" =~ ^[0-9]+$ ]] && (( max_completion > 0 )) || {
  echo "DEEPSWE_MAX_COMPLETION_TOKENS must be positive" >&2
  exit 2
}
[[ -n "${PIER_KIMI_CODE_BINARY:-}" ]] || {
  echo "PIER_KIMI_CODE_BINARY is required" >&2
  exit 2
}
[[ -x "$PIER_KIMI_CODE_BINARY" ]] || {
  echo "PIER_KIMI_CODE_BINARY must be executable: $PIER_KIMI_CODE_BINARY" >&2
  exit 2
}
actual_kimi_version=$("$PIER_KIMI_CODE_BINARY" --version)
[[ "$actual_kimi_version" == "$KIMI_CODE_VERSION" ]] || {
  echo "Expected Kimi Code $KIMI_CODE_VERSION, got $actual_kimi_version" >&2
  exit 2
}

docker info >/dev/null
docker compose version
curl -fsS --max-time 10 "${base_url%/v1}/readiness" >/dev/null

artifact_dir=${PWD}/.ci-artifacts/deepswe
jobs_dir=${artifact_dir}/jobs
dataset_root=${RUNNER_TEMP:-/tmp}/deep-swe-${GITHUB_RUN_ID:-local}-${GITHUB_RUN_ATTEMPT:-1}
job_name=deepswe-${GITHUB_RUN_ID:-local}-${GITHUB_RUN_ATTEMPT:-1}
mkdir -p "$artifact_dir" "$jobs_dir"
{
  printf 'version=%s\n' "$actual_kimi_version"
  sha256sum "$PIER_KIMI_CODE_BINARY"
} > "${artifact_dir}/kimi-code.txt"

git clone --filter=blob:none --no-checkout "$DEEPSWE_REPOSITORY" "$dataset_root"
git -C "$dataset_root" checkout --detach "$DEEPSWE_REVISION"

export PYTHONPATH="${PWD}/test/ci/deepswe${PYTHONPATH:+:${PYTHONPATH}}"
# Pier emits naive timestamps; make the offline deadline join unambiguous.
export TZ=UTC
pier_status=0
pier run \
  --path "$dataset_root/tasks" \
  --job-name "$job_name" \
  --jobs-dir "$jobs_dir" \
  --agent-import-path kimi_code_pier_agent:KimiCodeAgent \
  --model "$model" \
  --agent-kwarg "version=${KIMI_CODE_VERSION}" \
  --agent-env "KIMI_MODEL_BASE_URL=${base_url}" \
  --agent-env KIMI_MODEL_API_KEY=EMPTY \
  --agent-env "KIMI_MODEL_MAX_CONTEXT_SIZE=${max_context}" \
  --agent-env "KIMI_MODEL_MAX_COMPLETION_TOKENS=${max_completion}" \
  --agent-env KIMI_MODEL_THINKING_EFFORT=max \
  --environment-import-path kimi_code_environment:KimiCodeDockerEnvironment \
  --n-concurrent "$concurrency" \
  --max-retries 1 \
  --cpus ignore \
  --memory ignore \
  --n-tasks "$task_count" \
  --sample-seed "$sample_seed" \
  --yes || pier_status=$?

result=${jobs_dir}/${job_name}/result.json
summary=${artifact_dir}/summary.md
cp "$result" "${artifact_dir}/result.json"
summary_status=0
python3 test/ci/deepswe/summarize.py \
  "$result" \
  --minimum-score "$minimum_score" \
  --output "$summary" || summary_status=$?
if (( pier_status != 0 )); then
  exit "$pier_status"
fi
exit "$summary_status"
