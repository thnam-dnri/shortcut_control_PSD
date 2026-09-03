#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/mnt/Data/ML_DetA"
SESSION_ID="${1:?usage: $0 SESSION_ID [DAQ_TMUX_SESSION]}"
DAQ_TMUX_SESSION="${2:-DAQ_CO60_${SESSION_ID}}"
POLL_SECONDS="${POLL_SECONDS:-30}"
REFERENCE_ROOT="${REFERENCE_ROOT:-${PROJECT_DIR}/raw_data/th232_preamp_250msps_20260813_195521_thr10_1.root}"
SESSION_LOG="${PROJECT_DIR}/raw_data/co60_preamp_250msps_session_${SESSION_ID}.log"
MONITOR_LOG="${PROJECT_DIR}/raw_data/co60_preamp_250msps_monitor_${SESSION_ID}.log"
QC_OUTPUT_DIR="${PROJECT_DIR}/outputs/data_quality"

mkdir -p "${PROJECT_DIR}/raw_data" "${QC_OUTPUT_DIR}"
exec > >(tee -a "${MONITOR_LOG}") 2>&1

echo "=== Co-60 DAQ monitor ==="
echo "monitor_started=$(date --iso-8601=seconds)"
echo "session_id=${SESSION_ID}"
echo "daq_tmux_session=${DAQ_TMUX_SESSION}"
echo "session_log=${SESSION_LOG}"
echo "reference_root=${REFERENCE_ROOT}"

if [[ ! -f "${SESSION_LOG}" ]]; then
    echo "ERROR: session log does not exist yet: ${SESSION_LOG}" >&2
    exit 2
fi
if [[ ! -f "${REFERENCE_ROOT}" ]]; then
    echo "ERROR: reference ROOT file does not exist: ${REFERENCE_ROOT}" >&2
    exit 2
fi

completion_marker='DAQ_SESSION_FINISHED files=10 total_events=1000000'
while :; do
    if grep -Fq "${completion_marker}" "${SESSION_LOG}"; then
        echo "DAQ completion marker detected at $(date --iso-8601=seconds)"
        break
    fi

    if ! tmux has-session -t "${DAQ_TMUX_SESSION}" 2>/dev/null; then
        echo "ERROR: DAQ tmux session ended before completion marker" >&2
        tail -n 40 "${SESSION_LOG}" || true
        exit 1
    fi

    latest_event=$(grep -oE 'Evt=[0-9]+' "${SESSION_LOG}" | tail -1 || true)
    latest_file=$(grep -oE -- '--- (Starting|Completed) file [0-9]+/10 ---' "${SESSION_LOG}" | tail -1 || true)
    echo "monitor=$(date --iso-8601=seconds) ${latest_file:-file status unavailable} ${latest_event:-event status unavailable}"
    sleep "${POLL_SECONDS}"
done

# Let the final tee output and ROOT close before collecting paths.
for _ in $(seq 1 60); do
    if ! tmux has-session -t "${DAQ_TMUX_SESSION}" 2>/dev/null; then
        break
    fi
    sleep 2
done

mapfile -t ROOT_FILES < <(
    grep -oE 'output_root=[^[:space:]]+\.root' "${SESSION_LOG}" \
        | sed 's/^output_root=//' \
        | awk '!seen[$0]++'
)
if (( ${#ROOT_FILES[@]} != 10 )); then
    echo "ERROR: expected 10 unique ROOT paths, found ${#ROOT_FILES[@]}" >&2
    printf '%s\n' "${ROOT_FILES[@]}"
    exit 1
fi
for root_file in "${ROOT_FILES[@]}"; do
    if [[ ! -s "${root_file}" ]]; then
        echo "ERROR: missing or empty ROOT file: ${root_file}" >&2
        exit 1
    fi
done

echo "ROOT files collected: ${#ROOT_FILES[@]}"
printf '  %s\n' "${ROOT_FILES[@]}"

date_part="${SESSION_ID%%_*}"
time_part="${SESSION_ID##*_}"
QC_SESSION_ID="session_${date_part}_co60_1m_${time_part}"

# Native PyROOT runtime required by check_waveform_quality.py.
source /mnt/Data/FADC500/root/bin/thisroot.sh
source /home/adminministrator/Other/data_engine/analysis-venv/bin/activate

set +e
python "${PROJECT_DIR}/scripts/check_waveform_quality.py" \
    --root-files "${ROOT_FILES[@]}" \
    --reference-root "${REFERENCE_ROOT}" \
    --output-dir "${QC_OUTPUT_DIR}" \
    --session-id "${QC_SESSION_ID}" \
    --expected-files 10 \
    --sample-events 1000 \
    --sample-mode first
QC_RC=$?
set -e

echo "QC exit code=${QC_RC}"
REPORT="${QC_OUTPUT_DIR}/${QC_SESSION_ID}/session_qc_report.json"
if [[ ! -s "${REPORT}" ]]; then
    echo "ERROR: QC report was not created: ${REPORT}" >&2
    exit 1
fi

# The requested gate is no per-file FAIL. WARN is allowed; FAIL is not.
if ! python - "${REPORT}" <<'PY'
import json
import sys

report_path = sys.argv[1]
report = json.load(open(report_path, encoding="utf-8"))
files = report.get("files", [])
bad = [item.get("file_name", item.get("file", "<unknown>")) for item in files if item.get("status") == "FAIL"]
if report.get("root_file_count") != 10 or len(files) != 10:
    print(f"QC GATE BLOCKED: file count report={report.get('root_file_count')} rows={len(files)}", file=sys.stderr)
    raise SystemExit(1)
if bad:
    print("QC GATE BLOCKED: per-file FAIL: " + ", ".join(bad), file=sys.stderr)
    raise SystemExit(1)
print(f"QC GATE PASSED: no per-file FAIL; overall status={report.get('status')}")
PY
then
    echo "No new Co-60 session will be started."
    exit 0
fi

NEXT_SESSION_ID=$(date +%Y%m%d_%H%M%S)
NEXT_TMUX_SESSION="DAQ_CO60_${NEXT_SESSION_ID}"
if tmux has-session -t "${NEXT_TMUX_SESSION}" 2>/dev/null; then
    echo "ERROR: next tmux session already exists: ${NEXT_TMUX_SESSION}" >&2
    exit 1
fi

tmux new-session -d -s "${NEXT_TMUX_SESSION}" \
    "cd '${PROJECT_DIR}' && exec env SOURCE_LABEL=Co60 SESSION_ID='${NEXT_SESSION_ID}' EVENTS_PER_FILE=100000 FILES_PER_SESSION=10 TIMEOUT_SECONDS=2400 THRESHOLD_ADC=10 bash scripts/run_1m_daq_session.sh"

echo "Started next Co-60 DAQ session in background"
echo "next_session_id=${NEXT_SESSION_ID}"
echo "next_tmux_session=${NEXT_TMUX_SESSION}"
echo "monitor_finished=$(date --iso-8601=seconds)"
