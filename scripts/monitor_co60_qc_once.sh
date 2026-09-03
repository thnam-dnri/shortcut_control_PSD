#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/mnt/Data/ML_DetA}
SESSION_ID=${1:?Usage: monitor_co60_qc_once.sh SESSION_ID [REFERENCE_ROOT]}
REFERENCE_ROOT=${2:-$PROJECT_ROOT/raw_data/co60_preamp_250msps_20260814_180957_thr10_1.root}
POLL_SECONDS=${POLL_SECONDS:-15}

SESSION_LOG="$PROJECT_ROOT/raw_data/co60_preamp_250msps_session_${SESSION_ID}.log"
DAQ_TMUX="DAQ_CO60_${SESSION_ID}"
SESSION_DATE=${SESSION_ID%%_*}
SESSION_STAMP=${SESSION_ID##*_}
QC_ID="session_${SESSION_DATE}_co60_1m_${SESSION_STAMP}"
QC_OUTPUT_DIR="$PROJECT_ROOT/outputs/data_quality"
COMPLETION_MARKER='DAQ_SESSION_FINISHED files=10 total_events=1000000'

log() {
    printf '[%s] %s\n' "$(date --iso-8601=seconds)" "$*"
}

log "MONITOR_STARTED session_id=$SESSION_ID"
log "session_log=$SESSION_LOG"
log "reference_root=$REFERENCE_ROOT"

while true; do
    if grep -qF "$COMPLETION_MARKER" "$SESSION_LOG" 2>/dev/null; then
        log 'DAQ_COMPLETION_MARKER_FOUND'
        break
    fi

    if ! tmux has-session -t "$DAQ_TMUX" 2>/dev/null; then
        log 'DAQ_SESSION_ENDED_BEFORE_COMPLETION'
        exit 2
    fi

    sleep "$POLL_SECONDS"
done

mapfile -t ROOT_FILES < <(
    grep '^DAQ_FINISHED ' "$SESSION_LOG" \
        | sed -n 's/.*output_root=\([^[:space:]]*\.root\).*/\1/p' \
        | awk '!seen[$0]++'
)

if [[ ${#ROOT_FILES[@]} -ne 10 ]]; then
    log "QC_REFUSED completed_root_count=${#ROOT_FILES[@]} expected=10"
    exit 3
fi

for root_file in "${ROOT_FILES[@]}" "$REFERENCE_ROOT"; do
    if [[ ! -s "$root_file" ]]; then
        log "QC_REFUSED missing_root=$root_file"
        exit 3
    fi
done

source /mnt/Data/FADC500/root/bin/thisroot.sh
source /home/adminministrator/Other/data_engine/analysis-venv/bin/activate
python -m py_compile "$PROJECT_ROOT/scripts/check_waveform_quality.py"

log "QC_STARTED qc_id=$QC_ID root_count=${#ROOT_FILES[@]}"
set +e
python "$PROJECT_ROOT/scripts/check_waveform_quality.py" \
    --root-files "${ROOT_FILES[@]}" \
    --reference-root "$REFERENCE_ROOT" \
    --output-dir "$QC_OUTPUT_DIR" \
    --session-id "$QC_ID" \
    --expected-files 10 \
    --sample-events 1000 \
    --sample-mode first
QC_EXIT_CODE=$?
set -e

if [[ -s "$QC_OUTPUT_DIR/$QC_ID/session_qc_report.json" && \
      -s "$QC_OUTPUT_DIR/$QC_ID/session_qc_summary.csv" ]]; then
    log "QC_REPORTS_READY output_dir=$QC_OUTPUT_DIR/$QC_ID"
else
    log 'QC_REPORTS_MISSING'
    exit 4
fi

log "QC_FINISHED exit_code=$QC_EXIT_CODE"
exit "$QC_EXIT_CODE"
