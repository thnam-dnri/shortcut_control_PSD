#!/usr/bin/env bash
set -euo pipefail

# Collect one million events as ten complete 100,000-event ROOT files.
# The per-file wrapper keeps each ROOT file bounded and records its own log.

project_dir="/mnt/Data/ML_DetA"
source_label="${SOURCE_LABEL:-Cs137}"
source_tag="${source_label,,}"
session_id="${SESSION_ID:-$(date +%Y%m%d_%H%M%S)}"
events_per_file="${EVENTS_PER_FILE:-100000}"
files_per_session="${FILES_PER_SESSION:-10}"
timeout_seconds="${TIMEOUT_SECONDS:-2400}"
threshold_adc="${THRESHOLD_ADC:-10}"
session_log="${project_dir}/raw_data/${source_tag}_preamp_250msps_session_${session_id}.log"

case "${source_label}" in (*[!A-Za-z0-9]*|'') echo "SOURCE_LABEL must contain only letters and digits" >&2; exit 2;; esac
case "${session_id}" in (*[!A-Za-z0-9_-]*|'') echo "SESSION_ID contains unsupported characters" >&2; exit 2;; esac
case "${events_per_file}" in (*[!0-9]*|'') echo "EVENTS_PER_FILE must be a positive integer" >&2; exit 2;; esac
case "${files_per_session}" in (*[!0-9]*|'') echo "FILES_PER_SESSION must be a positive integer" >&2; exit 2;; esac
case "${timeout_seconds}" in (*[!0-9]*|'') echo "TIMEOUT_SECONDS must be a positive integer" >&2; exit 2;; esac
case "${threshold_adc}" in (*[!0-9]*|'') echo "THRESHOLD_ADC must be a non-negative integer" >&2; exit 2;; esac
if (( events_per_file <= 0 || events_per_file > 100000 )); then
    echo "EVENTS_PER_FILE must be in 1..100000" >&2
    exit 2
fi
if (( files_per_session <= 0 )); then
    echo "FILES_PER_SESSION must be positive" >&2
    exit 2
fi
if (( events_per_file * files_per_session != 1000000 )); then
    echo "EVENTS_PER_FILE * FILES_PER_SESSION must equal 1000000" >&2
    exit 2
fi

mkdir -p "${project_dir}/raw_data"
cd "${project_dir}"

{
    echo "=== ${source_label} 1,000,000-event DAQ session ==="
    echo "session_id=${session_id}"
    date --iso-8601=seconds
    echo "source=${source_label}"
    echo "source_tag=${source_tag}"
    echo "files_per_session=${files_per_session}"
    echo "events_per_file=${events_per_file}"
    echo "total_events=$((events_per_file * files_per_session))"
    echo "timeout_seconds_per_file=${timeout_seconds}"
    echo "threshold_adc=${threshold_adc}"
    echo "output_directory=${project_dir}/raw_data"
    echo

    for ((file_index = 1; file_index <= files_per_session; file_index++)); do
        echo "--- Starting file ${file_index}/${files_per_session} ---"
        EVENTS="${events_per_file}" \
        TIMEOUT_SECONDS="${timeout_seconds}" \
        THRESHOLD_ADC="${threshold_adc}" \
        SOURCE_LABEL="${source_label}" \
            "${project_dir}/scripts/run_co60_preamp_250msps.sh"
        echo "--- Completed file ${file_index}/${files_per_session} ---"
        echo
    done

    echo "DAQ_SESSION_FINISHED files=${files_per_session} total_events=$((events_per_file * files_per_session))"
} 2>&1 | tee "${session_log}"
