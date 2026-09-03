#!/usr/bin/env bash
set -euo pipefail

project_dir="/mnt/Data/ML_DetA"
run_id="$(date +%Y%m%d_%H%M%S)"
events="${EVENTS:-100000}"
timeout_seconds="${TIMEOUT_SECONDS:-300}"
threshold_adc="${THRESHOLD_ADC:-10}"
source_label="${SOURCE_LABEL:-Co60}"
source_tag="${source_label,,}"
log_path="${project_dir}/raw_data/${source_tag}_preamp_250msps_${run_id}.log"
expected_root="${project_dir}/raw_data/${source_tag}_preamp_250msps_${run_id}_thr${threshold_adc}_1.root"

case "${events}" in (*[!0-9]*|'') echo "EVENTS must be a positive integer" >&2; exit 2;; esac
case "${timeout_seconds}" in (*[!0-9]*|'') echo "TIMEOUT_SECONDS must be a positive integer" >&2; exit 2;; esac
case "${threshold_adc}" in (*[!0-9]*|'') echo "THRESHOLD_ADC must be a non-negative integer" >&2; exit 2;; esac
case "${source_label}" in (*[!A-Za-z0-9]*|'') echo "SOURCE_LABEL must contain only letters and digits" >&2; exit 2;; esac
if (( events > 100000 )); then
    echo "EVENTS must be <= 100000 per ROOT file" >&2
    exit 2
fi

export NKHOME="/mnt/Data/FADC500"
export ROOTSYS="${NKHOME}/root"
export PATH="${ROOTSYS}/bin:${NKHOME}/bin:${PATH:-}"
export LD_LIBRARY_PATH="${ROOTSYS}/lib:${NKHOME}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export ROOT_INCLUDE_PATH="${NKHOME}/include:${NKHOME}/DAQ/lib:${ROOT_INCLUDE_PATH:-}"

mkdir -p "${project_dir}/raw_data"
cd "${project_dir}"

{
    echo "=== ${source_label} direct-preamp 250-MSPS acquisition ==="
    echo "run_id=${run_id}"
    date --iso-8601=seconds
    echo "source=${source_label}"
    echo "source_tag=${source_tag}"
    echo "input=direct_preamp"
    echo "observed_polarity=negative"
    echo "threshold_adc=${threshold_adc}"
    echo "sample_rate_msps=250"
    echo "sample_period_ns=4"
    echo "stored_samples=4500"
    echo "nominal_window_us=18"
    echo "nominal_pretrigger_us=6"
    echo "nominal_posttrigger_us=12"
    echo "pwt=120"
    echo "dt_ns=500000"
    echo "events=${events}"
    echo "timeout_seconds=${timeout_seconds}"
    echo "output_root=${expected_root}"
    lsusb | grep '0547:1502'
    "${ROOTSYS}/bin/root" -l -b -q -e \
        "gROOT->ProcessLine(\".L scripts/run_co60_preamp_250msps.C\"); int rc=run_co60_preamp_250msps(${threshold_adc},${events},${timeout_seconds},\"${run_id}\",\"${source_tag}\"); gSystem->Exit(rc==0?0:rc);"
} 2>&1 | tee "${log_path}"
