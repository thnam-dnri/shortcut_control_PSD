from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "audit_experiment_protocol.py"
SPEC = importlib.util.spec_from_file_location("audit_experiment_protocol", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def make_record(
    *,
    qc_status: str = "PASS",
    processing_status: str = "OK",
    complete_input: bool = True,
) -> dict[str, object]:
    return {
        "hdf5": f"processed_data/{qc_status.lower()}.h5",
        "source": "ba133",
        "partition": "train",
        "qc_status": qc_status,
        "processing_status": processing_status,
        "complete_input": complete_input,
    }


def test_acquisition_block_is_deterministic() -> None:
    result = MODULE.parse_acquisition_block(
        "co60_preamp_250msps_20260814_172138_thr10_1.root", "co60"
    )
    assert result == {
        "acquisition_date": "20260814",
        "acquisition_time": "172138",
        "acquisition_block_id": "co60_20260814_172138",
    }


def test_qc_policy_admission_excludes_fail_and_unknown() -> None:
    assert MODULE.policy_admits(make_record(qc_status="PASS"), "pass_warn")
    assert MODULE.policy_admits(make_record(qc_status="WARN"), "pass_warn")
    assert not MODULE.policy_admits(make_record(qc_status="FAIL"), "pass_warn")
    assert not MODULE.policy_admits(make_record(qc_status="UNKNOWN"), "pass_warn")
    assert not MODULE.policy_admits(make_record(complete_input=False), "pass_warn")
    assert not MODULE.policy_admits(make_record(processing_status="INPUT_INVALID"), "pass_warn")


def test_policy_summary_reports_exclusion_reasons() -> None:
    records = [
        make_record(qc_status="PASS"),
        make_record(qc_status="WARN"),
        make_record(qc_status="FAIL"),
        make_record(qc_status="UNKNOWN"),
        make_record(complete_input=False),
    ]
    summary = MODULE.summarize_policy(records, "pass_warn")
    assert summary["admitted_file_count"] == 2
    assert summary["excluded_file_count"] == 3
    assert summary["excluded_reasons"] == {
        "incomplete_input": 1,
        "qc_status=FAIL": 1,
        "qc_status=UNKNOWN": 1,
    }


def test_percentile_interpolates_sorted_values() -> None:
    assert MODULE.percentile([1.0, 2.0, 3.0, 4.0], 0.5) == pytest.approx(2.5)
    assert MODULE.percentile([1.0, 2.0, 3.0, 4.0], 0.95) == pytest.approx(3.85)


def test_extract_qc_metrics_preserves_limits_and_rail_fraction() -> None:
    metrics = MODULE.extract_qc_metrics(
        {
            "reference_root": "/reference.root",
            "thresholds": {
                "noise_warn_factor": 1.25,
                "noise_fail_factor": 1.5,
            },
            "reference_metrics": {
                "baseline": {"baseline_noise_rms_adc": {"p95": 1.0, "p99": 1.2}}
            },
        },
        {
            "integrity": {"sample_mode": "distributed", "entries": 100000},
            "baseline": {
                "noise_numpy_slice": "0:1000",
                "noise_samples_inclusive": [1, 1000],
                "baseline_noise_rms_adc": {"p95": 1.4, "p99": 1.8},
            },
            "waveform_integrity": {
                "lower_rail_event_fraction": 0.001,
                "upper_rail_event_fraction": 0.003,
            },
            "timing": {"timestamp_monotonic": True},
        },
    )
    assert metrics["noise_p95_warn_limit_adc"] == pytest.approx(1.25)
    assert metrics["noise_p99_fail_limit_adc"] == pytest.approx(1.8)
    assert metrics["max_rail_event_fraction"] == pytest.approx(0.003)
    assert metrics["file_sample_mode"] == "distributed"
