"""Offline privacy and resource-bound tests for the T033-F readout."""

from __future__ import annotations

from contextlib import redirect_stdout
import copy
import importlib.util
import io
import json
from pathlib import Path
import sys
import unittest
from unittest import mock
import uuid


REPOSITORY = Path(__file__).resolve().parents[2]
SCRIPT = REPOSITORY / "scripts" / "read-barge-in-diagnostics.py"
SPEC = importlib.util.spec_from_file_location("t033f_readout", SCRIPT)
readout = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(readout)
SECRET = "PRIVATE-SENTINEL-phone-call-token"


def summary(sequence: int = 1) -> dict:
    totals = {
        "received_callbacks": 8,
        "received_bytes": 1280,
        "received_min_bytes": 160,
        "received_max_bytes": 160,
        "analyzed_bytes": 1280,
        "max_rms": 900,
        "max_consecutive": 8,
        "trigger_count": 1,
        "forwarded_callbacks": 1,
        "forwarded_bytes": 160,
        "dropped_callbacks": 7,
        "dropped_bytes": 1120,
        "speaking_seconds": 0.25,
    }
    window = {
        "sequence": sequence,
        "elapsed_seconds": 0.25,
        "outcome": "interrupted",
        "received_callbacks": 8,
        "received_bytes": 1280,
        "received_min_bytes": 160,
        "received_max_bytes": 160,
        "analyzed_callbacks": 8,
        "analyzed_bytes": 1280,
        "no_audio_count": 0,
        "grace_suppressed_count": 0,
        "decode_failed_count": 0,
        "below_threshold_count": 0,
        "above_threshold_count": 7,
        "gate_trigger_count": 1,
        "forwarded_callbacks": 1,
        "forwarded_bytes": 160,
        "dropped_callbacks": 7,
        "dropped_bytes": 1120,
        "max_rms": 900,
        "max_consecutive": 8,
        "dispatch_gap_count": 7,
        "dispatch_gap_min_seconds": 0.019,
        "dispatch_gap_mean_seconds": 0.02,
        "dispatch_gap_max_seconds": 0.021,
        "interrupt_succeeded_count": 1,
        "interrupt_not_owned_count": 0,
        "interrupt_failed_count": 0,
        "reset_succeeded_count": 1,
        "reset_not_owned_count": 0,
        "reset_failed_count": 0,
    }
    return {
        "schema": 1,
        "event": readout.SUMMARY_EVENT,
        "diagnostic_id": f"{sequence:032x}",
        "elapsed_seconds": 1.0,
        "windows_started": 1,
        "windows_retained": 1,
        "window_overflow_count": 0,
        "observer_failure_count": 0,
        "failure_category": None,
        "totals": totals,
        "windows": [window],
    }


class ReadoutContractTests(unittest.TestCase):
    def test_import_has_no_subprocess_or_log_read_side_effect(self):
        spec = importlib.util.spec_from_file_location(
            "t033f_readout_import_" + uuid.uuid4().hex, SCRIPT
        )
        module = importlib.util.module_from_spec(spec)
        with mock.patch("subprocess.Popen") as popen:
            assert spec.loader is not None
            spec.loader.exec_module(module)
        popen.assert_not_called()

    def test_prefixed_real_format_summary_is_rebuilt_without_neighbor_secrets(self):
        valid = json.dumps(summary()).encode()
        lines = [
            ("provider body " + SECRET).encode(),
            b"2026-09-17 | INFO | diagnostics - " + valid,
            (json.dumps({"unknown": SECRET}) + " trailing").encode(),
        ]
        result = readout.build_readout(lines)
        encoded = json.dumps(result)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["summaries"], [summary()])
        self.assertNotIn(SECRET, encoded)

    def test_unknown_fields_wrong_types_categories_and_nonfinite_are_rejected(self):
        cases = []
        unknown = summary()
        unknown["private"] = SECRET
        cases.append(unknown)
        wrong_type = summary()
        wrong_type["totals"]["received_bytes"] = True
        cases.append(wrong_type)
        category = summary()
        category["windows"][0]["outcome"] = SECRET
        cases.append(category)
        nonfinite = summary()
        nonfinite["elapsed_seconds"] = float("nan")
        cases.append(nonfinite)
        bad_id = summary()
        bad_id["diagnostic_id"] = SECRET
        cases.append(bad_id)
        malformed = b'{"event":"barge_in_diagnostics_summary"'
        lines = [json.dumps(item).encode() for item in cases] + [malformed]

        result = readout.build_readout(lines)
        self.assertEqual(result["summaries"], [])
        self.assertEqual(result["rejected_records"], len(cases) + 1)
        self.assertNotIn(SECRET, json.dumps(result))

    def test_excess_windows_and_oversized_records_are_rejected(self):
        excess = summary()
        excess["windows"] = [copy.deepcopy(excess["windows"][0]) for _ in range(17)]
        excess["windows_started"] = 17
        excess["windows_retained"] = 17
        oversized = b"x" * (readout.MAX_LINE_BYTES + 1)
        result = readout.build_readout([json.dumps(excess).encode(), oversized])
        self.assertEqual(result["summaries"], [])
        self.assertEqual(result["rejected_records"], 2)

    def test_output_is_capped_to_five_newest_summaries(self):
        lines = [json.dumps(summary(index)).encode() for index in range(1, 8)]
        result = readout.build_readout(lines)
        self.assertEqual(result["status"], "output_limited")
        self.assertEqual(result["omitted_summaries"], 2)
        self.assertEqual(
            [item["diagnostic_id"] for item in result["summaries"]],
            [f"{index:032x}" for index in range(3, 8)],
        )

    def test_scan_limit_has_fixed_status_and_no_raw_content(self):
        result = readout.build_readout(
            [("raw " + SECRET).encode()], scan_limited=True
        )
        self.assertEqual(result["status"], "scan_limited")
        self.assertNotIn(SECRET, json.dumps(result))

    def test_fixed_docker_command_has_no_caller_controlled_arguments(self):
        observed = []

        def run(command):
            observed.append(command)
            return [json.dumps(summary()).encode()], False

        with mock.patch.object(readout, "_run_bounded", side_effect=run):
            result = readout.read_recent_summaries()
        self.assertEqual(
            observed,
            [
                (
                    "docker",
                    "logs",
                    "--since",
                    "30m",
                    "--tail",
                    "2000",
                    "ai-voice-app",
                )
            ],
        )
        self.assertEqual(result["status"], "ok")

    def test_subprocess_error_prints_only_fixed_failure_document(self):
        output = io.StringIO()
        with mock.patch.object(
            readout,
            "read_recent_summaries",
            side_effect=readout.ReadoutFailure(SECRET),
        ), redirect_stdout(output):
            self.assertEqual(readout.main(), 1)
        self.assertNotIn(SECRET, output.getvalue())
        self.assertEqual(json.loads(output.getvalue())["status"], "read_failed")

    def test_real_subprocess_stderr_is_not_attached_to_failure(self):
        command = (
            sys.executable,
            "-c",
            "import sys; sys.stderr.write(" + repr(SECRET) + "); sys.exit(7)",
        )
        with self.assertRaises(readout.ReadoutFailure) as caught:
            readout._run_bounded(command)
        self.assertEqual(str(caught.exception), "log_command_failed")
        self.assertNotIn(SECRET, str(caught.exception))

    def test_bounded_runner_terminates_and_joins_on_timeout(self):
        command = (
            sys.executable,
            "-c",
            "import time; print('started',flush=True); time.sleep(10)",
        )
        with mock.patch.object(readout, "COMMAND_TIMEOUT_SECONDS", 0.15):
            with self.assertRaisesRegex(readout.ReadoutFailure, "timeout"):
                readout._run_bounded(command)

    def test_bounded_runner_stops_on_byte_limit_without_leaking_output(self):
        command = (sys.executable, "-c", "print('x'*10000)")
        with mock.patch.object(readout, "MAX_CAPTURE_BYTES", 512):
            lines, limited = readout._run_bounded(command)
        self.assertTrue(limited)
        self.assertEqual(lines, [])


if __name__ == "__main__":
    unittest.main()
