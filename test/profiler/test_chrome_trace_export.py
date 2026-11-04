# Owner(s): ["oncall: profiler"]
# ruff: noqa: F841

import gzip
import json
import os
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch


# Suppress libkineto USDT profiler_start/profiler_stop logs before initialization.
os.environ.setdefault("KINETO_LOG_LEVEL", "6")

import torch
import torch.nn as nn
from torch._C._profiler import _ExperimentalConfig
from torch._utils import try_import
from torch.profiler import (
    kineto_available,
    profile,
    ProfilerActivity,
    supported_activities,
)
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_profiler import initialize_kineto_with_accelerator
from torch.testing._internal.common_utils import (
    HardwareClassification,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    skipIfTorchDynamo,
    TemporaryFileName,
    TEST_WITH_ROCM,
    TestCase,
)


def get_profiler_activities(device_type):
    activities = [ProfilerActivity.CPU]
    if device_type not in ("cpu", "meta"):
        device_activity = getattr(ProfilerActivity, device_type.upper(), None)
        if device_activity and device_activity in supported_activities():
            activities.append(device_activity)
    return activities


def setUpModule():
    initialize_kineto_with_accelerator()


# Keep the optional tqdm monitor thread out of profiling results.
try:
    import tqdm

    tqdm.tqdm.monitor_interval = 0
except ImportError:
    pass


@unittest.skipIf(not kineto_available(), "Kineto is required")
class TestProfilerEventsParity(TestCase):
    """Tests validating parity between events() and export_chrome_trace() JSON."""

    hw_classification = HardwareClassification.GENERIC

    @skipIfTorchDynamo("profiler gets ignored if dynamo activated")
    def test_python_function_events_in_events(self):
        class DummyModule(nn.Module):
            def forward(self, x):
                return x + 1

        mod = DummyModule()
        with profile(
            activities=[ProfilerActivity.CPU],
            with_stack=True,
            experimental_config=_ExperimentalConfig(verbose=True),
        ) as prof:
            mod(torch.randn(4, 4))

        events = prof.events()
        python_events = [e for e in events if e.is_python_function]
        self.assertGreater(len(python_events), 0)
        for e in python_events:
            self.assertIsInstance(e.name, str)
            self.assertGreater(e.time_range.end - e.time_range.start, 0)

        with TemporaryFileName(mode="w+") as fname:
            prof.export_chrome_trace(fname)
            with open(fname) as f:
                trace = json.load(f)

            json_py = [
                e
                for e in trace["traceEvents"]
                if e.get("cat") == "python_function" and e.get("ph") == "X"
            ]
            self.assertEqual(len(python_events), len(json_py))

            # Verify python_id/parent_id/module_id parity with JSON args
            fe_mod = next((e for e in events if "DummyModule" in e.name), None)
            self.assertIsNotNone(fe_mod)
            self.assertGreater(fe_mod.python_id, 0)
            self.assertGreaterEqual(fe_mod.python_module_id, 0)

            json_mod = next(
                (e for e in json_py if "DummyModule" in e.get("name", "")),
                None,
            )
            self.assertIsNotNone(json_mod)
            args = json_mod["args"]
            self.assertEqual(fe_mod.python_id, args["Python id"])
            self.assertEqual(fe_mod.python_parent_id, args["Python parent id"])
            self.assertEqual(fe_mod.python_module_id, args["Python module id"])

    def test_key_averages_excludes_python_functions_by_default(self):
        """key_averages() must not include Python function events (e.g. threading.py: wait)
        by default; they can be opted in with include_python_functions=True."""
        t = threading.Thread(target=lambda: time.sleep(0.05))
        with profile(activities=[ProfilerActivity.CPU], with_stack=True) as prof:
            t.start()
            t.join()

        avgs = prof.key_averages()
        threading_entries = [e for e in avgs if "threading" in e.key]
        self.assertEqual(
            len(threading_entries),
            0,
            f"key_averages() should not include threading.py events by default, got: {[e.key for e in threading_entries]}",
        )

        avgs_with_py = prof.key_averages(include_python_functions=True)
        threading_entries_with_py = [e for e in avgs_with_py if "threading" in e.key]
        self.assertGreater(
            len(threading_entries_with_py),
            0,
            "key_averages(include_python_functions=True) should include threading.py events",
        )

    @skipIfTorchDynamo("profiler gets ignored if dynamo activated")
    def test_profiler_fwdbwd_flow_events_parity(self):
        """Verify that fwd->bwd flow fields on events() match Chrome trace JSON."""
        with profile(activities=[ProfilerActivity.CPU]) as prof:
            t1 = torch.ones(1, requires_grad=True)
            t2 = torch.ones(1, requires_grad=True)
            z = torch.add(t1, t2)
            y = torch.ones(1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
            loss.backward()

        fwdbwd_events = [
            e for e in prof.events() if e.flow_type == 1 and e.flow_id != 0
        ]
        self.assertGreater(
            len(fwdbwd_events), 0, "No fwdbwd flow events found via events()"
        )

        events_flow_starts = {e.flow_id for e in fwdbwd_events if e.flow_start}
        events_flow_ends = {e.flow_id for e in fwdbwd_events if not e.flow_start}

        with TemporaryFileName(mode="w+") as fname:
            prof.export_chrome_trace(fname)
            with open(fname) as f:
                j = json.load(f)

            json_flow_events = [
                e
                for e in j["traceEvents"]
                if e.get("ph") in ("s", "f") and e.get("cat") == "fwdbwd"
            ]
            json_flow_starts = {e["id"] for e in json_flow_events if e["ph"] == "s"}
            json_flow_ends = {e["id"] for e in json_flow_events if e["ph"] == "f"}

            self.assertEqual(
                json_flow_starts,
                events_flow_starts,
                "fwdbwd flow start IDs differ between events() and Chrome trace",
            )
            self.assertEqual(
                json_flow_ends,
                events_flow_ends,
                "fwdbwd flow end IDs differ between events() and Chrome trace",
            )

    @skipIfTorchDynamo("profiler gets ignored if dynamo activated")
    def test_profiler_op_args_events_parity(self):
        """Verify that cpu_op args on events() match Chrome trace JSON args."""
        base_tensor = torch.randn(1024, dtype=torch.float32)
        a = base_tensor.as_strided((16, 16), (17, 1), 0)
        b = base_tensor.as_strided((16, 16), (25, 2), 272)
        t1 = torch.ones((64, 32))
        t2 = torch.ones((64, 32))
        with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
            torch.add(a, b)
            torch.cat([t1, t2])

        fe_add = next((e for e in prof.events() if e.name == "aten::add"), None)
        self.assertIsNotNone(fe_add)
        fe_cat = next((e for e in prof.events() if e.name == "aten::cat"), None)
        self.assertIsNotNone(fe_cat)

        with TemporaryFileName(mode="w+") as fname:
            prof.export_chrome_trace(fname)
            with open(fname) as f:
                j = json.load(f)

            json_add = next(
                (
                    e
                    for e in j["traceEvents"]
                    if e.get("name") == "aten::add" and e.get("cat") == "cpu_op"
                ),
                None,
            )
            self.assertIsNotNone(json_add)
            args = json_add["args"]
            self.assertEqual(fe_add.structured_input_shapes, args["Input Dims"])
            self.assertEqual(fe_add.structured_input_strides, args["Input Strides"])
            self.assertEqual(fe_add.input_dtypes, args["Input type"])

            # Test a case with TensorList inputs -- structured_input_shapes
            # should handle TensorList nesting correctly.
            json_cat = next(
                (
                    e
                    for e in j["traceEvents"]
                    if e.get("name") == "aten::cat" and e.get("cat") == "cpu_op"
                ),
                None,
            )
            self.assertIsNotNone(json_cat)
            args_cat = json_cat["args"]
            self.assertEqual(fe_cat.structured_input_shapes, args_cat["Input Dims"])
            self.assertEqual(fe_cat.structured_input_strides, args_cat["Input Strides"])
            self.assertEqual(fe_cat.input_dtypes, args_cat["Input type"])


@unittest.skipIf(not kineto_available(), "Kineto is required")
class TestProfilerEventsParityDevice(TestCase):
    """Parity tests that need a real accelerator to produce device events."""

    hw_classification = HardwareClassification.ACCELERATOR

    def test_profiler_flow_events_parity(self, device):
        """Verify that async CPU->GPU flow fields on events() match Chrome trace JSON."""
        device_type = device.split(":")[0]
        with profile(activities=get_profiler_activities(device_type)) as prof:
            x = torch.randn(32, 32, device=device)
            torch.mm(x, x)

        # Collect async CPU->GPU flow info from events()
        events_with_flow = [
            e for e in prof.events() if e.flow_id is not None and e.flow_id != 0
        ]
        self.assertGreater(
            len(events_with_flow), 0, "No flow events found via events()"
        )

        for e in events_with_flow:
            self.assertIsInstance(e.flow_id, int)
            self.assertIsInstance(e.flow_type, int)
            self.assertIsInstance(e.flow_start, bool)

        # Verify parity with Chrome trace JSON for async CPU->GPU flow
        with TemporaryFileName(mode="w+") as fname:
            prof.export_chrome_trace(fname)
            with open(fname) as f:
                j = json.load(f)

            json_flow_events = [
                e
                for e in j["traceEvents"]
                if e.get("ph") in ("s", "f") and e.get("cat") == "ac2g"
            ]
            json_flow_starts = {e["id"] for e in json_flow_events if e["ph"] == "s"}
            json_flow_ends = {e["id"] for e in json_flow_events if e["ph"] == "f"}

            # kLinkAsyncCpuGpu = 2
            ac2g_events = [e for e in events_with_flow if e.flow_type == 2]
            events_flow_starts = {e.flow_id for e in ac2g_events if e.flow_start}
            events_flow_ends = {e.flow_id for e in ac2g_events if not e.flow_start}

            self.assertEqual(
                json_flow_starts,
                events_flow_starts,
                "Async CPU->GPU flow start IDs differ between events() and Chrome trace",
            )
            self.assertEqual(
                json_flow_ends,
                events_flow_ends,
                "Async CPU->GPU flow end IDs differ between events() and Chrome trace",
            )

    def test_profiler_timestamp_consistency(self, device):
        """Verify that FunctionEvent timestamps can reconstruct Chrome trace ts values."""
        device_type = device.split(":")[0]
        with profile(activities=get_profiler_activities(device_type)) as prof:
            x = torch.randn(32, 32, device=device)
            torch.mm(x, x)

        trace_start_ns = prof.profiler.kineto_results.trace_start_ns()

        with TemporaryFileName(mode="w+") as fname:
            prof.export_chrome_trace(fname)
            with open(fname) as f:
                j = json.load(f)

            # Chrome trace is relative to a different base time which is not exposed in Python.
            # It's probably not important to do so as we still have the relative differences
            # in duration.
            base_time_ns = j.get("baseTimeNanoseconds", 0)

            # Grab mm timestamp from events() and json
            fe_mm = next((e for e in prof.events() if e.name == "aten::mm"), None)
            json_mm = next(
                (
                    e
                    for e in j["traceEvents"]
                    if e.get("name") == "aten::mm" and e.get("ph") == "X"
                ),
                None,
            )

            # Reconstruct Chrome trace ts from events():
            # absolute_ns = mm_op_start_us * 1000 + trace_start_ns
            # chrome_ts = (absolute_ns - base_time_ns) / 1000 -> realign with json timeframe
            absolute_ns = int(fe_mm.time_range.start * 1000) + trace_start_ns
            recovered_ts = (absolute_ns - base_time_ns) / 1000
            self.assertEqual(
                recovered_ts,
                json_mm["ts"],
                msg="Recovered Chrome trace ts doesn't match JSON for aten::mm",
            )

    def test_profiler_external_id_parity(self, device):
        """Verify that FunctionEvent.external_id matches External id in Chrome trace JSON."""
        from collections import Counter

        device_type = device.split(":")[0]
        with profile(activities=get_profiler_activities(device_type)) as prof:
            with torch.profiler.record_function("test_region"):
                x = torch.randn(32, 32, device=device)
                y = torch.mm(x, x)
                z = y + x
                z.cpu()
                torch.accelerator.synchronize()

        with TemporaryFileName(mode="w+") as fname:
            prof.export_chrome_trace(fname)
            with open(fname) as f:
                j = json.load(f)

            json_name_ext = Counter(
                (e["name"], e["args"]["External id"])
                for e in j["traceEvents"]
                if e.get("args", {}).get("External id") is not None
            )
            events_name_ext = Counter(
                (ev.name, ev.external_id) for ev in prof.events() if ev.external_id != 0
            )

            self.assertEqual(
                events_name_ext,
                json_name_ext,
                "(name, external_id) pairs differ between events() and Chrome trace JSON",
            )

    def test_profiler_activity_type_parity(self, device):
        """Verify activity_type on events() matches Chrome trace cat field."""
        device_type = device.split(":")[0]
        with profile(activities=get_profiler_activities(device_type)) as prof:
            x = torch.randn(32, 32, device=device)
            torch.mm(x, x)

        events = prof.events()
        for e in events:
            self.assertIsInstance(e.activity_type, str)
            self.assertGreater(len(e.activity_type), 0)

        mm_event = next((e for e in events if e.name == "aten::mm"), None)
        self.assertIsNotNone(mm_event)
        self.assertEqual(mm_event.activity_type, "cpu_op")

        with TemporaryFileName(mode="w+") as fname:
            prof.export_chrome_trace(fname)
            with open(fname) as f:
                j = json.load(f)

            json_name_cats = {
                (e["name"], e["cat"])
                for e in j["traceEvents"]
                if e.get("ph") == "X" and "cat" in e
            }
            for e in events:
                self.assertIn(
                    (e.name, e.activity_type),
                    json_name_cats,
                    lambda msg: f"{msg}\nactivity_type mismatch for {e.name}",
                )


instantiate_device_type_tests(
    TestProfilerEventsParityDevice, globals(), except_for=("cpu",)
)


@unittest.skipIf(not kineto_available(), "Kineto is required")
class TestProfilerEventsParityCudaOnly(TestCase):
    """Structured-metadata parity tests keyed on CUPTI-specific trace
    categories/fields (cuda_runtime, cbid, cid, ...) that have no
    cross-backend equivalent."""

    hw_classification = HardwareClassification.CUDA

    def test_structured_metadata_matches_chrome_trace(self):
        # Compare metadata fields between events() and Chrome trace JSON to make sure they stay in parity
        # 1. Run a dummy workload with profiling enabled and collect the json/events() outputs
        # 2. Parse each event instance in the json and events() to create a key->value mapping
        #      - The key is a tuple of metadata fields that should be unique for each event
        #      - The value is a dict of metadata fields for that event
        # 3. Ensure that the keys and values match between the json and events() outputs

        from torch.autograd.profiler_util import _EVENT_METADATA_KEYS

        target_cats = ("cuda_runtime", "gpu_memcpy", "kernel")
        allowed_non_structured_trace_keys = {
            "External id",
            "correlation",
            "cbid",
            "cid",
            "device",
            "kind",
            "kernel",
            "ptr",
            "src",
            "dst",
        }
        supported_trace_keys = set(_EVENT_METADATA_KEYS).union(
            allowed_non_structured_trace_keys
        )

        def metadata_dict_from_trace_args(args):
            out = {}
            for kineto_key, (field_name, convert) in _EVENT_METADATA_KEYS.items():
                if kineto_key in args:
                    raw_value = args[kineto_key]
                    out[field_name] = (
                        convert(raw_value) if isinstance(raw_value, str) else raw_value
                    )
            return out

        def metadata_dict_from_function_event(fe):
            if fe.event_metadata is None:
                return {}

            out = {}
            for field_name, _ in _EVENT_METADATA_KEYS.values():
                val = getattr(fe.event_metadata, field_name)
                if val is not None:
                    out[field_name] = val
            return out

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            experimental_config=torch._C._profiler._ExperimentalConfig(
                expose_kineto_event_metadata=True
            ),
        ) as prof:
            x = torch.randn(10, 10, device="cuda")
            y = torch.mm(x, x)
            z = x + y
            z.cpu()

        # Build a mapping from key to events() FunctionEvent metadata
        event_records = {}
        for fe in prof.events():
            if fe.external_id == 0 or fe.id == 0 or fe.activity_type not in target_cats:
                continue
            # Using just one of these keys could result in collisions, so try to uniquely identify the event with all of them
            key = (fe.name, fe.activity_type, fe.external_id, fe.id)
            self.assertNotIn(
                key,
                event_records,
                lambda msg: f"{msg}\nDuplicate FunctionEvent record key encountered: {key}",
            )
            event_records[key] = metadata_dict_from_function_event(fe)

        with TemporaryFileName(mode="w+") as fname:
            prof.export_chrome_trace(fname)
            with open(fname) as f:
                trace = json.load(f)

        json_records = {}
        # Track unexpected (event_name, cat, key) combos, deduplicated
        unexpected_combos: set[tuple[str, str, str]] = set()

        # Loop through the trace events to perform a comparison
        for te in trace["traceEvents"]:
            cat = te.get("cat", "")
            args = te.get("args", {})
            ext_id = args.get("External id")
            correlation = args.get("correlation")

            if ext_id is None or correlation is None:
                continue
            if cat not in target_cats:
                continue

            # Any metadata keys that show up in JSON should show up in events()
            for k in set(args) - supported_trace_keys:
                unexpected_combos.add((te["name"][:100], cat, k))

            # Build the same key from JSON to try to match with a FunctionEvent
            key = (te["name"], te["cat"], ext_id, correlation)
            self.assertNotIn(
                key,
                json_records,
                lambda msg: f"{msg}\nDuplicate Chrome trace record key encountered: {key}",
            )
            json_records[key] = metadata_dict_from_trace_args(args)

        failure_msg = """\
====================================================================================
IMPORTANT: Are you making a Kineto change and seeing this message?

New metadata keys (see below) were found in the Chrome trace JSON that are not
yet exposed through the profiler's events() API (i.e. EventMetadata in
torch/autograd/profiler_util.py).

To fix this properly, you need to make sure the new Kineto data makes its way
to the events() property. The steps are:

1. Add the new key(s) to _EVENT_METADATA_KEYS in torch/autograd/profiler_util.py
   with the appropriate field name and type converter.
2. Add corresponding field(s) to the EventMetadata dataclass in the same file.
3. If the key should NOT be mapped (e.g. it duplicates an existing FunctionEvent
   attribute), add it to allowed_non_structured_trace_keys in this test instead.

For a model PR to follow, see: https://github.com/pytorch/pytorch/pull/180100
===================================================================================="""
        if unexpected_combos:
            summary = "\n".join(
                f"  {name} ({cat}): {key!r}"
                for name, cat, key in sorted(unexpected_combos)
            )
            raise AssertionError(f"\n{failure_msg}\n\nUnmapped keys:\n{summary}")

        self.assertGreater(len(json_records), 0, "No device-side records were compared")
        self.assertEqual(
            set(event_records),
            set(json_records),
            "Device event identities differ between events() and Chrome trace JSON",
        )

        for key in json_records:
            expected_meta = json_records[key]
            actual_meta = event_records[key]
            self.assertEqual(
                actual_meta,
                expected_meta,
                lambda msg: f"{msg}\n{key}: structured metadata differs between events() and Chrome trace JSON",
            )

    def test_typed_metadata_matches_chrome_trace(self):
        # Match events to Chrome trace records by identity (via name, cat, external/correlation id),
        # then verify every metadata field has the same value in the matching trace record.
        target_cats = ("cuda_runtime", "gpu_memcpy", "kernel")

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            experimental_config=torch._C._profiler._ExperimentalConfig(
                expose_kineto_event_metadata=True
            ),
        ) as prof:
            x = torch.randn(10, 10, device="cuda")
            y = torch.mm(x, x)
            z = x + y
            z.cpu()

        event_records = {}
        for event in prof.events():
            if (
                event.external_id == 0
                or event.id == 0
                or event.activity_type not in target_cats
            ):
                continue
            key = (event.name, event.activity_type, event.external_id, event.id)
            self.assertNotIn(key, event_records)
            event_records[key] = dict(event.metadata or {})

        with TemporaryFileName(mode="w+") as fname:
            prof.export_chrome_trace(fname)
            with open(fname) as f:
                trace = json.load(f)

        json_records = {}
        for trace_event in trace["traceEvents"]:
            category = trace_event.get("cat", "")
            args = trace_event.get("args", {})
            external_id = args.get("External id")
            correlation = args.get("correlation")
            if (
                external_id is None
                or correlation is None
                or category not in target_cats
            ):
                continue
            key = (trace_event["name"], category, external_id, correlation)
            self.assertNotIn(key, json_records)
            json_records[key] = args

        self.assertGreater(len(json_records), 0)
        self.assertEqual(set(event_records), set(json_records))

        for key, trace_metadata in json_records.items():
            typed_metadata = event_records[key]
            self.assertGreater(len(typed_metadata), 0)
            self.assertEqual(set(typed_metadata) - set(trace_metadata), set())
            for field, value in typed_metadata.items():
                self.assertEqual(value, trace_metadata[field])


instantiate_device_type_tests(
    TestProfilerEventsParityCudaOnly, globals(), only_for=("cuda",)
)


@unittest.skipIf(not kineto_available(), "Kineto is required")
@instantiate_parametrized_tests
class TestProfilerTraceCompression(TestCase):
    hw_classification = HardwareClassification.GENERIC

    @parametrize("use_python_export", [False, True])
    @parametrize(
        "suffix",
        [".json", ".json.gz", ".json.zst", "", ".unknown", ".zst.bak", ".gz.bak"],
    )
    def test_export(self, use_python_export, suffix):
        if suffix.endswith(".zst"):
            zstd = try_import("pyzstd") or try_import("zstandard")
            if zstd is None:
                self.skipTest("requires zstandard or pyzstd")
            opener = zstd.open
        elif suffix.endswith(".gz"):
            opener = gzip.open
        else:
            opener = open

        value = "café 中"
        escape_unicode = use_python_export and not suffix.endswith(".zst")
        metadata = json.dumps(value, ensure_ascii=escape_unicode)
        with profile(activities=[ProfilerActivity.CPU]) as prof:
            prof.add_metadata_json("compression_test", metadata)
            torch.ones(8) + 1
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "trace" + suffix)
            prof.export_chrome_trace(path, use_python_export=use_python_export)
            with opener(path, "rt", encoding="utf-8") as f:
                trace = json.load(f)
            self.assertEqual(trace["compression_test"], value)

    @parametrize("suffix", [".json.gz", ".json.zst"])
    def test_export_keeps_kineto_sibling(self, suffix):
        if suffix.endswith(".zst") and not (
            try_import("pyzstd") or try_import("zstandard")
        ):
            self.skipTest("requires zstandard or pyzstd")
        with profile(activities=[ProfilerActivity.CPU]) as prof:
            torch.ones(8) + 1
        kineto_export = prof.profiler.export_chrome_trace

        def export_with_sibling(path, *args, **kwargs):
            # Some Kineto builds write extra files next to the JSON.
            kineto_export(path, *args, **kwargs)
            with open(path.removesuffix(".json") + ".extra", "wb") as f:
                f.write(b"extra")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "trace" + suffix)
            with patch.object(
                prof.profiler, "export_chrome_trace", export_with_sibling
            ):
                prof.export_chrome_trace(path)
            sibling = os.path.join(tmpdir, "trace.extra")
            self.assertTrue(os.path.exists(path))
            self.assertTrue(os.path.exists(sibling))
            with open(sibling, "rb") as f:
                self.assertEqual(f.read(), b"extra")
            # The uncompressed JSON stays in the temporary directory.
            self.assertFalse(os.path.exists(os.path.join(tmpdir, "trace.json")))

    @parametrize("use_python_export", [False, True])
    def test_missing_zstd(self, use_python_export):
        with profile(activities=[ProfilerActivity.CPU]) as prof:
            torch.ones(8) + 1

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "trace.json.zst")
            with patch.dict(sys.modules, {"zstandard": None, "pyzstd": None}):
                message = "requires zstandard or pyzstd.*pip install zstandard.*pip install pyzstd"
                with self.assertRaisesRegex(ImportError, message):
                    prof.export_chrome_trace(path, use_python_export=use_python_export)
                self.assertFalse(os.path.exists(path))
                # A missing dependency must not consume the native trace.
                gzip_path = os.path.join(tmpdir, "trace.json.gz")
                prof.export_chrome_trace(gzip_path, use_python_export=use_python_export)
                with gzip.open(gzip_path, "rt") as f:
                    self.assertIn("traceEvents", json.load(f))


@unittest.skipIf(not kineto_available(), "Kineto is required")
class TestPythonChromeTraceExport(TestCase):
    """Verify that the Python streaming exporter produces traces equivalent
    to the C++ Kineto save() path."""

    hw_classification = HardwareClassification.ACCELERATOR

    def _profile_workload(self, device):
        x = torch.randn(64, 64, device=device)
        with profile(activities=get_profiler_activities(device.split(":")[0])) as prof:
            for _ in range(20):
                torch.mm(x, x)
                torch.add(x, x)
            torch.accelerator.synchronize()
        return prof

    def test_python_export_matches_kineto(self, device):
        prof = self._profile_workload(device)

        with tempfile.TemporaryDirectory() as tmpdir:
            kineto_path = os.path.join(tmpdir, "kineto.json")
            prof.profiler.kineto_results.save(kineto_path)

            py_path = os.path.join(tmpdir, "python.json")
            from torch.profiler._chrome_trace_export import (
                export_chrome_trace as _export,
            )

            _export(prof.profiler.kineto_results, py_path)

            with open(kineto_path) as f:
                kineto_trace = json.load(f)
            with open(py_path) as f:
                py_trace = json.load(f)

        kineto_events = kineto_trace["traceEvents"]
        py_events = py_trace["traceEvents"]

        def _activity_events(events):
            return [
                e
                for e in events
                if e.get("ph") == "X" and e.get("cat") not in ("Trace",)
            ]

        kineto_acts = _activity_events(kineto_events)
        py_acts = _activity_events(py_events)
        self.assertEqual(
            len(kineto_acts),
            len(py_acts),
            lambda msg: f"{msg}\nActivity event count mismatch: kineto={len(kineto_acts)}, python={len(py_acts)}",
        )

        from collections import Counter

        kineto_names = Counter((e["cat"], e["name"]) for e in kineto_acts)
        py_names = Counter((e["cat"], e["name"]) for e in py_acts)
        self.assertEqual(
            kineto_names, py_names, "Event (cat, name) distribution differs"
        )

        def _strip_ev_idx(event):
            """Return a copy with the internal 'Ev Idx' arg removed."""
            e = dict(event)
            args = dict(e.get("args", {}))
            args.pop("Ev Idx", None)
            e["args"] = args
            return e

        kineto_sorted = sorted(
            [_strip_ev_idx(e) for e in kineto_acts],
            key=lambda e: (e["cat"], e["name"], e["ts"]),
        )
        py_sorted = sorted(
            [_strip_ev_idx(e) for e in py_acts],
            key=lambda e: (e["cat"], e["name"], e["ts"]),
        )
        # Compare every key of every activity event.
        for ke, pe in zip(kineto_sorted, py_sorted):
            self.assertEqual(
                ke,
                pe,
                lambda msg: f"{msg}\nEvent mismatch for {ke['cat']}::{ke['name']} at ts={ke['ts']}",
            )

        # Flow event parity (ph, id, cat).
        kineto_flow = sorted(
            [
                (e["ph"], e["id"], e.get("cat", ""))
                for e in kineto_events
                if e.get("ph") in ("s", "f")
            ],
        )
        py_flow = sorted(
            [
                (e["ph"], e["id"], e.get("cat", ""))
                for e in py_events
                if e.get("ph") in ("s", "f")
            ],
        )
        self.assertEqual(kineto_flow, py_flow, "Flow events differ")

    def test_python_export_gzip(self, device):
        import gzip as gzip_mod

        prof = self._profile_workload(device)

        with tempfile.TemporaryDirectory() as tmpdir:
            gz_path = os.path.join(tmpdir, "trace.json.gz")
            prof.export_chrome_trace(gz_path, use_python_export=True)

            with gzip_mod.open(gz_path, "rt") as f:
                trace = json.load(f)

        self.assertIn("traceEvents", trace)
        x_events = [e for e in trace["traceEvents"] if e.get("ph") == "X"]
        self.assertGreater(len(x_events), 0)

    def test_python_export_plain_json(self, device):
        prof = self._profile_workload(device)

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = os.path.join(tmpdir, "trace.json")
            prof.export_chrome_trace(json_path, use_python_export=True)

            with open(json_path) as f:
                trace = json.load(f)

        self.assertIn("traceEvents", trace)
        x_events = [e for e in trace["traceEvents"] if e.get("ph") == "X"]
        self.assertGreater(len(x_events), 0)


instantiate_device_type_tests(
    TestPythonChromeTraceExport, globals(), except_for=("cpu",)
)


@unittest.skipIf(not kineto_available(), "Kineto is required")
@unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
class TestPythonChromeTraceExportCudaOnly(TestCase):
    """Python streaming exporter checks keyed on CUDA-specific data (CUDA runtime/
    driver versions, CUDA graph annotations) that have no cross-backend equivalent."""

    hw_classification = HardwareClassification.CUDA

    def _profile_workload(self):
        x = torch.randn(64, 64, device="cuda")
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(20):
                torch.mm(x, x)
                torch.add(x, x)
            torch.cuda.synchronize()
        return prof

    # The envelope is NVIDIA-shaped on both sides, and on ROCm the two sides disagree:
    # kineto's ROCm backend writes hip_driver_version/hip_runtime_version instead of the
    # cuda_* keys, and its device-property set names two fields differently
    # (maxSharedMemoryPerMultiProcessor, regsPerBlock) from the cudaDeviceProp shape the
    # exporter reproduces. The event-stream comparisons above hold on ROCm; only the
    # envelope has no defined ROCm equivalence yet.
    @unittest.skipIf(TEST_WITH_ROCM, "Chrome-trace envelope is not validated on ROCm")
    def test_python_export_envelope_matches_kineto(self):
        """Top-level trace keys, which the profiler result does not expose and the
        exporter therefore recomputes: device properties and the CUDA versions kineto
        writes from logGpuVersions()."""
        prof = self._profile_workload()

        with tempfile.TemporaryDirectory() as tmpdir:
            kineto_path = os.path.join(tmpdir, "kineto.json")
            prof.profiler.kineto_results.save(kineto_path)
            py_path = os.path.join(tmpdir, "python.json")
            prof.export_chrome_trace(py_path, use_python_export=True)

            with open(kineto_path) as f:
                kineto_trace = json.load(f)
            with open(py_path) as f:
                py_trace = json.load(f)

        self.assertEqual(
            py_trace["cuda_runtime_version"], kineto_trace["cuda_runtime_version"]
        )
        # The Python exporter omits the driver version when the optional cuda-bindings
        # package is unavailable or cannot query its CUDA runtime.
        if "cuda_driver_version" in py_trace:
            self.assertEqual(
                py_trace["cuda_driver_version"], kineto_trace["cuda_driver_version"]
            )
        # regsPerBlock is the one field torch's device properties do not carry, so the
        # exporter omits it; everything else must match what kineto wrote.
        for py_device, kineto_device in zip(
            py_trace["deviceProperties"], kineto_trace["deviceProperties"]
        ):
            self.assertEqual(
                py_device,
                {k: v for k, v in kineto_device.items() if k != "regsPerBlock"},
            )

    def test_cuda_graph_annotations_reach_the_trace(self):
        """End to end: annotations recorded at capture are baked into a replay's
        kernel events by export_chrome_trace(cuda_graph_annotations=...)."""
        from torch.cuda._graph_annotations import _reset_kernel_annotations
        from torch.cuda.graph_annotations import (
            get_kernel_annotations,
            is_available,
            mark_kernels,
        )

        if not is_available():
            self.skipTest("CUDA graph annotations are unavailable")

        _reset_kernel_annotations()
        self.addCleanup(_reset_kernel_annotations)
        x = torch.randn(64, 64, device="cuda")
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, enable_annotations=True):
            with mark_kernels({"name": "phase_a", "stream": 61}):
                _ = x + 1
        self.assertGreater(len(get_kernel_annotations()), 0)

        # The first profiled iterations can drop kernels while CUPTI warms up.
        for _ in range(3):
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
            ) as prof:
                graph.replay()
                torch.cuda.synchronize()
            with tempfile.TemporaryDirectory() as tmpdir:
                path = os.path.join(tmpdir, "trace.json")
                prof.export_chrome_trace(
                    path,
                    cuda_graph_annotations=get_kernel_annotations(),
                    graph_lanes="all",
                )
                with open(path) as f:
                    trace = json.load(f)
            annotated = [
                e
                for e in trace["traceEvents"]
                if e.get("args", {}).get("name") == "phase_a"
            ]
            if annotated:
                break

        self.assertTrue(trace.get("cudaGraphInlineAnnotated"))
        self.assertGreater(len(annotated), 0)
        for event in annotated:
            self.assertEqual(event["tid"], 61)
            self.assertEqual(event["args"]["stream"], 61)
            self.assertIn("original_stream", event["args"])


class _StubActivity:
    """Minimal stand-in for ITraceActivity: the accessors the exporter calls."""

    def __init__(self, cat, name, rid, ts=100, metadata_json="", did=0):
        self._cat, self._name, self._rid = cat, name, rid
        self._ts, self._metadata_json, self._did = ts, metadata_json, did

    def type(self):
        return self._cat

    def name(self):
        return self._name

    def device_id(self):
        return self._did

    def resource_id(self):
        return self._rid

    def timestamp(self):
        return self._ts

    def duration(self):
        return 10

    def correlation_id(self):
        return 0

    def linked_correlation_id(self):
        return 0

    def linked_activity(self):
        return None

    def metadata_json(self):
        return self._metadata_json

    def flow_id(self):
        return 0

    def flow_start(self):
        return False

    def flow_type(self):
        return 2


class _StubResults:
    def __init__(self, activities):
        self._activities = activities

    def trace_activities(self):
        return self._activities


class TestChromeTraceInlineAnnotations(TestCase):
    """Inline CUDA-graph annotations, driven through stub activities so the branches
    are covered without a capture or a live profiler."""

    hw_classification = HardwareClassification.GENERIC

    def _export(self, activities, **kwargs):
        from torch.profiler._chrome_trace_export import export_chrome_trace

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "trace.json")
            export_chrome_trace(_StubResults(activities), path, **kwargs)
            with open(path) as f:
                return json.load(f)

    @staticmethod
    def _metadata(graph_node_id):
        return f'"stream": 3, "graph node id": {graph_node_id}'

    @staticmethod
    def _kernels(trace):
        return [e for e in trace["traceEvents"] if e.get("cat") == "kernel"]

    def test_annotation_fields_and_lane(self):
        trace = self._export(
            [_StubActivity("kernel", "k", rid=3, metadata_json=self._metadata(42))],
            annotations={42: [{"name": "phase_a", "stream": 61}]},
            graph_lanes="all",
        )
        self.assertTrue(trace["cudaGraphInlineAnnotated"])
        [kernel] = self._kernels(trace)
        self.assertEqual(kernel["args"]["name"], "phase_a")
        # tid, args.stream and the annotation agree on the lane; the hardware stream
        # the work actually ran on is kept as original_stream.
        self.assertEqual(kernel["tid"], 61)
        self.assertEqual(kernel["args"]["stream"], 61)
        self.assertEqual(kernel["args"]["original_stream"], 3)
        # The lane gets a name of its own.
        names = [
            e
            for e in trace["traceEvents"]
            if e.get("name") == "thread_name" and e["tid"] == 61
        ]
        self.assertEqual(len(names), 1)

    def test_lane_named_by_annotation(self):
        trace = self._export(
            [_StubActivity("kernel", "k", rid=3, metadata_json=self._metadata(42))],
            annotations={42: [{"Process Group Description": "DP", "stream": 61}]},
            graph_lanes="all",
        )
        [name_event] = [
            e
            for e in trace["traceEvents"]
            if e.get("name") == "thread_name" and e["tid"] == 61
        ]
        self.assertEqual(name_event["args"]["name"], "DP")

    def test_unannotated_graph_work_moves_to_default_stream(self):
        trace = self._export(
            [_StubActivity("kernel", "k", rid=3, metadata_json=self._metadata(42))],
            annotations={99: [{"name": "other"}]},
            graph_lanes="all",
        )
        # Nothing matched, so no inline flag -- but graphed work still collapses onto
        # the default lane instead of scattering over the replay's hardware streams.
        self.assertNotIn("cudaGraphInlineAnnotated", trace)
        [kernel] = self._kernels(trace)
        self.assertEqual(kernel["tid"], 7)
        self.assertEqual(kernel["args"]["stream"], 7)
        self.assertEqual(kernel["args"]["original_stream"], 3)

    def test_non_graph_work_untouched(self):
        trace = self._export(
            [_StubActivity("kernel", "k", rid=3, metadata_json='"stream": 3')],
            annotations={42: [{"name": "phase_a"}]},
        )
        [kernel] = self._kernels(trace)
        self.assertEqual(kernel["tid"], 3)
        self.assertNotIn("original_stream", kernel["args"])

    def test_orphaned_user_annotation_slices_dropped(self):
        # CUPTI replicates gpu_user_annotation onto every stream a graph replays on.
        # A slice keeps its lane only if work stayed there: lane 3's kernel moved to 61.
        trace = self._export(
            [
                _StubActivity("kernel", "k", rid=3, metadata_json=self._metadata(42)),
                _StubActivity("gpu_user_annotation", "ann_moved", rid=3),
                _StubActivity("kernel", "eager", rid=9),
                _StubActivity("gpu_user_annotation", "ann_kept", rid=9),
            ],
            annotations={42: [{"name": "phase_a", "stream": 61}]},
            graph_lanes="all",
        )
        kept = [
            e["name"]
            for e in trace["traceEvents"]
            if e.get("cat") == "gpu_user_annotation"
        ]
        self.assertEqual(kept, ["ann_kept"])

    def test_args_carry_one_stream_key(self):
        # The metadata fragment kineto supplies already has a "stream"; the lane must
        # replace it rather than append a second (duplicate keys are parser-dependent).
        from torch.profiler._chrome_trace_export import export_chrome_trace

        def reject_duplicates(pairs):
            keys = [k for k, _ in pairs]
            self.assertEqual(len(keys), len(set(keys)), f"duplicate keys in {keys}")
            return dict(pairs)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "trace.json")
            export_chrome_trace(
                _StubResults(
                    [
                        _StubActivity(
                            "kernel", "k", rid=3, metadata_json=self._metadata(42)
                        )
                    ]
                ),
                path,
                annotations={42: [{"name": "phase_a", "stream": 61}]},
                graph_lanes="all",
            )
            with open(path) as f:
                trace = json.load(f, object_pairs_hook=reject_duplicates)

        [kernel] = self._kernels(trace)
        self.assertEqual(kernel["args"]["stream"], 61)
        self.assertEqual(kernel["args"]["original_stream"], 3)

    def test_empty_annotations_leave_events_alone(self):
        # Nothing to inject, so no lane pass: an empty mapping reads as None rather than
        # collapsing every graphed event onto one lane for no benefit.
        trace = self._export(
            [_StubActivity("kernel", "k", rid=3, metadata_json=self._metadata(42))],
            annotations={},
        )
        [kernel] = self._kernels(trace)
        self.assertEqual(kernel["tid"], 3)
        self.assertNotIn("original_stream", kernel["args"])

    def test_annotations_alone_move_nothing(self):
        # The default: annotate the events, leave the trace's stream layout alone, even
        # for an annotation that names a lane.
        trace = self._export(
            [_StubActivity("kernel", "k", rid=3, metadata_json=self._metadata(42))],
            annotations={42: [{"name": "phase_a", "stream": 61}]},
        )
        [kernel] = self._kernels(trace)
        self.assertEqual(kernel["args"]["name"], "phase_a")
        self.assertEqual(kernel["tid"], 3)
        self.assertEqual(kernel["args"]["stream"], 3)
        self.assertNotIn("original_stream", kernel["args"])

    def test_invalid_graph_lanes_rejected(self):
        with self.assertRaisesRegex(ValueError, "graph_lanes"):
            self._export([], annotations={1: [{"name": "x"}]}, graph_lanes="lanes")

    def test_graph_lanes_all_without_annotations_rejected(self):
        # Standalone it would only collapse every graphed event onto default_stream.
        with self.assertRaisesRegex(ValueError, "needs cuda_graph_annotations"):
            self._export([], graph_lanes="all")

    def test_recorded_stream_reported_when_lanes_off(self):
        # graph_lanes="none" must not act on the recorded lane, but must not lose it
        # either: args.stream still names the stream the event actually ran on.
        trace = self._export(
            [_StubActivity("kernel", "k", rid=3, metadata_json=self._metadata(42))],
            annotations={42: [{"name": "phase_a", "stream": 61}]},
        )
        [kernel] = self._kernels(trace)
        self.assertEqual(kernel["tid"], 3)
        self.assertEqual(kernel["args"]["stream"], 3)
        self.assertEqual(kernel["args"]["annotated_stream"], 61)
        self.assertNotIn("original_stream", kernel["args"])

    def test_without_annotations_nothing_changes(self):
        activities = [
            _StubActivity("kernel", "k", rid=3, metadata_json=self._metadata(42)),
            _StubActivity("gpu_user_annotation", "ann", rid=3),
        ]
        trace = self._export(activities)
        self.assertNotIn("cudaGraphInlineAnnotated", trace)
        [kernel] = self._kernels(trace)
        self.assertEqual(kernel["tid"], 3)
        self.assertNotIn("original_stream", kernel["args"])
        self.assertEqual(
            len(
                [
                    e
                    for e in trace["traceEvents"]
                    if e.get("cat") == "gpu_user_annotation"
                ]
            ),
            1,
        )


if __name__ == "__main__":
    run_tests()
