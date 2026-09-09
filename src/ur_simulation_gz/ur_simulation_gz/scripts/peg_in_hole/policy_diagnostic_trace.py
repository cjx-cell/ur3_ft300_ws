"""Opt-in diagnostic recording; never modifies observations or commands."""
import json
import os
import time
from pathlib import Path

import numpy as np


def enabled():
    return bool(os.environ.get("POLICY_DIAGNOSTIC_TRACE_DIR"))


def event(kind, **values):
    if not enabled():
        return
    try:
        root = Path(os.environ["POLICY_DIAGNOSTIC_TRACE_DIR"])
        root.mkdir(parents=True, exist_ok=True)
        record = dict(kind=kind, wall_time=time.time(), **values)
        with (root / "controller.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
    except Exception as exc:
        print(f"DIAGNOSTIC trace failed: {exc}", flush=True)


def snapshot(chunk_id, **values):
    if not enabled():
        return
    try:
        arrays = {}

        def flatten(prefix, value):
            if value is None:
                return
            if isinstance(value, dict):
                for key, item in value.items():
                    flatten(prefix + "/" + str(key), item)
            elif isinstance(value, (list, tuple)):
                for index, item in enumerate(value):
                    flatten(prefix + "/" + str(index), item)
            else:
                if hasattr(value, "detach"):
                    value = value.detach().cpu()
                    if str(value.dtype) == "torch.bfloat16":
                        value = value.float()
                    value = value.numpy()
                array = np.asarray(value)
                if array.dtype != object:
                    arrays[prefix] = array

        for key, value in values.items():
            flatten(key, value)
        root = Path(os.environ["POLICY_DIAGNOSTIC_TRACE_DIR"])
        root.mkdir(parents=True, exist_ok=True)
        # Refuse overwrite: a duplicate chunk is itself a diagnostic error.
        with (root / f"chunk_{chunk_id:05d}.npz").open("xb") as stream:
            np.savez(stream, **arrays)
    except Exception as exc:
        print(f"DIAGNOSTIC snapshot failed: {exc}", flush=True)
