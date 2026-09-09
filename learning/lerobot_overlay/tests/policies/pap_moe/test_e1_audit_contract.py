"""Probe diagnostics must not leak held-out targets or alter continuous actions."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import torch


SCRIPTS = Path('/home/ubuntu/ur3_ft300_ws/scripts')
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('e1_audit_contract', SCRIPTS / 'audit_e1_information_fusion.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_norm_match_changes_direction_not_magnitude():
    source = torch.tensor([[3., 4.]])
    donor = torch.tensor([[0., 2.]])
    result = audit.norm_match(donor, source)
    torch.testing.assert_close(result, torch.tensor([[0., 5.]]))
    torch.testing.assert_close(source, torch.tensor([[3., 4.]]))


def test_probe_never_uses_test_target_values():
    rng = np.random.default_rng(42)
    x, y = rng.normal(size=(12, 4)), rng.normal(size=(12, 2))
    train = np.arange(12) < 8
    first = audit.ridge_predict(x, y, train, ~train, .01)
    y[~train] += 10000
    second = audit.ridge_predict(x, y, train, ~train, .01)
    np.testing.assert_array_equal(first, second)


def test_probe_constant_features_are_finite():
    x, y = np.ones((12, 5)), np.arange(24).reshape(12, 2).astype(float)
    train = np.arange(12) < 8
    prediction = audit.ridge_predict(x, y, train, ~train, .01)
    np.testing.assert_allclose(prediction, np.tile(y[train].mean(0), (4, 1)))
