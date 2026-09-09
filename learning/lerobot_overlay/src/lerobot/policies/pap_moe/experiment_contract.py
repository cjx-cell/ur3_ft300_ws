"""Opt-in, fail-closed Workspace50 contract for the corrected PAP experiment."""
import json
from pathlib import Path


def verify_contract(dataset_root, policy, preprocessor, postprocessor, output_dir):
    import numpy as np
    import pyarrow.parquet as pq

    root = Path(dataset_root)
    cfg = policy.config
    assert cfg.physical_fusion_architecture == 'action_input_tokens_v2'
    assert cfg.physics_gate_architecture == 'physics_gate_v2'
    assert cfg.action_step_routing and not cfg.temporal_gate_two_pass_inference
    assert policy.model.route_forecaster is None
    assert cfg.balance_loss_weight == 0 and cfg.expert_action_anchor_weight == 0
    assert not cfg.bounded_action_conditioning
    assert cfg.mask_invalid_prefix_tokens and cfg.mask_invalid_history_cameras
    info = json.loads((root / 'meta/info.json').read_text())
    assert info['total_frames'] == 14418 and info['total_episodes'] == 50
    stats = json.loads((root / 'meta/stats.json').read_text())
    features = ['action', 'observation.state', 'observation.state_history',
                'observation.force', 'observation.force_fast', 'observation.force_slow']
    arrays = pq.read_table(sorted((root / 'data').rglob('*.parquet')), columns=features).to_pydict()
    checks = {}
    for key in features:
        x = np.asarray(arrays[key], dtype=np.float64)
        x = x.reshape(-1, x.shape[-1])
        assert np.isfinite(x).all(), key
        metrics = ('q01', 'q99') if key in features[:3] else ('mean', 'std')
        for metric in metrics:
            value = (np.quantile(x, .01 if metric == 'q01' else .99, axis=0)
                     if metric.startswith('q') else getattr(x, metric)(axis=0))
            np.testing.assert_allclose(stats[key][metric], value, rtol=1e-5, atol=1e-6)
            for label, processor in [('pre', preprocessor), ('post', postprocessor)]:
                if label == 'post' and key != 'action':
                    continue
                found = False
                for step in processor.steps:
                    tensors = getattr(step, '_tensor_stats', {})
                    if key in tensors and metric in tensors[key]:
                        saved = tensors[key][metric].detach().cpu().numpy()
                        np.testing.assert_allclose(saved, value, rtol=1e-5, atol=1e-6)
                        checks[f'{label}/{key}/{metric}'] = float(np.max(np.abs(saved-value)))
                        found = True
                assert found, (label, key, metric)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    preprocessor.save_pretrained(output)
    postprocessor.save_pretrained(output)
    (output / 'verified_contract.json').write_text(json.dumps(checks, indent=2))
    print('CORRECTED_PAP_CONTRACT_PASS: actual processors match full-frame statistics; input fusion; direct Gate; no Forecaster', flush=True)
