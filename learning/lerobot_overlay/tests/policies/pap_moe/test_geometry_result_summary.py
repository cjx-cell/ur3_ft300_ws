import importlib.util
import json
from pathlib import Path

PATH = Path('/home/ubuntu/ur3_ft300_ws/scripts/write_gazebo_eval_result.py')
spec = importlib.util.spec_from_file_location('geometry_result_writer', PATH)
writer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(writer)


def test_missing_geometry_not_scored_as_failure(tmp_path):
    assert writer.geometry_summary(tmp_path/'missing') == {'available': False}


def test_geometry_summary_separates_inserted_seated_and_invalid(tmp_path):
    path = tmp_path/'geometry.jsonl'
    records = [dict(valid=False, reason='not spawned'), dict(valid=True, geometry=dict(
        schema='cad_seating_shadow_v2', candidate_inserted=True, candidate_fully_seated=False))]
    path.write_text('\n'.join(json.dumps(r) for r in records)+'\n{"partial":')
    result = writer.geometry_summary(path)
    assert result['valid_samples'] == 1
    assert result['invalid_or_other_schema_samples'] == 1
    assert result['malformed_samples'] == 1
    assert result['observed_inserted']
    assert not result['observed_fully_seated']
    assert not result['release_stability_verified']
