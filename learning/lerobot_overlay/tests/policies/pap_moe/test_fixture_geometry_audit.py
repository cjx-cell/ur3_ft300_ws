import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path('/home/ubuntu/ur3_ft300_ws')
PATH = ROOT / 'src/ur_simulation_gz/ur_simulation_gz/scripts/peg_in_hole/fixture_geometry_audit.py'
spec = importlib.util.spec_from_file_location('geometry_audit', PATH)
geo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(geo)
IDENTITY = [0., 0., 0., 1.]


@pytest.mark.parametrize('position,quaternion,expected', [
    ([0, 0, .110], IDENTITY, True),
    ([0, 0, .113], IDENTITY, True),
    ([0, 0, .13], IDENTITY, False),
    ([0, 0, -.1], IDENTITY, False),
    ([0, 0, .10], IDENTITY, False),
    ([.008, 0, .110], IDENTITY, False),
    ([0, 0, .110], [0, 1, 0, 0], False),
    ([0, 0, .110], [0, np.sqrt(.5), 0, np.sqrt(.5)], False),
])
def test_geometric_positive_and_negative_cases(position, quaternion, expected):
    result = geo.score_seating((position, quaternion), ([0, 0, 0], IDENTITY))
    assert result['candidate_inserted'] is expected
    assert result['release_stability_verified'] is False


def test_score_invariant_under_shared_world_rigid_transform():
    q = [0, np.sin(.3), 0, np.cos(.3)]
    r = geo.rotation(q)
    translation = np.array([.3, -.1, .77])
    result = geo.score_seating((translation + r @ [0, 0, .110], q), (translation, q))
    assert result['candidate_inserted']
    assert result['candidate_fully_seated']
    assert result['insertion_depth_m'] == pytest.approx(.080)


def test_pose_parser_handles_protobuf_omitted_zero_and_exact_names():
    text = '''pose {name: "peg::link" position {z: 9} orientation {w: 1}}
    pose {name: "peg" position {z: 0.11} orientation {w: 1}}
    pose {name: "hole_plate" position {} orientation {w: 1}}'''
    poses = geo.parse_model_poses(text)
    assert set(poses) == {'peg', 'hole_plate'}
    assert geo.score_seating(poses['peg'], poses['hole_plate'])['candidate_inserted']


def test_insertion_is_not_full_seating():
    result = geo.score_seating(([0, 0, .113], IDENTITY), ([0, 0, 0], IDENTITY))
    assert result['candidate_inserted']
    assert not result['candidate_fully_seated']
    assert result['nominal_seating_gap_m'] == pytest.approx(.003)


@pytest.mark.parametrize('text', [
    'pose {name: "peg" position {} orientation {}}',
    'pose {name: "peg" position {x: nan} orientation {w: 1}}',
    'pose {name: "peg" position {} }',
    'pose {name: "peg" position {} orientation {w: 1}}' * 2,
])
def test_pose_parser_fails_closed(text):
    with pytest.raises(ValueError):
        geo.parse_model_poses(text)


def test_installed_cad_meshes_match_generator_zero_clearance():
    import struct
    generator = ROOT / 'pap_moe_framework/scripts/generate_real_fixture_meshes.py'
    spec = importlib.util.spec_from_file_location('fixture_generator', generator)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    models = ROOT / 'src/ur_simulation_gz/ur_simulation_gz/models'
    for relative, expected in [
        ('pap_moe_real_peg/meshes/peg.stl', module.make_peg()),
        ('pap_moe_real_hole/meshes/hole.stl', module.make_socket(0.)),
        ('pap_moe_real_peg/meshes/peg_body_collision.stl', module.make_peg_body_collision()),
        ('pap_moe_real_hole/meshes/hole_side_collision.stl', module.make_socket_side_collision(0.)),
    ]:
        data = (models / relative).read_bytes()
        count, = struct.unpack_from('<I', data, 80)
        faces = [struct.unpack_from('<12fH', data, 84 + 50*i)[3:12] for i in range(count)]
        np.testing.assert_allclose(np.array(faces).reshape(-1, 3, 3), expected, atol=1e-8)
