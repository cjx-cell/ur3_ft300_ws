"""Shadow-only CAD seating check. Never supplies policy observations/actions.

Geometry: generate_real_fixture_meshes.py, peg tip z=-.090, taper top
z=-.010; socket floor z=.020 and mouth z=.100. Tolerances below are
diagnostic candidates, not a declaration of precision or release stability.
"""
import re

import numpy as np


def _blocks(text, field):
    for match in re.finditer(r'\b' + re.escape(field) + r'\s*\{', text):
        start, depth = match.end(), 1
        for end in range(start, len(text)):
            depth += (text[end] == '{') - (text[end] == '}')
            if depth == 0:
                yield text[start:end]
                break


def parse_model_poses(text, names=('peg', 'hole_plate')):
    """Extract exact model names from one Gazebo Pose_V text message.

    Protobuf omits zero scalar coordinates; omitted scalars mean zero.
    Missing messages or an invalid quaternion are never identity fallback.
    """
    result = {}
    for block in _blocks(text, 'pose'):
        match = re.search(r'\bname:\s*"([^"\n]+)"', block)
        if match is None or match[1] not in names:
            continue
        name = match[1]
        if name in result:
            raise ValueError('Duplicate model pose: ' + name)
        vectors = []
        for field, axes in [('position', 'xyz'), ('orientation', 'xyzw')]:
            sections = list(_blocks(block, field))
            if len(sections) != 1:
                raise ValueError('Missing or ambiguous ' + field)
            values = []
            for axis in axes:
                matches = re.findall(r'\b' + axis + r':\s*([^\s}]+)', sections[0])
                if len(matches) > 1:
                    raise ValueError('Duplicate coordinate')
                values.append(float(matches[0]) if matches else 0.)
            vectors.append(np.array(values))
        rotation(vectors[1])
        if not np.isfinite(vectors[0]).all():
            raise ValueError('Nonfinite position')
        result[name] = vectors
    return result


def rotation(quaternion):
    q = np.asarray(quaternion, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all():
        raise ValueError('Invalid quaternion')
    norm = np.linalg.norm(q)
    if abs(norm - 1.) > 1e-3:
        raise ValueError('Quaternion must be unit length')
    x, y, z, w = q / norm
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])


def score_seating(peg, hole):
    """Evaluate tip/floor and taper-envelope consistency in socket frame.

    128 angular samples mirror CAD tessellation. This is not a full physics
    collision solver; report metrics even when the candidate check rejects.
    """
    pp, pq = peg
    hp, hq = hole
    pp, hp = np.asarray(pp, dtype=float), np.asarray(hp, dtype=float)
    if pp.shape != (3,) or hp.shape != (3,) or not np.isfinite([pp, hp]).all():
        raise ValueError('Invalid position')
    rh = rotation(hq)
    relative = rh.T @ rotation(pq)
    origin = rh.T @ (pp - hp)
    axis_angle = float(np.degrees(np.arccos(np.clip(relative[2, 2], -1., 1.))))
    theta = np.arange(128) * (2*np.pi/128)
    rings = []
    for z in np.linspace(-.090, -.010, 17):
        radius = .010 + (z + .090) * (.010/.080)
        local = np.column_stack((radius*np.cos(theta), radius*np.sin(theta), np.full(128, z)))
        rings.append(local @ relative.T + origin)
    points = np.concatenate(rings)
    tip = origin + relative @ np.array([0., 0., -.090])
    depth = float(.100 - tip[2])
    floor_penetration = float(max(0., .020 - points[:, 2].min()))
    inside = points[:, 2] <= .100
    wall_penetration = 0.
    if inside.any():
        section = points[inside]
        radius = .010 + np.clip(section[:, 2] - .020, 0., .080) * (.010/.080)
        wall_penetration = float(max(0., (np.linalg.norm(section[:, :2], axis=1) - radius).max()))
    checks = dict(depth=.075 <= depth <= .081, axis=axis_angle <= 3.,
                  floor=floor_penetration <= .001, wall=wall_penetration <= .001)
    return dict(schema='cad_seating_shadow_v2', candidate_inserted=bool(all(checks.values())),
                candidate_fully_seated=bool(all(checks.values()) and depth >= .079),
                nominal_seating_gap_m=float(.080 - depth),
                checks={k: bool(v) for k, v in checks.items()}, insertion_depth_m=depth,
                axis_angle_deg=axis_angle, tip_radial_m=float(np.linalg.norm(tip[:2])),
                floor_penetration_m=floor_penetration, wall_penetration_m=wall_penetration,
                release_stability_verified=False)
