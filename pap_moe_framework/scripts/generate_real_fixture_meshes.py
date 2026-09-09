#!/usr/bin/env python3
"""Generate the lab-scale PAP-MoE peg and mating socket STL meshes.

All dimensions are metres.  The peg origin is at its centre of height so it
rests on the table when spawned at TABLE_Z + PEG_HEIGHT / 2.  The socket origin
is at its bottom face.  It reproduces the laboratory part: only the peg's
80 mm conical-frustum section enters the blind cavity; the lower 20 mm of the
100 mm socket is solid and there is no cylindrical cavity above the taper.
"""

from __future__ import annotations

import argparse
import math
import struct
from pathlib import Path


VISUAL_SEGMENTS = 128
# Keep the exact-fit collision profile at the same angular density as the
# visual/CAD profile. A 64-sided diagnostic reduced solver cost but produced
# a 129 N seating impulse after a tiny relative yaw, so it is not admissible
# for force-supervised training data.
COLLISION_SEGMENTS = 128


def _ring(
    radius: float,
    z: float,
    segments: int = VISUAL_SEGMENTS,
) -> list[tuple[float, float, float]]:
    return [
        (
            radius * math.cos(2.0 * math.pi * i / segments),
            radius * math.sin(2.0 * math.pi * i / segments),
            z,
        )
        for i in range(segments)
    ]


def _quad_strip(
    lower: list[tuple[float, float, float]],
    upper: list[tuple[float, float, float]],
    inward: bool = False,
) -> list[tuple[tuple[float, float, float], ...]]:
    triangles = []
    segments = len(lower)
    for i in range(segments):
        j = (i + 1) % segments
        faces = [
            (lower[i], lower[j], upper[j]),
            (lower[i], upper[j], upper[i]),
        ]
        if inward:
            faces = [tuple(reversed(face)) for face in faces]
        triangles.extend(faces)
    return triangles


def _annulus(
    inner: list[tuple[float, float, float]],
    outer: list[tuple[float, float, float]],
    upward: bool,
) -> list[tuple[tuple[float, float, float], ...]]:
    triangles = []
    segments = len(inner)
    for i in range(segments):
        j = (i + 1) % segments
        faces = [
            (inner[i], outer[i], outer[j]),
            (inner[i], outer[j], inner[j]),
        ]
        if not upward:
            faces = [tuple(reversed(face)) for face in faces]
        triangles.extend(faces)
    return triangles


def _disk(
    radius: float,
    z: float,
    upward: bool,
    segments: int = VISUAL_SEGMENTS,
):
    ring = _ring(radius, z, segments)
    centre = (0.0, 0.0, z)
    triangles = []
    for i in range(segments):
        j = (i + 1) % segments
        face = (centre, ring[i], ring[j])
        triangles.append(face if upward else tuple(reversed(face)))
    return triangles


def make_peg():
    # bottom tip -> frustum -> 20 mm collar -> 80 mm grasp handle
    profile = [
        (0.010, -0.090),
        (0.020, -0.010),
        (0.020, 0.010),
        (0.010, 0.010),
        (0.010, 0.090),
    ]
    rings = [_ring(radius, z) for radius, z in profile]
    triangles = _disk(profile[0][0], profile[0][1], upward=False)
    for lower, upper in zip(rings[:-1], rings[1:]):
        triangles.extend(_quad_strip(lower, upper))
    triangles.extend(_disk(profile[-1][0], profile[-1][1], upward=True))
    return triangles


def make_peg_body_collision():
    """Insertion/collar collision, recessed 1 mm above the physical bottom.

    The SDF supplies the missing envelope as an overlapping analytic rigid
    cylinder, making the first seating contact primitive-to-primitive.  The
    80 mm grasp handle is also supplied as an analytic cylinder in the SDF;
    keeping it out of this triangular mesh prevents facet-edge normals from
    ejecting the peg during force-limited gripping.
    """
    profile = [
        (0.010, -0.089),
        (0.020, -0.010),
        (0.020, 0.010),
    ]
    rings = [
        _ring(radius, z, COLLISION_SEGMENTS) for radius, z in profile
    ]
    triangles = _disk(
        profile[0][0], profile[0][1], upward=False,
        segments=COLLISION_SEGMENTS,
    )
    for lower, upper in zip(rings[:-1], rings[1:]):
        triangles.extend(_quad_strip(lower, upper))
    triangles.extend(_disk(
        profile[-1][0], profile[-1][1], upward=True,
        segments=COLLISION_SEGMENTS,
    ))
    return triangles


def make_socket(clearance: float = 0.0):
    outer_radius = 0.030
    socket_height = 0.100
    cavity_bottom = 0.020
    inner_bottom = 0.010 + clearance
    inner_top = 0.020 + clearance

    outer_bottom = _ring(outer_radius, 0.0)
    outer_top = _ring(outer_radius, socket_height)
    inner_opening = _ring(inner_top, socket_height)
    inner_cavity_bottom = _ring(inner_bottom, cavity_bottom)

    triangles = _disk(outer_radius, 0.0, upward=False)
    triangles.extend(_quad_strip(outer_bottom, outer_top))
    triangles.extend(_annulus(inner_opening, outer_top, upward=True))
    triangles.extend(_quad_strip(inner_cavity_bottom, inner_opening, inward=True))
    # Blind-hole floor; upward normal faces the cavity.
    triangles.extend(_disk(inner_bottom, cavity_bottom, upward=True))
    return triangles


def make_socket_side_collision(clearance: float = 0.0):
    """Watertight rigid socket wall without a triangulated cavity floor.

    The matching SDF adds the same 20 mm floor as an analytic rigid cylinder.
    This preserves the physical geometry while avoiding the numerically harsh
    dynamic-mesh to static-mesh bottom contact in DART.
    """
    outer_radius = 0.030
    socket_height = 0.100
    cavity_bottom = 0.020
    inner_bottom_radius = 0.010 + clearance
    inner_top_radius = 0.020 + clearance

    outer_bottom = _ring(outer_radius, 0.0, COLLISION_SEGMENTS)
    outer_top = _ring(outer_radius, socket_height, COLLISION_SEGMENTS)
    inner_bottom = _ring(
        inner_bottom_radius, 0.0, COLLISION_SEGMENTS
    )
    inner_floor = _ring(
        inner_bottom_radius, cavity_bottom, COLLISION_SEGMENTS
    )
    inner_opening = _ring(
        inner_top_radius, socket_height, COLLISION_SEGMENTS
    )

    triangles = _annulus(inner_bottom, outer_bottom, upward=False)
    triangles.extend(_quad_strip(outer_bottom, outer_top))
    triangles.extend(_annulus(inner_opening, outer_top, upward=True))
    triangles.extend(_quad_strip(inner_floor, inner_opening, inward=True))
    triangles.extend(_quad_strip(inner_bottom, inner_floor, inward=True))
    return triangles


def _normal(face):
    a, b, c = face
    ab = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
    ac = (c[0] - a[0], c[1] - a[1], c[2] - a[2])
    cross = (
        ab[1] * ac[2] - ab[2] * ac[1],
        ab[2] * ac[0] - ab[0] * ac[2],
        ab[0] * ac[1] - ab[1] * ac[0],
    )
    norm = math.sqrt(sum(value * value for value in cross))
    if norm == 0.0:
        return (0.0, 0.0, 0.0)
    return tuple(value / norm for value in cross)


def write_binary_stl(path: Path, name: str, triangles) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = name.encode("ascii")[:80].ljust(80, b"\0")
    with path.open("wb") as stream:
        stream.write(header)
        stream.write(struct.pack("<I", len(triangles)))
        for face in triangles:
            values = (*_normal(face), *face[0], *face[1], *face[2])
            stream.write(struct.pack("<12fH", *values, 0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "src/ur_simulation_gz/ur_simulation_gz/models",
    )
    parser.add_argument(
        "--radial-clearance",
        type=float,
        default=0.0,
        help=(
            "Optional diagnostic-only radial enlargement in metres. "
            "The laboratory/CAD contract uses zero nominal clearance."
        ),
    )
    args = parser.parse_args()
    if args.radial_clearance < 0.0:
        parser.error("--radial-clearance must be non-negative")

    peg_path = args.output_root / "pap_moe_real_peg/meshes/peg.stl"
    peg_collision_path = (
        args.output_root / "pap_moe_real_peg/meshes/peg_body_collision.stl"
    )
    socket_path = args.output_root / "pap_moe_real_hole/meshes/hole.stl"
    socket_collision_path = (
        args.output_root
        / "pap_moe_real_hole/meshes/hole_side_collision.stl"
    )
    write_binary_stl(peg_path, "PAP-MoE real 180 mm peg", make_peg())
    write_binary_stl(
        peg_collision_path,
        "PAP-MoE rigid peg body collision",
        make_peg_body_collision(),
    )
    write_binary_stl(
        socket_path,
        "PAP-MoE real 100 mm tapered socket",
        make_socket(args.radial_clearance),
    )
    write_binary_stl(
        socket_collision_path,
        "PAP-MoE rigid socket side collision",
        make_socket_side_collision(args.radial_clearance),
    )
    print(peg_path)
    print(peg_collision_path)
    print(socket_path)
    print(socket_collision_path)


if __name__ == "__main__":
    main()
