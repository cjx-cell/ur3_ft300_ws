"""Reproduce the checked-in brake logic with the logged 500/100 Hz scheduler.

This is a deterministic scheduling unit test, not a Gazebo contact simulation.
No production settings or raw evaluation results are modified.
"""
import json
from pathlib import Path

ROOT = Path('/home/ubuntu/ur3_ft300_ws')


def simulate(fresh_read_only, blocked=False):
    position = measured_position = measured_velocity = 0.
    velocity = 0.
    latch = False
    cycles = 0
    rows = []
    for step in range(1, 101):
        # PreUpdate writes every 2 ms, PostUpdate reads once every 10 ms.
        fresh = step == 1 or (step - 1) % 5 == 0
        error = .8 - measured_position
        if not latch and error > .03 and (fresh or not fresh_read_only):
            cycles = cycles + 1 if abs(measured_velocity) < .10 else 0
            latch = cycles >= 5
        commanded_velocity = 0. if latch else min(100. * error, .5)
        velocity = 0. if blocked else commanded_velocity
        position += velocity * .002
        if step % 5 == 0:
            measured_position = position
            measured_velocity = velocity
        rows.append(dict(ms=step*2, position=position, measured_velocity=measured_velocity,
                         cycles=cycles, latched=latch))
    return dict(final_position=position,
                first_latch_ms=next((r['ms'] for r in rows if r['latched']), None), rows=rows)


def main():
    old_free = simulate(False)
    fresh_free = simulate(True)
    fresh_blocked = simulate(True, True)
    assert old_free['first_latch_ms'] == 10
    assert abs(old_free['final_position'] - .004) < 1e-12
    assert fresh_free['first_latch_ms'] is None
    assert fresh_blocked['first_latch_ms'] == 42
    audit = json.loads((ROOT/'artifacts/deployment_validation_20260909_tolerance_v1/summary.json').read_text())
    logs = []
    for item in audit:
        source = Path(item['artifact'])/'gazebo.log'
        matches = [line for line in source.read_text().splitlines() if 'contact stop latched' in line]
        logs.append(dict(name=item['name'], lines=matches,
                         measured_increment=item['final_measured_gripper']-item['start_measured_gripper']))
        assert matches, item['name']
    result = dict(old_free=old_free, fresh_free=fresh_free, fresh_blocked=fresh_blocked,
                  original_log_evidence=logs,
                  limitation='Scheduling reproduction, not proof of actual contact absence; fresh-read fix not deployed.')
    target = ROOT/'artifacts/deployment_validation_20260909_tolerance_v1/scheduler_probe.json'
    with target.open('x') as out:
        json.dump(result, out, indent=2)
    print(json.dumps({k:{n:v for n,v in x.items() if n != 'rows'} for k,x in result.items() if k in ('old_free','fresh_free','fresh_blocked')}, indent=2))


if __name__ == '__main__':
    main()
