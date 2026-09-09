"""Run existing demonstration through isolated candidate physics; never model score."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT=Path('/home/ubuntu/ur3_ft300_ws')
out=ROOT/'artifacts/deployment_validation_20260909_gripper_candidate_demo_v2'
out.mkdir(exist_ok=False)
runner=ROOT/'scripts/run_workspace50_lerobot_policy_gazebo.sh'
source=runner.read_text()
marker='source "$SETUP_FILE"\n'
assert source.count(marker)==1
prefix=ROOT/'artifacts/deployment_validation_20260909_gripper_candidate/install'
source=source.replace(marker,marker+f'source "{prefix}/local_setup.bash"\n')
(out/'runner_snapshot.sh').write_text(source)
(out/'manifest.json').write_text(json.dumps(dict(original_runner_sha256=hashlib.sha256(runner.read_bytes()).hexdigest(),
    candidate_prefix=str(prefix),kind='engineering_demonstration_replay',episode=1,
    note='Same saved actions and normal safety thresholds. Not a model rollout; not appended to formal scores.'),indent=2))
env=dict(os.environ,WORKSPACE50_ROS_DOMAIN_ID='83',WORKSPACE50_RECORD_VIDEO='true',
         WORKSPACE50_DIAGNOSTIC_TRACE='true',WORKSPACE50_GEOMETRY_SHADOW='true',
         POLICY_ACTION_CHUNK_MAX_STEP_RAD='0.13')
checkpoint=ROOT/'outputs/train/pi05_workspace50_global_stats_expert_only_30k_20260827/checkpoints/030000/pretrained_model'
with (out/'run.log').open('x') as log:
    result=subprocess.run(['bash',out/'runner_snapshot.sh','demonstration',checkpoint,'1','false'],env=env,
                          stdout=log,stderr=subprocess.STDOUT,timeout=1200)
(out/'completion.json').write_text(json.dumps(dict(exit_code=result.returncode)))
raise SystemExit(result.returncode)
