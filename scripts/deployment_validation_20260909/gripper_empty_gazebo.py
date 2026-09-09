"""Isolated empty-gripper integration probe. Launches and stops only its own Gazebo.

Run with system Python after sourcing ROS Humble and the workspace. The caller
selects original or isolated candidate plugin overlay; no model is involved.
"""
import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--candidate-prefix',type=Path)
    parser.add_argument('--ros-domain-id',type=int,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    os.environ['ROS_DOMAIN_ID']=str(args.ros_domain_id)
    os.environ['IGN_PARTITION']=f'pap_gripper_probe_{os.getpid()}'
    import rclpy
    from rclpy.action import ActionClient
    from sensor_msgs.msg import JointState
    from control_msgs.action import FollowJointTrajectory
    from controller_manager_msgs.srv import ListControllers
    from trajectory_msgs.msg import JointTrajectoryPoint
    rclpy.init()
    node=rclpy.create_node('empty_gripper_probe')
    observed=[]; feedback=[]
    def state_callback(msg):
        observed.append(dict(wall=time.time(),stamp=msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9,
                             name=list(msg.name),position=list(msg.position),velocity=list(msg.velocity)))
    subscription=node.create_subscription(JointState,'/joint_states',state_callback,20)
    client=ActionClient(node,FollowJointTrajectory,'/joint_trajectory_controller/follow_joint_trajectory')
    list_client=node.create_client(ListControllers,'/controller_manager/list_controllers')
    source='source /opt/ros/humble/setup.bash\nsource /home/ubuntu/ur3_ft300_ws/install/setup.bash\n'
    if args.candidate_prefix:
        source+=f'source {args.candidate_prefix}/local_setup.bash\n'
    source+='exec ros2 launch ur_simulation_gz ur3_ft300_robotiq.launch.py gazebo_gui:=false launch_rviz:=false'
    # This package's Gazebo-only launch does not start MoveIt or spawn peg/hole.
    (args.output/'launch_command.txt').write_text(source)
    log=(args.output/'gazebo.log').open('x')
    process=subprocess.Popen(['bash','-c',source],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    outcome={}
    try:
        deadline=time.monotonic()+100
        while (not observed or not client.server_is_ready()) and time.monotonic()<deadline:
            rclpy.spin_once(node,timeout_sec=.1)
            if process.poll() is not None: raise RuntimeError('Gazebo launch exited')
        if not observed or not client.server_is_ready(): raise TimeoutError('controller readiness')
        # An action endpoint exists while the JTC is still inactive/configuring.
        # Wait for lifecycle activation, not only action service discovery.
        active=False
        while time.monotonic()<deadline and not active:
            if list_client.wait_for_service(timeout_sec=.2):
                pending=list_client.call_async(ListControllers.Request())
                while not pending.done() and time.monotonic()<deadline:
                    rclpy.spin_once(node,timeout_sec=.05)
                if pending.done():
                    active=any(c.name=='joint_trajectory_controller' and c.state=='active'
                               for c in pending.result().controller)
            rclpy.spin_once(node,timeout_sec=.1)
        if not active: raise TimeoutError('controller lifecycle activation')
        assert node.count_publishers('/joint_states')==1, 'More than one simulator publishes joint states'
        (args.output/'isolation.json').write_text(json.dumps(dict(
            ros_domain_id=args.ros_domain_id,partition=os.environ['IGN_PARTITION'],
            joint_state_publishers=node.count_publishers('/joint_states'))))
        # Capture the loaded plugin, not merely the requested overlay path.
        loaded=[]
        for path in Path('/proc').glob('[0-9]*/maps'):
            try:
                if os.getpgid(int(path.parent.name))!=process.pid: continue
                entries=[s for s in path.read_text().splitlines() if 'libgz_hardware_plugins.so' in s]
                if entries: loaded.extend(entries)
            except (OSError,ProcessLookupError): pass
        assert loaded, 'Could not verify loaded gripper plugin'
        (args.output/'loaded_plugin.json').write_text(json.dumps(loaded,indent=2))
        if args.candidate_prefix:
            assert all(str(args.candidate_prefix) in x for x in loaded), loaded
        def wait_future(future,timeout):
            until=time.monotonic()+timeout
            while not future.done() and time.monotonic()<until:
                rclpy.spin_once(node,timeout_sec=.05)
            if not future.done(): raise TimeoutError('action response/result')
            return future.result()
        def command(value,duration,label):
            goal=FollowJointTrajectory.Goal()
            goal.trajectory.joint_names=['robotiq_85_left_knuckle_joint']
            point=JointTrajectoryPoint();point.positions=[value]
            point.time_from_start.sec=int(duration);point.time_from_start.nanosec=int((duration-int(duration))*1e9)
            goal.trajectory.points=[point]
            def receive(msg):
                f=msg.feedback
                feedback.append(dict(label=label,wall=time.time(),joints=list(f.joint_names),
                                     actual=list(f.actual.positions),desired=list(f.desired.positions)))
            sent=time.time()
            handle=wait_future(client.send_goal_async(goal,feedback_callback=receive),15)
            if not handle.accepted: raise RuntimeError('action rejected')
            result=wait_future(handle.get_result_async(),65)
            return dict(target=value,duration=duration,sent_wall=sent,status=result.status,
                        error_code=result.result.error_code,error_string=result.result.error_string)
        outcome['open']=command(0.,1.,'open')
        # Let state feedback include a genuine stationary read before closing.
        until=time.monotonic()+1.
        while time.monotonic()<until: rclpy.spin_once(node,timeout_sec=.05)
        outcome['close']=command(.8,.2,'close')
        until=time.monotonic()+1.
        while time.monotonic()<until: rclpy.spin_once(node,timeout_sec=.05)
        close=[r for r in feedback if r['label']=='close']
        j=close[-1]['joints'].index('robotiq_85_left_knuckle_joint')
        outcome['final_gripper']=close[-1]['actual'][j]
        outcome['close_target']=close[-1]['desired'][j]
        last_state=observed[-1]
        outcome['settled_gripper']=last_state['position'][last_state['name'].index('robotiq_85_left_knuckle_joint')]
        outcome['completed']=True
        print(json.dumps(outcome),flush=True)
    except Exception as exc:
        outcome.update(completed=False,error=repr(exc));raise
    finally:
        (args.output/'summary.json').write_text(json.dumps(outcome,indent=2))
        (args.output/'feedback.json').write_text(json.dumps(feedback))
        (args.output/'joint_states.json').write_text(json.dumps(observed))
        if process.poll() is None:
            os.killpg(process.pid,signal.SIGINT)
            try: process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGTERM)
                process.wait(timeout=10)
        # launch may exit before its Ruby/Gazebo grandchild. Clean the known
        # process group even after the launch parent has returned.
        try:
            os.killpg(process.pid,signal.SIGTERM)
            until=time.monotonic()+5.
            while time.monotonic()<until:
                try: os.killpg(process.pid,0)
                except ProcessLookupError: break
                time.sleep(.1)
            else: os.killpg(process.pid,signal.SIGKILL)
        except ProcessLookupError: pass
        log.close();node.destroy_node();rclpy.shutdown()


if __name__=='__main__': main()
