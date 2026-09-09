import rclpy
from rclpy.node import Node
import tf2_ros
import time

def main():
    rclpy.init()
    node = Node("test_tf_node")
    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer, node)
    
    print("Listening to TF for 5 seconds...")
    t_end = time.time() + 5.0
    while time.time() < t_end:
        rclpy.spin_once(node, timeout_sec=0.1)
        try:
            trans = tf_buffer.lookup_transform("world", "tool0", rclpy.time.Time())
            z = trans.transform.translation.z
            print(f"Current tool0 Z in world: {z:.4f}")
        except Exception as e:
            print(f"TF lookup failed: {e}")
        time.sleep(0.5)
        
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
