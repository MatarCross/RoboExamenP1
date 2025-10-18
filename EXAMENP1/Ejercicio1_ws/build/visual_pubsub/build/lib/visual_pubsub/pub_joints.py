import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState # <- tipo de mensaje con 'name', 'position', 'velocity', 'effort'
from builtin_interfaces.msg import Time


class JointStatePublisher(Node):

    def __init__(self):
        super().__init__('joint_state_publisher')  # nombre del nodo
        self.publisher_ = self.create_publisher(JointState, 'joint_states', 10) # tópico
        self.timer = self.create_timer(
            0.1, self.publish_joint_states)  # Publish every 0.1s

    def publish_joint_states(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()  # Add timestamp
        msg.name = [ # <- nombres de juntas que se publican (deben existir en tu URDF si quieres que RViz se anime)
            'q1', 'q2', 'q3'
        ]
        msg.position = [2.4, -4.8, 0.76] # <- posiciones de las juntas

        self.publisher_.publish(msg)
        self.get_logger().info(
            f'Published Joint States: {msg.position}')  # Debugging info


def main(args=None):
    rclpy.init(args=args)
    node = JointStatePublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
