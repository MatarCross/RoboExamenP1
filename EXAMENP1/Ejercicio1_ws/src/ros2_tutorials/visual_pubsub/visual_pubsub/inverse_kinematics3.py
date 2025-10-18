import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Point
import numpy as np
import time
from numpy import cos, sin  # <-- ARREGLO 1: Importar funciones trigonométricas


class InverseKinematics(Node):

    def __init__(self):
        super().__init__('inverse_kinematics')
        self.joint_pub = self.create_publisher(JointState, 'joint_states', 10)
        self.target_sub = self.create_subscription(Point, 'target_position',
                                                   self.target_callback, 10)

        # Initial joint angles (6-DOF)
        self.q = np.array([-0.22, 0.01, 0.5, 0.1, 0.6, 0.7])  # [q1, q2, q3, q4, q5, q6]

        # Robot link lengths
        self.l1 = 2.0
        self.l2 = 1.5
        self.l3 = 0.8
        self.l4 = 0.5
        self.l5 = 1.3
        self.l6 = 0.9

        self.timer = self.create_timer(0.1, self.update_joints)
        self.target_pos = np.array([1.0, 0.05, 0.001])  # Default target

        # Parameters for IK
        self.step_size = 0.05
        self.max_iterations = 100
        self.tolerance = 0.1
        self.damping_factor = 0.1  # For damped least squares method

    def forward_kinematics(self, q):
        # ARREGLO 2: Desempaquetar las 6 articulaciones
        q1, q2, q3, q4, q5, q6 = q

        # ARREGLO 3: Usar 'self.l...' para todas las longitudes
        x = self.l1*cos(q1) - self.l2*sin(q1)*sin(q2) + self.l2*cos(q1)*cos(q2) - self.l3*sin(q3)*sin(q1 + q2) + self.l3*cos(q3)*cos(q1 + q2) + self.l5*cos(q5)*cos(q1 + q2 + q3 + q4) - self.l6*sin(q5)*sin(q6)*cos(q1 + q2 + q3 + q4) + self.l6*cos(q5)*cos(q6)*cos(q1 + q2 + q3 + q4)

        y = self.l1*sin(q1) + self.l2*sin(q1 + q2) + self.l3*sin(q1 + q2 + q3) + self.l5*sin(q1 + q2 + q3 + q4)*cos(q5) + self.l6*sin(q1 + q2 + q3 + q4)*cos(q5 + q6)

        z = self.l4 + self.l5*sin(q5) + self.l6*sin(q5 + q6)
        return np.array([x, y, z])

    def jacobian(self, q):
        # ARREGLO 2: Desempaquetar las 6 articulaciones
        q1, q2, q3, q4, q5, q6 = q

        # ARREGLO 3: Usar 'self.l...' para todas las longitudes
        j11 = -self.l1*sin(q1) - self.l2*sin(q1)*cos(q2) - self.l2*sin(q2)*cos(q1) - self.l3*sin(q3)*cos(q1 + q2) - self.l3*sin(q1 + q2)*cos(q3) - self.l5*sin(q1 + q2 + q3 + q4)*cos(q5) + self.l6*sin(q5)*sin(q6)*sin(q1 + q2 + q3 + q4) - self.l6*sin(q1 + q2 + q3 + q4)*cos(q5)*cos(q6)
        j12 = -self.l2*sin(q1)*cos(q2) - self.l2*sin(q2)*cos(q1) - self.l3*sin(q3)*cos(q1 + q2) - self.l3*sin(q1 + q2)*cos(q3) - self.l5*sin(q1 + q2 + q3 + q4)*cos(q5) + self.l6*sin(q5)*sin(q6)*sin(q1 + q2 + q3 + q4) - self.l6*sin(q1 + q2 + q3 + q4)*cos(q5)*cos(q6)
        j13 = -self.l3*sin(q3)*cos(q1 + q2) - self.l3*sin(q1 + q2)*cos(q3) - self.l5*sin(q1 + q2 + q3 + q4)*cos(q5) + self.l6*sin(q5)*sin(q6)*sin(q1 + q2 + q3 + q4) - self.l6*sin(q1 + q2 + q3 + q4)*cos(q5)*cos(q6)
        J14 = -self.l5*sin(q1 + q2 + q3 + q4)*cos(q5) + self.l6*sin(q5)*sin(q6)*sin(q1 + q2 + q3 + q4) - self.l6*sin(q1 + q2 + q3 + q4)*cos(q5)*cos(q6)
        J15 = -self.l5*sin(q5)*cos(q1 + q2 + q3 + q4) - self.l6*sin(q5)*cos(q6)*cos(q1 + q2 + q3 + q4) - self.l6*sin(q6)*cos(q5)*cos(q1 + q2 + q3 + q4)
        J16 = -self.l6*sin(q5)*cos(q6)*cos(q1 + q2 + q3 + q4) - self.l6*sin(q6)*cos(q5)*cos(q1 + q2 + q3 + q4)
        
        j21 = self.l1*cos(q1) + self.l2*cos(q1 + q2) + self.l3*cos(q1 + q2 + q3) + self.l5*cos(q5)*cos(q1 + q2 + q3 + q4) + self.l6*cos(q5 + q6)*cos(q1 + q2 + q3 + q4)
        j22 = self.l2*cos(q1 + q2) + self.l3*cos(q1 + q2 + q3) + self.l5*cos(q5)*cos(q1 + q2 + q3 + q4) + self.l6*cos(q5 + q6)*cos(q1 + q2 + q3 + q4)
        j23 = self.l3*cos(q1 + q2 + q3) + self.l5*cos(q5)*cos(q1 + q2 + q3 + q4) + self.l6*cos(q5 + q6)*cos(q1 + q2 + q3 + q4)
        J24 = self.l5*cos(q5)*cos(q1 + q2 + q3 + q4) + self.l6*cos(q5 + q6)*cos(q1 + q2 + q3 + q4)
        J25 = -self.l5*sin(q5)*sin(q1 + q2 + q3 + q4) - self.l6*sin(q5 + q6)*sin(q1 + q2 + q3 + q4)
        J26 = -self.l6*sin(q5 + q6)*sin(q1 + q2 + q3 + q4)
        
        j31 = 0
        j32 = 0
        j33 = 0
        J34 = 0
        J35 = self.l5*cos(q5) + self.l6*cos(q5 + q6)
        J36 = self.l6*cos(q5 + q6)

        # ARREGLO 4: La Jacobiana debe ser 3x6
        return np.array([[j11, j12, j13, J14, J15, J16], 
                         [j21, j22, j23, J24, J25, J26], 
                         [j31, j32, j33, J34, J35, J36]])

    def target_callback(self, msg):
        self.target_pos = np.array([msg.x, msg.y, msg.z])
        self.get_logger().info(
            f"New target received: [{msg.x}, {msg.y}, {msg.z}]")

    def update_joints(self):
        current_pos = self.forward_kinematics(self.q)
        error = self.target_pos - current_pos
        error_norm = np.linalg.norm(error)

        self.get_logger().info(f"Current position: {current_pos}")
        self.get_logger().info(f"Target position: {self.target_pos}")
        self.get_logger().info(f"Error: {error_norm}")

        if error_norm > self.tolerance:
            # Compute Jacobian
            J = self.jacobian(self.q) # J es 3x6
            JtJ = J.T @ J # JtJ es 6x6

            # ARREGLO 5: Calcular determinante de JtJ (6x6)
            determinant = np.linalg.det(JtJ)
            self.get_logger().info(f"Jacobian determinant: {determinant}")

            damping = self.damping_factor * np.eye(JtJ.shape[0]) # 6x6
            J_dls = np.linalg.solve(JtJ + damping, J.T @ error) # J_dls (delta_q) es 6x1

            # Apply update with step size
            self.q += J_dls * self.step_size

        # Publish updated joint states
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        # ARREGLO 6: Publicar los 6 nombres
        msg.name = ['q1', 'q2', 'q3', 'q4', 'q5', 'q6']
        msg.position = self.q.tolist()
        self.joint_pub.publish(msg)


def main():
    rclpy.init()
    node = InverseKinematics()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()