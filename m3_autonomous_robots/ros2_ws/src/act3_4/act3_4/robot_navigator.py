import math
import random
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Pose2D
from std_msgs.msg import Float32

# == Explorer parameters ==================================================
FORWARD_VEL   = 0.05   # m/s — forward speed
TURN_VEL      = 0.15   # rad/s — rotation speed while avoiding
OBSTACLE_STOP = 50.0   # cm — hard stop: start turning
OBSTACLE_SLOW = 70.0   # cm — soft zone: slow down


class RobotNavigator(Node):
    """
    Wall-avoiding explorer.

    The robot drives forward and rotates when it detects an obstacle.
    It does NOT depend on the MCL pose estimate to move

    Flow with MCL:
      1. Robot moves forward  →  motors spin  →  MCL updates particles via odometry
      2. Sensor detects a wall  →  MCL assigns weights  →  particles converge
      3. Repeat: each cycle the cloud concentrates more around the robot's real position
    """

    def __init__(self):
        super().__init__('robot_navigator')

        self.front_dist_cm: float = -1.0       # no detection (free space ahead)
        self.estimated_pose: Pose2D | None = None

        self._turning    = False
        self._turn_ticks = 0      # remaining 0.1 s cycles of the current turn
        self._turn_sign  = 1.0    # +1 turns left / -1 turns right
        self._log_tick   = 0

        self.dist_sub = self.create_subscription(
            Float32, '/front_dist_cm',  self._dist_cb, 10)
        self.pose_sub = self.create_subscription(
            Pose2D,  '/estimated_pose', self._pose_cb, 10)

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.create_timer(0.1, self._control_loop)   # 10 Hz

    def _dist_cb(self, msg: Float32):
        self.front_dist_cm = msg.data

    def _pose_cb(self, msg: Pose2D):
        self.estimated_pose = msg

    def _stop(self):
        self.cmd_pub.publish(Twist())

    def _control_loop(self):
        cmd = Twist()

        if self._turning:
            # Keep rotating until the scheduled tick count runs out
            cmd.angular.z = self._turn_sign * TURN_VEL
            cmd.linear.x  = 0.0
            self._turn_ticks -= 1
            if self._turn_ticks <= 0:
                self._turning = False

        else:
            obstacle_hard = (self.front_dist_cm > 0 and
                             self.front_dist_cm < OBSTACLE_STOP)
            obstacle_soft = (self.front_dist_cm > 0 and
                             self.front_dist_cm < OBSTACLE_SLOW)

            if obstacle_hard:
                # Close obstacle: start a random turn of 90–180° to find a clear path
                self._turning    = True
                self._turn_ticks = random.randint(15, 30)   # 1.5 – 3 s at 10 Hz
                self._turn_sign  = random.choice([-1.0, 1.0])
                cmd.angular.z    = self._turn_sign * TURN_VEL
                cmd.linear.x     = 0.0

            elif obstacle_soft:
                # Slow-down zone
                ratio         = ((self.front_dist_cm - OBSTACLE_STOP) /
                                 (OBSTACLE_SLOW - OBSTACLE_STOP))
                cmd.linear.x  = FORWARD_VEL * max(0.1, ratio)
                cmd.angular.z = 0.0

            else:
                # Clear path
                cmd.linear.x  = FORWARD_VEL
                cmd.angular.z = 0.0

        self.cmd_pub.publish(cmd)

        # Log the MCL estimated position roughly every 2 s
        self._log_tick += 1
        if self._log_tick % 20 == 0 and self.estimated_pose is not None:
            p = self.estimated_pose
            self.get_logger().info(
                f'[MCL] Posición estimada → '
                f'X:{p.x:.0f} cm  Y:{p.y:.0f} cm  θ:{math.degrees(p.theta):.0f}°'
                f'  |  Sensor: {self.front_dist_cm:.0f} cm'
            )


# ======================================================================== #

def main(args=None):
    rclpy.init(args=args)
    node = RobotNavigator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
