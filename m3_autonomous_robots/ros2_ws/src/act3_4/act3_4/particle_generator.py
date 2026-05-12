import sys
import math
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Pose2D
from std_msgs.msg import Float32

sys.path.append('/home/kim/CoppeliaSim/programming/zmqRemoteApi/clients/python/src')
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

"""
=================================================
  Coppelia Simulator Plane Coordinates (cm)
  (Viewing from the chosen angle)
=================================================
  (0,0) ----------------------- (500,0)
    |                               |
    |                               |
    |                               |
  (0,500) --------------------- (500,500)
=================================================
"""

# == Walls (cm) ==========================================================
WALLS_CM = [
    (200,   0, 500,   0),   # Top
    (500,   0, 500, 500),   # Full Right
    (500, 500, 100, 500),   # Bottom
    (100, 500, 100, 300),   # Lower Left
    (100, 300, 200, 300),   # Step
    (200, 300, 200,   0),   # Upper Left
    (300, 300, 500, 300),   # Inner Divider
]

INTERIOR_SEED_PX = (350, 150)   # A known point inside the free space
MAP_W = MAP_H    = 600          # Canvas size in pixels (1 px = 1 cm)
WALL_THICKNESS   = 3            

# == Alignment Offset ====================================================
OFFSET_X_M =  -2.5   # Align particles with the designed map
OFFSET_Y_M =   2.5   

# == Particle Parameters =================================================
NUM_PARTICLES   = 500
PARTICLE_Z      = 0.05   # Z height to make particles visible above the ground
PARTICLE_SIZE   = 6.0    # Visual size of the points in CoppeliaSim
UPDATE_PERIOD_S = 0.3    # Seconds between re-sampling

# == Pioneer P3DX Drive Parameters =======================================
WHEEL_RADIUS_M  = 0.097   # meters
WHEEL_BASE_M    = 0.381   # meters (distance between left and right wheels)


# == Free Space Map ======================================================
class FreeSpaceMap:
    """ Use of OpenCV to find which coordinates are valid for placing particles """

    def __init__(self):
        # 1. Create a black canvas the size of the map
        wall_img = np.zeros((MAP_H, MAP_W), dtype=np.uint8)
        
        # 2. Draw the walls as white lines on the canvas
        for x1, y1, x2, y2 in WALLS_CM:
            cv2.line(wall_img, (x1, y1), (x2, y2), 255, WALL_THICKNESS)
        canvas = wall_img.copy()
        
        # 3. FloodFill: Starts at the INTERIOR_SEED_PX point and paints all empty space gray until it hits the walls.
        cv2.floodFill(canvas, np.zeros((MAP_H + 2, MAP_W + 2), dtype=np.uint8), INTERIOR_SEED_PX, 128)

        # 4. Find and store all coordinates that were painted gray
        free_mask = (canvas == 128)
        self.free_cells = np.argwhere(free_mask)  # Stores a list of [Y, X] coordinates
        self.free_mask  = free_mask                # Boolean grid for fast wall-collision checks
        print(f'[FreeSpaceMap] {len(self.free_cells)} celdas libres '
              f'({len(self.free_cells)/(MAP_W*MAP_H)*100:.1f}% del canvas)')

        # 5. Visualize the OpenCV map for debugging
        print('[FreeSpaceMap] Mostrando mapa de OpenCV. Presiona cualquier tecla en la ventana de la imagen para continuar...')
        cv2.imshow('OpenCV - Free Space Map', canvas)
        cv2.waitKey(2000)  # Muestra el mapa 2 segundos para no bloquear el inicio de ROS 2
        cv2.destroyAllWindows()

    def sample(self, n: int) -> np.ndarray:
        """Chooses 'n' random coordinates from our list of free spaces."""
        idx = np.random.randint(0, len(self.free_cells), size=n)
        rc = self.free_cells[idx]    # Get the selected [row, col]
        return rc[:, [1, 0]]         # Swap to [col=x, row=y]


# == ROS 2 Node ===============================================================
class MCLParticleSampler(Node):

    def __init__(self):
        super().__init__('mcl_particle_sampler')

        self.get_logger().info('Connecting to CoppeliaSim...') # Connection via ZMQ 
        self.client = RemoteAPIClient()
        self.sim    = self.client.getObject('sim')

        # Handle of Laser Sensor in CoppeliaSim
        self.laser_handle = None
        try:
            self.laser_handle = self.sim.getObject('/PioneerP3DX/proximitySensor')
        except Exception:
            self.get_logger().warn('Sensor no encontrado. Verifica que la escena esté corriendo.')

        # Handles for the Pioneer drive motors
        self.left_motor  = self.sim.getObject('/PioneerP3DX/leftMotor')
        self.right_motor = self.sim.getObject('/PioneerP3DX/rightMotor')
        # Ensure robot starts stopped
        self.sim.setJointTargetVelocity(self.left_motor,  0.0)
        self.sim.setJointTargetVelocity(self.right_motor, 0.0)
        self.get_logger().info('Connection Established!')

        
        self.free_map = FreeSpaceMap()

        # Create a special drawing object in CoppeliaSim for efficient point rendering
        self.draw_handle = self.sim.addDrawingObject(self.sim.drawing_points, PARTICLE_SIZE, 0.0, -1, NUM_PARTICLES + 10, [1.0, 0.15, 0.15])
        self.particles = self._sample()
        self._draw()

        # Calls the function every second
        self.create_timer(UPDATE_PERIOD_S, self._on_timer)
        self.get_logger().info(
            f'{NUM_PARTICLES} active particles | '
            f'updating every {UPDATE_PERIOD_S}s'
        )

        # State of the real robot 
        self.v_actual = 0.0  # Linear velocity (cm/s)
        self.w_actual = 0.0  # Angular velocity (rad/s)

        # Publishers
        self.pose_pub = self.create_publisher(Pose2D,   '/estimated_pose', 10)
        self.dist_pub = self.create_publisher(Float32,  '/front_dist_cm',  10)

        # Subscription to listen for when you move the robot
        self.cmd_sub = self.create_subscription(
            Twist,
            '/cmd_vel',
            self._cmd_vel_callback,
            10
        )

    def _cmd_vel_callback(self, msg: Twist):
        """Reads teleop velocity and drives the Pioneer motors directly via ZMQ."""
        self.v_actual = msg.linear.x
        self.w_actual = msg.angular.z

        # Differential drive: convert (v, w) → individual wheel speeds (rad/s)
        left_vel  = (self.v_actual - self.w_actual * WHEEL_BASE_M / 2.0) / WHEEL_RADIUS_M
        right_vel = (self.v_actual + self.w_actual * WHEEL_BASE_M / 2.0) / WHEEL_RADIUS_M

        self.sim.setJointTargetVelocity(self.left_motor,  left_vel)
        self.sim.setJointTargetVelocity(self.right_motor, right_vel)

    def _motion_update_step_G_H(self, dt_segundos):
        """
        Steps G & H: Move all particles based on odometry (Dead Reckoning).
        Each particle is displaced by the robot's velocity plus Gaussian noise
        to account for motion uncertainty.
        """
        # If the robot is stationary, there is no movement to propagate
        if self.v_actual == 0.0 and self.w_actual == 0.0:
            return

        v_cm_s = self.v_actual * 100.0

        sigma_v = max(0.5,   abs(v_cm_s       * 0.10))    # min 0.5 cm/s
        sigma_w = max(0.005, abs(self.w_actual * 0.10))   # min 0.005 rad/s

        ruido_v = np.random.normal(0, sigma_v, NUM_PARTICLES)
        ruido_w = np.random.normal(0, sigma_w, NUM_PARTICLES)

        v_con_ruido = v_cm_s + ruido_v
        w_con_ruido = self.w_actual + ruido_w

        self.particles[:, 0] += v_con_ruido * np.cos(self.particles[:, 2]) * dt_segundos
        # Y is negated: pixel Y grows downward, but world Y (and sin) grows upward
        self.particles[:, 1] -= v_con_ruido * np.sin(self.particles[:, 2]) * dt_segundos
        self.particles[:, 2] += w_con_ruido * dt_segundos

        # Remove particles that moved through a wall or out of bounds
        self._clip_to_free_space()

    def _sample(self) -> np.ndarray:
        """Creates particles by assigning them X, Y, Orientation, and a Weight (probability)."""
        # Get valid X, Y positions
        xy     = self.free_map.sample(NUM_PARTICLES).astype(float)
        # Generate random orientations between -180 and 180 degrees 
        theta  = np.random.uniform(-math.pi, math.pi, (NUM_PARTICLES, 1))
        # Initially, all particles have the same probability
        weight = np.full((NUM_PARTICLES, 1), 1.0 / NUM_PARTICLES)
        # Stack everything into a single array [X, Y, Theta, Weight]
        return np.hstack([xy, theta, weight])

    def _draw(self):
        """Sends the particles to CoppeliaSim to be displayed."""
        self.sim.addDrawingObjectItem(self.draw_handle, None)   # Clear previous points
        for p in self.particles:
            # Transform from cm to m and shift origin to match CoppeliaSim's world
            x_cop =  p[0] / 100.0 + OFFSET_X_M
            # Negative because Y axis grows downwards
            y_cop = -p[1] / 100.0 + OFFSET_Y_M   
            self.sim.addDrawingObjectItem(self.draw_handle, [x_cop, y_cop, PARTICLE_Z])

    def _clip_to_free_space(self):
        """Reassigns any particle that escaped into a wall back to a random free cell."""
        xi = np.clip(self.particles[:, 0].astype(int), 0, MAP_W - 1)
        yi = np.clip(self.particles[:, 1].astype(int), 0, MAP_H - 1)
        invalid = ~self.free_map.free_mask[yi, xi]
        if invalid.any():
            n = int(invalid.sum())
            new_xy = self.free_map.sample(n).astype(float)
            self.particles[invalid, 0] = new_xy[:, 0]
            self.particles[invalid, 1] = new_xy[:, 1]
            self.particles[invalid, 3] = 1.0 / NUM_PARTICLES

    def _sensor_update_step_E(self, distancia_real_cm):
        """
        Step E: assigns a weight to every particle by comparing the distance
        the real sensor reads with the distance each particle 'would see'
        if it were the robot (vectorised ray-cast over all 500 particles at once).
        """
        sigma = 15.0   # forgive up to ~15 cm of error

        px    = self.particles[:, 0]
        py    = self.particles[:, 1]
        theta = self.particles[:, 2]

        # Ray end-point far enough to guarantee hitting a wall
        rx = px + 1000.0 * np.cos(theta)
        ry = py - 1000.0 * np.sin(theta)   # Y-axis inverted in map

        dist_min = np.full(NUM_PARTICLES, 1000.0)

        for x1, y1, x2, y2 in WALLS_CM:
            den = (px - rx) * (y1 - y2) - (py - ry) * (x1 - x2)
            nz  = den != 0

            t = np.where(nz, ((px - x1) * (y1 - y2) - (py - y1) * (x1 - x2)) / np.where(nz, den, 1.0), -1.0)
            u = np.where(nz, -((px - rx) * (py - y1) - (py - ry) * (px - x1)) / np.where(nz, den, 1.0), -1.0)

            hit  = nz & (t >= 0.0) & (t <= 1.0) & (u >= 0.0) & (u <= 1.0)
            ix   = px + t * (rx - px)
            iy   = py + t * (ry - py)
            dist = np.sqrt((ix - px) ** 2 + (iy - py) ** 2)

            dist_min = np.where(hit & (dist < dist_min), dist, dist_min)

        error = distancia_real_cm - dist_min
        pesos = np.exp(-(error ** 2) / (2.0 * sigma ** 2))
        suma  = pesos.sum()
        self.particles[:, 3] = pesos / (suma if suma > 1e-10 else 1e-10)

    def _resample(self):
        """
        Systematic resampling: particles with high weight are duplicated,
        particles with low weight are discarded. This makes the cloud converge
        toward the robot's actual position.
        """
        weights = self.particles[:, 3]
        weights /= weights.sum()
        cumsum = np.cumsum(weights)
        cumsum[-1] = 1.0  # Correct floating-point drift
        positions = (np.random.random() + np.arange(NUM_PARTICLES)) / NUM_PARTICLES
        indices = np.searchsorted(cumsum, positions)
        self.particles = self.particles[indices].copy()

        # Post-resample roughening: avoids particle impoverishment.
        self.particles[:, 0] += np.random.normal(0, 3.0, NUM_PARTICLES)
        self.particles[:, 1] += np.random.normal(0, 3.0, NUM_PARTICLES)
        self.particles[:, 2] += np.random.normal(0, 0.05, NUM_PARTICLES)
        self.particles[:, 3]  = 1.0 / NUM_PARTICLES

        # Roughening can push particles into walls too
        self._clip_to_free_space()

    def _on_timer(self):
        # Dead Reckoning — move all particles based on robot odometry 
        self._motion_update_step_G_H(UPDATE_PERIOD_S)

        # Assign scores based on sensor reading 
        if self.laser_handle is not None:
            res, dist_m, _, _, _ = self.sim.readProximitySensor(self.laser_handle)

            # Publish sensor distance so the navigator can react to obstacles
            dist_msg      = Float32()
            dist_msg.data = float(dist_m * 100.0) if res > 0 else -1.0
            self.dist_pub.publish(dist_msg)

            if res > 0:
                dist_cm = dist_m * 100.0
                self._sensor_update_step_E(dist_cm)
                self.get_logger().info(f'[Sensor] {dist_cm:.1f} cm detectados')

                # Filter: resample only when we have real sensor data
                self._resample()
            else:
                self.get_logger().warning('[Sensor] Sin detección')

        # Publish best estimate every tick so the navigator always has a pose.
        mejor_idx = np.argmax(self.particles[:, 3])
        mejor_p   = self.particles[mejor_idx]
        self.get_logger().info(
            f'[MCL] Mejor → X:{mejor_p[0]:.0f} Y:{mejor_p[1]:.0f} '
            f'θ:{math.degrees(mejor_p[2]):.0f}° peso:{mejor_p[3]:.4f}'
        )
        pose_msg       = Pose2D()
        pose_msg.x     = float(mejor_p[0])
        pose_msg.y     = float(mejor_p[1])
        pose_msg.theta = float(mejor_p[2])
        self.pose_pub.publish(pose_msg)

        # Redraw - visualize current particle set in CoppeliaSim
        self._draw()
        


def main(args=None):
    rclpy.init(args=args)
    node = MCLParticleSampler()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()