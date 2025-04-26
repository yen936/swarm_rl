import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
from world import World, Drone, Building
import json
import math
import time
from datetime import datetime

class DroneCombatEnv(gym.Env):
    """
    Drone Combat Environment for Reinforcement Learning
    
    This environment simulates a 3D combat scenario between two drones (red and blue)
    in a world with buildings as obstacles.
    """
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}
    
    def __init__(self, config_path='world_config_init.json', render_mode=None, max_steps=500, record_replay=False, replay_path=None):
        super(DroneCombatEnv, self).__init__()
        
        # Load world configuration
        self.world = World(init_config_path=config_path)
        
        # Environment parameters
        self.max_steps = max_steps
        self.step_count = 0
        self.render_mode = render_mode
        self.drone_size = 0.25  # Size of drone (cube side length in meters)
        
        # Define action space: [dx, dy, dz, shoot]
        # dx, dy, dz are continuous values for movement
        # shoot is binary (0 or 1)
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32
        )
        
        # Define observation space: [own_x, own_y, own_z, opponent_x, opponent_y, opponent_z]
        # Each position is normalized to [0, 1] within the world bounds
        self.observation_space = spaces.Box(
            low=np.zeros(6),
            high=np.ones(6),
            dtype=np.float32
        )
        
        # Initialize drones
        self._initialize_drones()
        
        # Shooting parameters
        self.accuracy_base = 0.1  # Base accuracy (standard deviation at 1m distance)
        self.hit_radius = self.drone_size  # Hit if shot lands within this radius
        
        # Rewards
        self.hit_reward = 1.0
        self.kill_all_reward = 3.0  # Additional reward for killing all enemies
        self.step_penalty = -0.01
        self.missed_shot_penalty = -0.05  # Optional penalty for missed shots
        self.proximity_reward_factor = 0.02  # Reward for getting closer to the target
        self.collision_penalty = -1.0  # Penalty for colliding with a building
        
        # Visualization
        self.viewer = None
        
        # Replay recording
        self.record_replay = record_replay
        self.replay_path = replay_path
        self.replay_data = []
        self.current_step_data = {}
    
    def _initialize_drones(self):
        """Initialize the drones based on the world configuration"""
        # Get drones from the world
        if len(self.world.drones) >= 2:
            # Use existing drones from world config
            self.red_drone = next((d for d in self.world.drones if d.team == "red"), None)
            self.blue_drone = next((d for d in self.world.drones if d.team == "blue"), None)
        else:
            # Create default drones if not enough in config
            self.red_drone = Drone(0.5, 0.5, 1.0, "red")
            self.blue_drone = Drone(9.0, 9.0, 1.0, "blue")
        # Game state
        self.red_hit = False
        self.blue_hit = False
    
    def _get_normalized_observation(self, agent_team):
        """
        Get the normalized observation for a specific agent
        
        Args:
            agent_team: "red" or "blue" to specify which agent's perspective
            
        Returns:
            numpy array of normalized observations
        """
        world_size_x = self.world.size_x
        world_size_y = self.world.size_y
        world_size_z = self.world.size_z
        
        if agent_team == "red":
            own_pos = np.array([
                self.red_drone.x / world_size_x,
                self.red_drone.y / world_size_y,
                self.red_drone.z / world_size_z
            ])
            opponent_pos = np.array([
                self.blue_drone.x / world_size_x,
                self.blue_drone.y / world_size_y,
                self.blue_drone.z / world_size_z
            ])
        else:  # blue
            own_pos = np.array([
                self.blue_drone.x / world_size_x,
                self.blue_drone.y / world_size_y,
                self.blue_drone.z / world_size_z
            ])
            opponent_pos = np.array([
                self.red_drone.x / world_size_x,
                self.red_drone.y / world_size_y,
                self.red_drone.z / world_size_z
            ])
        
        return np.concatenate([own_pos, opponent_pos]).astype(np.float32)
    
    def _calculate_distance(self, drone1, drone2):
        """Calculate Euclidean distance between two drones"""
        return math.sqrt(
            (drone1.x - drone2.x) ** 2 +
            (drone1.y - drone2.y) ** 2 +
            (drone1.z - drone2.z) ** 2
        )
    
    def _check_building_collision(self, drone):
        """
        Check if a drone is colliding with any building
        
        Args:
            drone: The drone to check
            
        Returns:
            bool: True if collision, False otherwise
        """
        # Get drone dimensions (assuming cube)
        half_size = self.drone_size / 2
        
        # Check collision with each building
        for building in self.world.buildings:
            # Check if drone overlaps with building in all dimensions
            x_overlap = (drone.x + half_size > building.x) and (drone.x - half_size < building.x + building.width)
            y_overlap = (drone.y + half_size > building.y) and (drone.y - half_size < building.y + building.depth)
            z_overlap = (drone.z + half_size > 0) and (drone.z - half_size < building.height)
            
            if x_overlap and y_overlap and z_overlap:
                return True
        
        return False
    
    def _has_line_of_sight(self, shooter, target):
        """
        Check if shooter has line of sight to target (no buildings in between)
        
        Args:
            shooter: The shooting drone
            target: The target drone
            
        Returns:
            bool: True if line of sight exists, False otherwise
        """
        # Vector from shooter to target
        dx = target.x - shooter.x
        dy = target.y - shooter.y
        dz = target.z - shooter.z
        
        # Distance between drones
        distance = self._calculate_distance(shooter, target)
        
        # Number of points to check along the line
        num_points = 10
        
        # Check points along the line
        for i in range(1, num_points):
            # Calculate point position (exclude start and end points)
            t = i / num_points
            point_x = shooter.x + dx * t
            point_y = shooter.y + dy * t
            point_z = shooter.z + dz * t
            
            # Create a temporary drone at this position for collision check
            temp_drone = type('TempDrone', (), {'x': point_x, 'y': point_y, 'z': point_z})
            
            # Check if point is inside any building
            if self._check_building_collision(temp_drone):
                return False
        
        return True
    
    def _simulate_shot(self, shooter, target):
        """
        Simulate a shot from shooter to target with distance-based accuracy
        
        Returns:
            bool: True if hit, False otherwise
        """
        # Check line of sight first
        if not self._has_line_of_sight(shooter, target):
            return False
        
        # Calculate distance between drones
        distance = self._calculate_distance(shooter, target)
        
        # Accuracy decreases with distance (standard deviation increases)
        accuracy = self.accuracy_base * distance
        
        # Simulate shot with Gaussian noise
        shot_x = target.x + np.random.normal(0, accuracy)
        shot_y = target.y + np.random.normal(0, accuracy)
        shot_z = target.z + np.random.normal(0, accuracy)
        
        # Calculate distance between shot and target
        shot_distance = math.sqrt(
            (shot_x - target.x) ** 2 +
            (shot_y - target.y) ** 2 +
            (shot_z - target.z) ** 2
        )
        
        # Hit if shot lands within hit radius
        return shot_distance <= self.hit_radius
    
    def _move_drone(self, drone, dx, dy, dz):
        """
        Move a drone with the given deltas, respecting world boundaries and checking for collisions
        
        Args:
            drone: The drone to move
            dx, dy, dz: Movement deltas
            
        Returns:
            bool: True if collision occurred, False otherwise
        """
        # Scale movement (max 1 unit per step)
        max_movement = 0.2
        dx *= max_movement
        dy *= max_movement
        dz *= max_movement
        
        # Store original position for collision detection
        original_x, original_y, original_z = drone.x, drone.y, drone.z
        
        # Update position
        drone.x += dx
        drone.y += dy
        drone.z += dz
        
        # Clip to world boundaries
        drone.x = max(0, min(drone.x, self.world.size_x))
        drone.y = max(0, min(drone.y, self.world.size_y))
        drone.z = max(0, min(drone.z, self.world.size_z))
        
        # Check for collision with buildings
        if self._check_building_collision(drone):
            # Revert to original position if collision occurred
            drone.x, drone.y, drone.z = original_x, original_y, original_z
            return True
        
        return False
    
    def reset(self, seed=None, options=None):
        """
        Reset the environment to initial state
        
        Returns:
            observation: Initial observation for red agent
            info: Additional information
        """
        super().reset(seed=seed)
        
        # Reset step counter
        self.step_count = 0
        
        # Reset world to initial state
        self.world.reset()
        
        # Re-initialize drones
        self._initialize_drones()
        
        # Get initial observation (from red agent's perspective)
        observation = self._get_normalized_observation("red")
        
        # Additional info
        info = {}
        
        # Reset replay data if recording
        if self.record_replay:
            self.replay_data = []
            # Record initial state
            self._record_replay_frame(None, None, 0, False, False)
        
        return observation, info
    
    def step(self, action):
        """
        Take a step in the environment with the given action
        
        Args:
            action: numpy array [dx, dy, dz, shoot] for red agent
            
        Returns:
            observation: New observation
            reward: Reward for this step
            terminated: Whether episode is done
            truncated: Whether episode is truncated
            info: Additional information
        """
        # Increment step counter
        self.step_count += 1
        
        # Parse action
        dx_red, dy_red, dz_red, shoot_red = action
        
        # For now, blue agent takes random actions
        blue_action = self.action_space.sample()
        dx_blue, dy_blue, dz_blue, shoot_blue = blue_action
        
        # Calculate distance before movement
        distance_before = self._calculate_distance(self.red_drone, self.blue_drone)
        
        # Initialize rewards with base step penalty
        reward = self.step_penalty
        
        # Move drones and check for collisions
        red_collision = self._move_drone(self.red_drone, dx_red, dy_red, dz_red)
        blue_collision = self._move_drone(self.blue_drone, dx_blue, dy_blue, dz_blue)
        
        # Apply collision penalties if needed
        if red_collision:
            reward += self.collision_penalty
            self.red_hit = True  # End episode if drone collides with building
            
        if blue_collision:
            # For blue drone collision, we give a positive reward to red
            reward -= self.collision_penalty  # Negative of penalty is a reward
            self.blue_hit = True  # End episode if drone collides with building
        
        # Calculate distance after movement
        distance_after = self._calculate_distance(self.red_drone, self.blue_drone)
        
        # Calculate proximity reward (positive if getting closer)
        distance_change = distance_before - distance_after
        proximity_reward = distance_change * self.proximity_reward_factor
        
        # Add proximity reward
        reward += proximity_reward
        
        # Process shooting
        if shoot_red > 0.5:  # Red agent shoots if action[3] > 0.5
            hit = self._simulate_shot(self.red_drone, self.blue_drone)
            if hit:
                reward += self.hit_reward
                self.blue_hit = True
                # Add kill all reward if this was the only enemy
                if not any(d.team == 'blue' for d in self.world.drones if d != self.blue_drone):
                    reward += self.kill_all_reward
            else:
                reward += self.missed_shot_penalty
        
        if shoot_blue > 0.5:  # Blue agent shoots
            hit = self._simulate_shot(self.blue_drone, self.red_drone)
            if hit:
                reward -= self.hit_reward  # Negative reward if red gets hit
                # Also lose the kill all reward opportunity
                reward -= self.kill_all_reward
                self.red_hit = True
        
        # Check termination conditions
        terminated = self.red_hit or self.blue_hit or self.step_count >= self.max_steps
        truncated = False
        
        # Get new observation
        observation = self._get_normalized_observation("red")
        
        # Create info dictionary
        info = {
            "red_hit": self.red_hit,
            "blue_hit": self.blue_hit,
            "step_count": self.step_count,
            "red_collision": red_collision,
            "blue_collision": blue_collision,
            "has_line_of_sight": self._has_line_of_sight(self.red_drone, self.blue_drone)
        }
        
        # Update world state (for visualization)
        self.world.update()
        
        # Record replay frame if enabled
        if self.record_replay:
            self._record_replay_frame(action, reward, self.step_count, shoot_red > 0.5, shoot_blue > 0.5)
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        """
        Render the environment
        
        Returns:
            If render_mode is 'rgb_array': numpy array of rendered frame
            If render_mode is 'human': None (renders to screen)
        """
        if self.render_mode == "human":
            # Use the visualize_world.py script to render
            from visualize_world import visualize_world
            visualize_world('world.json')
            return None
        
        # For 'rgb_array' mode, we would need to implement a renderer that returns an image
        # This is more complex and would require using a library like matplotlib to render to an array
        # For now, we'll just return a placeholder
        if self.render_mode == "rgb_array":
            return np.zeros((400, 400, 3), dtype=np.uint8)  # Placeholder
    
    def _record_replay_frame(self, action, reward, step, red_shot, blue_shot):
        """Record a frame of replay data"""
        # Create frame data
        frame = {
            "step": step,
            "timestamp": datetime.now().isoformat(),
            "red_drone": {
                "x": float(self.red_drone.x),
                "y": float(self.red_drone.y),
                "z": float(self.red_drone.z),
                "team": self.red_drone.team
            },
            "blue_drone": {
                "x": float(self.blue_drone.x),
                "y": float(self.blue_drone.y),
                "z": float(self.blue_drone.z),
                "team": self.blue_drone.team
            },
            "buildings": [
                {
                    "x": float(b.x),
                    "y": float(b.y),
                    "width": float(b.width),
                    "depth": float(b.depth),
                    "height": float(b.height)
                } for b in self.world.buildings
            ],
            "has_line_of_sight": self._has_line_of_sight(self.red_drone, self.blue_drone),
            "red_shot": red_shot,
            "blue_shot": blue_shot,
            "red_hit": self.red_hit,
            "blue_hit": self.blue_hit
        }
        
        # Add action and reward if not initial frame
        if action is not None:
            frame["action"] = action.tolist() if hasattr(action, "tolist") else action
            frame["reward"] = float(reward)
        
        # Add to replay data
        self.replay_data.append(frame)
    
    def save_replay(self, path=None):
        """Save the recorded replay data to a JSON file"""
        if not self.record_replay or not self.replay_data:
            print("No replay data to save.")
            return
        
        # Use provided path or default
        if path is None:
            if self.replay_path is None:
                # Create a default filename with timestamp
                timestamp = int(time.time())
                path = f"replay_{timestamp}.json"
            else:
                path = self.replay_path
        
        # Create replay data structure
        replay = {
            "metadata": {
                "version": "1.0",
                "date": datetime.now().isoformat(),
                "total_steps": len(self.replay_data),
                "world_size": {
                    "x": self.world.size_x,
                    "y": self.world.size_y,
                    "z": self.world.size_z
                }
            },
            "frames": self.replay_data
        }
        
        # Save to file
        with open(path, 'w') as f:
            json.dump(replay, f, indent=2)
        
        print(f"Replay saved to {path}")
        return path
    
    def close(self):
        """Clean up resources"""
        # Save replay if recording was enabled
        if self.record_replay and self.replay_data:
            self.save_replay()
            
        if self.viewer:
            self.viewer.close()
            self.viewer = None

