import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
from world import World, Drone, Building
import json
import math
import time
from datetime import datetime
from scipy.optimize import linear_sum_assignment


class DroneCombatEnv(gym.Env):
    """
    Drone Combat Environment for Reinforcement Learning
    
    This environment simulates a 3D combat scenario between two drones (red and blue)
    in a world with buildings as obstacles.
    """
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}

    def __init__(self, 
    config_path='world_config.json', 
    render_mode=None, max_steps=500, 
    record_replay=False, 
    replay_path=None, 
    num_red_drones=1, 
    num_blue_drones=1,
    max_drones_per_team=5):
        super(DroneCombatEnv, self).__init__()

        # Drone team configuration
        self.num_red_drones = min(num_red_drones, max_drones_per_team)
        self.num_blue_drones = min(num_blue_drones, max_drones_per_team)
        self.max_drones_per_team = max_drones_per_team
        
        # Load world configuration
        self.world = World(config_path=config_path)

        # Environment parameters
        self.max_steps = max_steps
        self.step_count = 0
        self.render_mode = render_mode
        self.drone_size = 0.25  # Size of drone (cube side length in meters)

        # Define action space for all blue drones: [dx1, dy1, dz1, shoot1, dx2, dy2, dz2, shoot2, ...]
        # dx, dy, dz are continuous values for movement
        # shoot is binary (0 or 1)
        # Action space size depends on number of blue drones
        action_dim_per_drone = 4  # [dx, dy, dz, shoot] for each drone
        total_action_dim = action_dim_per_drone * self.num_blue_drones
        
        # Create action space with appropriate dimensions
        low_values = np.array([-1.0, -1.0, -1.0, 0.0] * self.num_blue_drones)
        high_values = np.array([1.0, 1.0, 1.0, 1.0] * self.num_blue_drones)
        
        self.action_space = spaces.Box(
            low=low_values,
            high=high_values,
            dtype=np.float32
        )
        
        # Calculate observation space size based on number of drones
        # For each drone: [x, y, z] coordinates
        # Format: [own_x, own_y, own_z, opponent_1_x, opponent_1_y, opponent_1_z, ...]
        obs_size = 3 + (3 * self.num_red_drones)  # For blue agent: own position + all red drone positions
        
        # Define observation space
        # Each position is normalized to [0, 1] within the world bounds
        self.observation_space = spaces.Box(
            low=np.zeros(obs_size),
            high=np.ones(obs_size),
            dtype=np.float32
        )

        # Initialize drones
        self._initialize_drones()

        # Shooting parameters
        self.accuracy_base = 0.1  # Base accuracy (standard deviation at 1m distance)
        self.hit_radius = self.drone_size  # Hit if shot lands within this radius

        # Rewards
        self.hit_reward = 1.0
        self.kill_all_reward = 5.0  # Additional reward for killing all enemies
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
        """Initialize the drones based on the world configuration and specified counts"""
        # Initialize drone lists
        self.red_drones = []
        self.blue_drones = []
        self.drone_hit_status = {}  # Track which drones have been hit
        
        # Get drones from the world configuration
        existing_red_drones = [d for d in self.world.drones if d.team == "red"]
        existing_blue_drones = [d for d in self.world.drones if d.team == "blue"]
        
        # Use existing drones from config if available
        for i in range(self.num_red_drones):
            if i < len(existing_red_drones):
                # Use existing drone from config
                drone = existing_red_drones[i]
            else:
                # Create new drone with default position
                # Spread drones out in a line formation
                x = 0.5 + (i * 0.5)
                y = 0.5
                z = 1.0
                drone = Drone(x, y, z, "red")
                # Add to world
                self.world.drones.append(drone)
            
            self.red_drones.append(drone)
            self.drone_hit_status[drone] = False
        
        for i in range(self.num_blue_drones):
            if i < len(existing_blue_drones):
                # Use existing drone from config
                drone = existing_blue_drones[i]
            else:
                # Create new drone with default position
                # Spread drones out in a line formation on opposite side
                x = 9.0 - (i * 0.5)
                y = 9.0
                z = 1.0
                drone = Drone(x, y, z, "blue")
                # Add to world
                self.world.drones.append(drone)
            
            self.blue_drones.append(drone)
            self.drone_hit_status[drone] = False
            
        # For backward compatibility - keep references to first drone of each team
        if self.red_drones:
            self.red_drone = self.red_drones[0]
        if self.blue_drones:
            self.blue_drone = self.blue_drones[0]
            
        # Game state flags
        self.red_hit = False  # Any red drone hit
        self.blue_hit = False  # Any blue drone hit

    def _get_normalized_observation(self, agent_team, agent_index=0):
        """
        Get the normalized observation for a specific agent
        
        Args:
            agent_team: "red" or "blue" to specify which agent's perspective
            agent_index: Index of the agent within its team (default: 0)
            
        Returns:
            numpy array of normalized observations
        """
        world_size_x = self.world.size_x
        world_size_y = self.world.size_y
        world_size_z = self.world.size_z
        
        # Normalize a position
        def normalize_position(drone):
            return np.array([
                drone.x / world_size_x,
                drone.y / world_size_y,
                drone.z / world_size_z
            ])
        
        if agent_team == "red":
            # Get the specific red drone's position as 'own position'
            if agent_index < len(self.red_drones):
                own_drone = self.red_drones[agent_index]
                own_pos = normalize_position(own_drone)
            else:
                # Fallback if index is out of range
                own_pos = np.zeros(3)
                
            # Get all blue drones' positions as 'opponent positions'
            opponent_positions = []
            for blue_drone in self.blue_drones:
                opponent_positions.append(normalize_position(blue_drone))
                
            # If no opponents, add zeros
            if not opponent_positions:
                opponent_positions = [np.zeros(3) for _ in range(self.num_blue_drones)]
                
        else:  # blue
            # Get the specific blue drone's position as 'own position'
            if agent_index < len(self.blue_drones):
                own_drone = self.blue_drones[agent_index]
                own_pos = normalize_position(own_drone)
            else:
                # Fallback if index is out of range
                own_pos = np.zeros(3)
                
            # Get all red drones' positions as 'opponent positions'
            opponent_positions = []
            for red_drone in self.red_drones:
                opponent_positions.append(normalize_position(red_drone))
                
            # If no opponents, add zeros
            if not opponent_positions:
                opponent_positions = [np.zeros(3) for _ in range(self.num_red_drones)]
        
        # Flatten opponent positions into a 1D array
        flattened_opponent_pos = np.concatenate(opponent_positions).flatten()
        
        # Combine own position with all opponent positions
        return np.concatenate([own_pos, flattened_opponent_pos]).astype(np.float32)

    def _calculate_distance(self, drone1, drone2):
        """Calculate Euclidean distance between two drones"""
        return math.sqrt(
            (drone1.x - drone2.x) ** 2 +
            (drone1.y - drone2.y) ** 2 +
            (drone1.z - drone2.z) ** 2
        )
        
    def _assign_targets_hungarian(self, shooters, targets):
        """
        Use the Hungarian algorithm to optimally assign targets to shooters
        
        Args:
            shooters: List of shooter drones (blue team)
            targets: List of target drones (red team)
            
        Returns:
            List of (shooter, target) pairs representing optimal assignments
        """
        if not shooters or not targets:
            return []  # No assignment possible if either list is empty
        
        # Create cost matrix: rows=shooters, cols=targets
        # Each cell contains the cost (distance) between a shooter and target
        cost_matrix = np.zeros((len(shooters), len(targets)))
        
        # Fill the cost matrix with distances
        for i, shooter in enumerate(shooters):
            for j, target in enumerate(targets):
                # Base cost is distance
                distance = self._calculate_distance(shooter, target)
                
                # Check line of sight - add large penalty if no line of sight
                has_los = self._has_line_of_sight(shooter, target)
                los_penalty = 0 if has_los else 1000  # Large penalty for no line of sight
                
                # Final cost combines distance and line of sight
                cost_matrix[i, j] = distance + los_penalty
        
        # Use Hungarian algorithm to find optimal assignment
        # This minimizes the total cost (distance + penalties)
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        # Create assignment pairs
        assignments = []
        for row_idx, col_idx in zip(row_indices, col_indices):
            # Only include assignments where there's line of sight (cost < 1000)
            if cost_matrix[row_idx, col_idx] < 1000:  # No line of sight penalty
                assignments.append((shooters[row_idx], targets[col_idx]))
        
        return assignments

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

        # Number of points to check along the line - increase for better accuracy
        num_points = 20

        # Check points along the line of sight
        for i in range(1, num_points):
            # Calculate point position (exclude start and end points)
            t = i / num_points
            point_x = shooter.x + dx * t
            point_y = shooter.y + dy * t
            point_z = shooter.z + dz * t

            # Check if this point intersects with any building
            for building in self.world.buildings:
                # Check if point is inside building boundaries
                if (point_x >= building.x and point_x <= building.x + building.width and
                    point_y >= building.y and point_y <= building.y + building.depth and
                    point_z >= 0 and point_z <= building.height):
                    # Point is inside a building, no line of sight
                    return False

        # If we got here, there's a clear line of sight
        return True

    def _simulate_shot(self, shooter, target):
        """
        Simulate a shot from shooter to target with distance-based accuracy
        
        Returns:
            bool: True if hit, False otherwise
        """
        # Line of sight is now checked before calling this method
        # So we can assume there is line of sight

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
            observation: Initial observation for blue agent
            info: Additional information
        """
        super().reset(seed=seed)

        # Reset step counter
        self.step_count = 0

        # Reset world to initial state
        self.world.reset()

        # Re-initialize drones
        self._initialize_drones()

        # Get initial observation (from blue agent's perspective)
        observation = self._get_normalized_observation("blue")

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
            action: numpy array [dx, dy, dz, shoot] for blue agent
            
        Returns:
            observation: New observation
            reward: Reward for this step
            terminated: Whether episode is done
            truncated: Whether episode is truncated
            info: Additional information
        """
        # Increment step counter
        self.step_count += 1

        # Initialize rewards with base step penalty
        reward = self.step_penalty
        
        # Track collisions for each team
        red_collisions = [False] * len(self.red_drones)
        blue_collisions = [False] * len(self.blue_drones)
        
        # Parse actions for all blue drones
        # Each drone has 4 action values: [dx, dy, dz, shoot]
        blue_actions = []
        for i in range(self.num_blue_drones):
            # Extract action for each blue drone
            start_idx = i * 4
            if start_idx + 4 <= len(action):  # Ensure we have enough action values
                drone_action = action[start_idx:start_idx+4]
                blue_actions.append(drone_action)
            else:
                # Fallback if action space doesn't match expected size
                blue_actions.append(np.array([-1.0, -1.0, -1.0, 0.0]))
        
        # Move all red drones (with random actions for now)
        for i, red_drone in enumerate(self.red_drones):
            # Generate random action for each red drone
            red_action = np.array([-1.0, -1.0, -1.0, 0.0])
            red_action[:3] = np.random.uniform(-1.0, 1.0, 3)  # Random movement
            red_action[3] = np.random.choice([0.0, 1.0])  # Random shoot decision
            dx_red, dy_red, dz_red, shoot_red = red_action
            
            # Move the drone and check for collisions
            red_collisions[i] = self._move_drone(red_drone, dx_red, dy_red, dz_red)
        
        # Move all blue drones (all controlled by the agent)
        for i, blue_drone in enumerate(self.blue_drones):
            if i < len(blue_actions):  # Make sure we have actions for this drone
                dx_blue, dy_blue, dz_blue, _ = blue_actions[i]  # Extract movement components
                blue_collisions[i] = self._move_drone(blue_drone, dx_blue, dy_blue, dz_blue)
        
        # Apply collision penalties for the primary blue drone
        if blue_collisions[0]:  # Only penalize the primary blue drone's collisions
            reward += self.collision_penalty
        
        # Calculate proximity rewards based on average distance to enemies
        if self.blue_drones and self.red_drones:  # Only if both teams have drones
            # Get the primary blue drone
            primary_blue = self.blue_drones[0]
            
            # Calculate average distance to all red drones before and after movement
            # This is a simplification - more sophisticated reward schemes could be implemented
            avg_distance_before = sum(self._calculate_distance(primary_blue, red_drone) 
                                     for red_drone in self.red_drones) / len(self.red_drones)
            
            # Distance after movement (using updated positions)
            avg_distance_after = sum(self._calculate_distance(primary_blue, red_drone) 
                                    for red_drone in self.red_drones) / len(self.red_drones)
            
            # Calculate proximity reward (positive if getting closer)
            distance_change = avg_distance_before - avg_distance_after
            proximity_reward = distance_change * self.proximity_reward_factor
            reward += proximity_reward
        
        # Process shooting for red drones
        red_shots = []
        for i, red_drone in enumerate(self.red_drones):
            # Randomly decide if this red drone shoots
            shoot_red = np.random.random() > 0.5
            red_shots.append(shoot_red)
            
            if shoot_red:
                # Find closest blue drone as target
                if self.blue_drones:  # Only if there are blue drones left
                    # Find the closest blue drone
                    distances = [self._calculate_distance(red_drone, blue_drone) 
                               for blue_drone in self.blue_drones]
                    closest_idx = np.argmin(distances)
                    target_blue = self.blue_drones[closest_idx]
                    
                    # Check line of sight
                    has_los = self._has_line_of_sight(red_drone, target_blue)
                    
                    if has_los:
                        hit = self._simulate_shot(red_drone, target_blue)
                        if hit:
                            # Mark the drone as hit
                            self.drone_hit_status[target_blue] = True
                            
                            # If this was the primary blue drone, apply negative reward
                            if target_blue == self.blue_drone:  # Primary blue drone
                                reward -= self.hit_reward
                                self.blue_hit = True
                                
                                # Check if all blue drones are hit
                                if all(self.drone_hit_status.get(d, False) for d in self.blue_drones):
                                    print("FAILURE: RED team eliminated all BLUE drones!")
                                    reward -= self.kill_all_reward
        
        # Process shooting for all blue drones
        blue_shots = [False] * len(self.blue_drones)
        
        # Identify blue drones that are shooting
        shooting_blue_drones = []
        for i, blue_drone in enumerate(self.blue_drones):
            if i < len(blue_actions):  # Make sure we have actions for this drone
                # Extract shoot decision from action
                _, _, _, shoot_blue = blue_actions[i]
                
                if shoot_blue > 0.5:  # Blue drone decides to shoot
                    blue_shots[i] = True
                    shooting_blue_drones.append(blue_drone)
        
        # Get available red targets (not already hit)
        available_targets = [d for d in self.red_drones if not self.drone_hit_status.get(d, False)]
        
        # Use Hungarian algorithm for optimal target assignment if we have shooters and targets
        if shooting_blue_drones and available_targets:
            # Get optimal shooter-target assignments
            assignments = self._assign_targets_hungarian(shooting_blue_drones, available_targets)
            
            # Process each assignment
            for shooter, target in assignments:
                # Simulate the shot
                hit = self._simulate_shot(shooter, target)
                
                if hit:
                    # Mark the drone as hit
                    self.drone_hit_status[target] = True
                    
                    # Apply positive reward - each hit gives reward
                    reward += self.hit_reward
                    
                    # Check if this was the primary red drone
                    if target == self.red_drone:
                        self.red_hit = True
                    
                    print(f"Blue drone at ({shooter.x:.1f}, {shooter.y:.1f}, {shooter.z:.1f}) hit red drone at ({target.x:.1f}, {target.y:.1f}, {target.z:.1f})")
                else:
                    # Small penalty for missed shot
                    reward += self.missed_shot_penalty / len(shooting_blue_drones)  # Divide penalty among shooting drones
                    print(f"Blue drone at ({shooter.x:.1f}, {shooter.y:.1f}, {shooter.z:.1f}) missed red drone at ({target.x:.1f}, {target.y:.1f}, {target.z:.1f})")
            
            # Check if all red drones are hit after this round of shooting
            if all(self.drone_hit_status.get(d, False) for d in self.red_drones):
                print("SUCCESS: BLUE team eliminated all RED drones using optimal targeting!")
                reward += self.kill_all_reward
        
        # Check termination conditions
        # Episode ends if primary drones of either team are hit or max steps reached
        terminated = self.red_hit or self.blue_hit or self.step_count >= self.max_steps
        truncated = False
        
        # Get new observation for the primary blue agent
        observation = self._get_normalized_observation("blue", 0)

        # Create info dictionary with enhanced information for multiple drones
        info = {
            "red_hit": self.red_hit,
            "blue_hit": self.blue_hit,
            "step_count": self.step_count,
            "red_collisions": red_collisions,
            "blue_collisions": blue_collisions,
            "num_red_drones": len(self.red_drones),
            "num_blue_drones": len(self.blue_drones),
            "drone_hit_status": {str(i): hit for i, hit in enumerate(self.drone_hit_status.values())}
        }
        
        # Add line of sight information if primary drones exist
        if self.red_drones and self.blue_drones:
            info["has_line_of_sight"] = self._has_line_of_sight(self.red_drones[0], self.blue_drones[0])

        # Update world state (for visualization)
        self.world.update()

        # Record replay frame if enabled
        if self.record_replay:
            self._record_replay_frame(action, reward, self.step_count, red_shots, blue_shots)

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

    def _record_replay_frame(self, action, reward, step, red_shots, blue_shots):
        """Record a frame of replay data with support for multiple drones"""
        # Convert boolean inputs to lists if needed
        if isinstance(red_shots, bool):
            red_shots = [red_shots] * len(self.red_drones)
        elif red_shots is None or not hasattr(red_shots, '__iter__'):
            red_shots = [False] * len(self.red_drones)
            
        if isinstance(blue_shots, bool):
            blue_shots = [blue_shots] * len(self.blue_drones)
        elif blue_shots is None or not hasattr(blue_shots, '__iter__'):
            blue_shots = [False] * len(self.blue_drones)
        
        # Create frame data
        frame = {
            "step": int(step),
            "timestamp": datetime.now().isoformat(),
            "red_drones": [
                {
                    "x": float(drone.x),
                    "y": float(drone.y),
                    "z": float(drone.z),
                    "team": str(drone.team),
                    "shot": bool(shot) if i < len(red_shots) else False,
                    "hit": bool(self.drone_hit_status.get(drone, False))
                } for i, (drone, shot) in enumerate(zip(self.red_drones, red_shots))
            ],
            "blue_drones": [
                {
                    "x": float(drone.x),
                    "y": float(drone.y),
                    "z": float(drone.z),
                    "team": str(drone.team),
                    "shot": bool(shot) if i < len(blue_shots) else False,
                    "hit": bool(self.drone_hit_status.get(drone, False))
                } for i, (drone, shot) in enumerate(zip(self.blue_drones, blue_shots))
            ],
            "red_hit": bool(self.red_hit),
            "blue_hit": bool(self.blue_hit)
        }
        
        # Add line of sight information if primary drones exist
        if self.red_drones and self.blue_drones:
            frame["has_line_of_sight"] = bool(self._has_line_of_sight(self.red_drones[0], self.blue_drones[0]))

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
                # Use a simple default filename
                path = "replay.json"
            else:
                path = self.replay_path

        # Ensure the directory exists
        import os
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        # Create replay data structure
        replay = {
            "metadata": {
                "version": "1.0",
                "date": datetime.now().isoformat(),
                "total_steps": len(self.replay_data),
                "world_size": {
                    "x": float(self.world.size_x),
                    "y": float(self.world.size_y),
                    "z": float(self.world.size_z)
                }
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
            "frames": self.replay_data
        }

        # Custom JSON encoder to handle NumPy types
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.bool_):
                    return bool(obj)
                return super(NumpyEncoder, self).default(obj)

        # Save to file with custom encoder
        with open(path, 'w') as f:
            json.dump(replay, f, indent=2, cls=NumpyEncoder)

        print(f"Replay saved to {path}")
        return path

    def close(self):
        """Clean up resources"""
        # Save replay if recording was enabled
        if self.record_replay and self.replay_data:
            try:
                replay_path = self.save_replay()
                print(f"Successfully saved replay to {replay_path}")
            except Exception as e:
                print(f"Error saving replay: {e}")

        if self.viewer:
            self.viewer.close()
            self.viewer = None
