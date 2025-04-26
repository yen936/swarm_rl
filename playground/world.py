import json
import datetime
import os

class Building:
    def __init__(self, x, y, width, depth, height):
        self.x = x
        self.y = y
        self.width = width
        self.depth = depth
        self.height = height

    def __repr__(self):
        return f"Building(x={self.x}, y={self.y}, w={self.width}, d={self.depth}, h={self.height})"

class Drone:
    def __init__(self, x, y, z, team="red"):
        self.x = x
        self.y = y
        self.z = z
        self.team = team

    def __repr__(self):
        return f"Drone(x={self.x}, y={self.y}, z={self.z}, team={self.team})"

class World:
    def __init__(self, init_config_path='world_config_init.json', current_state_path='world.json'):
        self.buildings = []
        self.drones = []
        self.simulation_step = 0
        self.timestamp = datetime.datetime.now().isoformat()
        self.init_config_path = init_config_path
        self.current_state_path = current_state_path
        
        # If current state exists, load it; otherwise, load from init config
        if os.path.exists(current_state_path):
            self.load_state(current_state_path)
        else:
            self.load_config(init_config_path)
            self.save_state(current_state_path)
    
    def load_config(self, config_path):
        """Load initial configuration from a file"""
        with open(config_path, 'r') as f:
            config = json.load(f)

        self.size_x = config['world']['size_x']
        self.size_y = config['world']['size_y']
        self.size_z = config['world']['size_z']

        for b in config['buildings']:
            building = Building(b['x'], b['y'], b['width'], b['depth'], b['height'])
            self.buildings.append(building)

        for d in config['drones']:
            team = d.get('team', 'red')  # Default to red if team not specified
            drone = Drone(d['x'], d['y'], d['z'], team)
            self.drones.append(drone)
    
    def load_state(self, state_path):
        """Load current state from a file"""
        with open(state_path, 'r') as f:
            state = json.load(f)
        
        self.size_x = state['world']['size_x']
        self.size_y = state['world']['size_y']
        self.size_z = state['world']['size_z']
        
        # Load buildings
        for b in state['buildings']:
            building = Building(b['x'], b['y'], b['width'], b['depth'], b['height'])
            self.buildings.append(building)
        
        # Load drones
        for d in state['drones']:
            team = d.get('team', 'red')  # Default to red if team not specified
            drone = Drone(d['x'], d['y'], d['z'], team)
            self.drones.append(drone)
        
        # Load simulation metadata
        self.simulation_step = state.get('simulation_step', 0)
        self.timestamp = state.get('timestamp', datetime.datetime.now().isoformat())
    
    def save_state(self, state_path=None):
        """Save current state to a file"""
        if state_path is None:
            state_path = self.current_state_path
        
        # Update timestamp
        self.timestamp = datetime.datetime.now().isoformat()
        
        # Create state dictionary
        state = {
            'world': {
                'size_x': self.size_x,
                'size_y': self.size_y,
                'size_z': self.size_z
            },
            'buildings': [
                {
                    'x': b.x,
                    'y': b.y,
                    'width': b.width,
                    'depth': b.depth,
                    'height': b.height
                } for b in self.buildings
            ],
            'drones': [
                {
                    'x': d.x,
                    'y': d.y,
                    'z': d.z,
                    'team': d.team
                } for d in self.drones
            ],
            'timestamp': self.timestamp,
            'simulation_step': self.simulation_step
        }
        
        # Write to file
        with open(state_path, 'w') as f:
            json.dump(state, f, indent=4)
    
    def reset(self):
        """Reset the world to initial configuration"""
        self.buildings = []
        self.drones = []
        self.simulation_step = 0
        self.timestamp = datetime.datetime.now().isoformat()
        self.load_config(self.init_config_path)
        self.save_state()
    
    def update(self):
        """Update the world state (placeholder for simulation logic)"""
        # This is where you would implement simulation logic to update drone positions, etc.
        self.simulation_step += 1
        self.timestamp = datetime.datetime.now().isoformat()
        # Save the updated state
        self.save_state()
    
    def __repr__(self):
        return f"World(size=({self.size_x}, {self.size_y}, {self.size_z}), buildings={len(self.buildings)}, drones={len(self.drones)}, step={self.simulation_step})"

# Example usage
if __name__ == "__main__":
    world = World('world_config_init.json', 'world.json')
    print(world)
    print(f"Buildings: {world.buildings}")
    print(f"Drones: {world.drones}")
