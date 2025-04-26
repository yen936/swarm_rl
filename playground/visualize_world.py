import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from world import World

def plot_building(ax, building):
    """Plot a building as a 3D cuboid"""
    # Define the vertices of the cuboid
    x = building.x
    y = building.y
    w = building.width
    d = building.depth
    h = building.height
    
    # Define the 8 vertices of the cuboid
    vertices = [
        [x, y, 0], [x+w, y, 0], [x+w, y+d, 0], [x, y+d, 0],
        [x, y, h], [x+w, y, h], [x+w, y+d, h], [x, y+d, h]
    ]
    
    # Define the 6 faces of the cuboid using the vertices
    faces = [
        [vertices[0], vertices[1], vertices[2], vertices[3]],  # bottom
        [vertices[4], vertices[5], vertices[6], vertices[7]],  # top
        [vertices[0], vertices[1], vertices[5], vertices[4]],  # front
        [vertices[2], vertices[3], vertices[7], vertices[6]],  # back
        [vertices[0], vertices[3], vertices[7], vertices[4]],  # left
        [vertices[1], vertices[2], vertices[6], vertices[5]]   # right
    ]
    
    # Create a 3D collection of polygons
    collection = Poly3DCollection(faces, alpha=0.5, linewidths=1, edgecolors='black')
    collection.set_facecolor('lightgray')
    ax.add_collection3d(collection)

def plot_drone(ax, drone):
    """Plot a drone as a colored sphere based on team"""
    # Use the team name as the color if it's a valid color name, otherwise default to red
    # Common colors in matplotlib: red, blue, green, yellow, cyan, magenta, black, etc.
    ax.scatter(drone.x, drone.y, drone.z, color=drone.team, s=100, marker='o', label=drone.team)

def visualize_world(config_path='world.json'):
    """Create a 3D visualization of the world with buildings and drones"""
    # Load the world from config
    world = World(config_path)
    
    # Create a 3D figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot each building
    for building in world.buildings:
        plot_building(ax, building)
    
    # Plot each drone
    for drone in world.drones:
        plot_drone(ax, drone)
    
    # Set axis labels and limits
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(0, world.size_x)
    ax.set_ylim(0, world.size_y)
    ax.set_zlim(0, world.size_z)
    
    # Set a grid
    ax.grid(True)
    
    # Set a title
    plt.title('3D Drone Playground Visualization')
    
    # Add a legend for drone teams
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), title="Teams")
    
    # Adjust the view angle for better visualization
    ax.view_init(elev=30, azim=45)
    
    plt.show()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize the 3D world with buildings and drones')
    parser.add_argument('--config', type=str, default='world.json', 
                        help='Path to the configuration file (default: world.json)')
    parser.add_argument('--init', action='store_true',
                        help='Use the initial configuration file (world_config_init.json)')
    args = parser.parse_args()
    
    if args.init:
        visualize_world('world_config_init.json')
    else:
        visualize_world(args.config)
