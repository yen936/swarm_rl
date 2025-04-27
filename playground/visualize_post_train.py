import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import os


def plot_building(ax, building):
    """Plot a building as a 3D cuboid"""
    x = building['x']
    y = building['y']
    w = building['width']
    d = building['depth']
    h = building['height']

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

    collection = Poly3DCollection(faces, alpha=0.5, linewidths=1, edgecolors='black')
    collection.set_facecolor('lightgray')
    ax.add_collection3d(collection)


def plot_drone_paths(ax, drone_paths, color, label_prefix, start_points=None, hit_points=None, highlight_ids=None):
    # drone_paths: dict of drone_id -> list of (x, y, z)
    for drone_id, path in drone_paths.items():
        path = np.array(path)
        if highlight_ids and drone_id in highlight_ids:
            ax.plot(path[:, 0], path[:, 1], path[:, 2], color='gold', linewidth=3, label=f"{label_prefix} {drone_id} (Winner)" if drone_id == 0 else None, zorder=10)
        else:
            ax.plot(path[:, 0], path[:, 1], path[:, 2], color=color, label=f"{label_prefix} {drone_id}" if drone_id == 0 else None)
    # Mark all starting points
    if start_points:
        starts = np.array(start_points)
        ax.scatter(starts[:, 0], starts[:, 1], starts[:, 2], color=color, marker='o', s=60, edgecolor='black', label=f"{label_prefix} Start")
    # Mark all hit points
    if hit_points:
        hits = np.array(hit_points)
        if len(hits) > 0:
            # Draw X with dark edge for visibility
            ax.scatter(hits[:, 0], hits[:, 1], hits[:, 2], color=color, marker='X', s=100, edgecolors='black', linewidths=2, label=f"{label_prefix} Hit")


def extract_paths(frames, team):
    # Returns a dict: drone_idx -> list of (x, y, z)
    paths = {}
    for frame in frames:
        drones = frame[f'{team}_drones']
        for idx, drone in enumerate(drones):
            if idx not in paths:
                paths[idx] = []
            paths[idx].append((drone['x'], drone['y'], drone['z']))
    return paths


def visualize_post_train(output_path='output.txt'):
    # Load replay data from output.txt
    with open(output_path, 'r') as f:
        data = json.load(f)
    replay = data['replay_data']
    frames = replay['frames']
    buildings = replay['buildings']
    world_size = replay['metadata']['world_size']

    # Extract drone paths
    blue_paths = extract_paths(frames, 'blue')
    red_paths = extract_paths(frames, 'red')

    # Gather starting points for blue and red drones (first frame only)
    blue_starts = [(d['x'], d['y'], d['z']) for d in frames[0]['blue_drones']]
    red_starts = [(d['x'], d['y'], d['z']) for d in frames[0]['red_drones']]

    # Gather all hit points for blue and red drones (across all frames)
    blue_hits = []
    red_hits = []
    for frame in frames:
        for d in frame['blue_drones']:
            if d.get('hit', False):
                blue_hits.append((d['x'], d['y'], d['z']))
        for d in frame['red_drones']:
            if d.get('hit', False):
                red_hits.append((d['x'], d['y'], d['z']))

    # Determine if blue killed all reds (all red drones are hit in the last frame)
    highlight_blue_ids = set()
    final_reds = frames[-1]['red_drones']
    if all(d.get('hit', False) for d in final_reds):
        # Find which blue drone(s) delivered the last hit(s)
        # We'll search backwards for the last frame(s) where a red drone's hit status changed to True
        red_hit_indices = [i for i, d in enumerate(final_reds)]
        hit_frames = {i: None for i in red_hit_indices}
        for t in reversed(range(len(frames))):
            for idx, d in enumerate(frames[t]['red_drones']):
                if d.get('hit', False) and hit_frames[idx] is None:
                    hit_frames[idx] = t
        # For each red drone, mark the blue drone(s) closest at that hit frame as the killer
        for idx, t in hit_frames.items():
            if t is not None:
                # Use the blue drone closest to the red drone at the hit frame
                red_pos = np.array([
                    frames[t]['red_drones'][idx]['x'],
                    frames[t]['red_drones'][idx]['y'],
                    frames[t]['red_drones'][idx]['z']
                ])
                blue_drones = frames[t]['blue_drones']
                blue_positions = np.array([[b['x'], b['y'], b['z']] for b in blue_drones])
                dists = np.linalg.norm(blue_positions - red_pos, axis=1)
                killer_idx = np.argmin(dists)
                highlight_blue_ids.add(killer_idx)

    # Plot
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # Plot buildings
    for building in buildings:
        plot_building(ax, building)

    # Plot drone paths and special markers
    plot_drone_paths(ax, blue_paths, color='blue', label_prefix='Blue', start_points=blue_starts, hit_points=blue_hits, highlight_ids=highlight_blue_ids if highlight_blue_ids else None)
    plot_drone_paths(ax, red_paths, color='red', label_prefix='Red', start_points=red_starts, hit_points=red_hits)

    # Set axis labels and limits
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(0, world_size['x'])
    ax.set_ylim(0, world_size['y'])
    ax.set_zlim(0, world_size['z'])
    ax.grid(True)
    plt.title('Drone Paths After Training')

    # Only show unique labels in legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), title="Teams")
    ax.view_init(elev=30, azim=45)
    plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Visualize all drone paths from output.txt after training')
    parser.add_argument('--output', type=str, default='output.txt', help='Path to output.txt containing replay data')
    args = parser.parse_args()
    visualize_post_train(args.output)
