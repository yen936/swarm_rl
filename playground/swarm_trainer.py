from drone_combat_env import DroneCombatEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import os
from datetime import datetime
"""
example: 
python3 swarm_trainer.py --timesteps 10 --mode test
python swarm_trainer.py --mode train --timesteps 50000 --model-path drone_combat_model --record-replay
python3 swarm_trainer.py --mode visualize --model-path drone_combat_model   
python3 swarm_trainer.py --mode train --timesteps 20000 --model-path drone_combat_model
python3 swarm_trainer.py --mode evaluate --model-path drone_combat_model --episodes 1
"""

def train_agent(total_timesteps=10000,
                       save_path="drone_agent_model",
                       record_replay=True,  
                       replay_path="replay.json",
                       max_steps=10000,
                       num_blue_drones=1,
                       num_red_drones=1,
                       base_model_path="dronebase_model"):
    """
    Train a simple PPO agent on the drone combat environment
    
    Args:
        total_timesteps: Number of timesteps to train for
        save_path: Path to save the trained model
        record_replay: Whether to record replay data
        replay_path: Path to save replay data
        max_steps: Maximum steps per episode
        num_blue_drones: Number of blue drones (controlled by the agent)
        num_red_drones: Number of red drones (opponents)
    
    Returns:
        tuple: (trained PPO model, replay data as dict or None)
    """

    # Create environment with replay recording if specified
    env = DroneCombatEnv(
        record_replay=record_replay, 
        replay_path=replay_path, 
        max_steps=max_steps,
        num_blue_drones=num_blue_drones,
        num_red_drones=num_red_drones
    )

    if base_model_path is not None and os.path.exists(base_model_path):
        print(f"Loading base PPO model from {base_model_path} for fine-tuning...")
        model = PPO.load(base_model_path, env=env)
        print("Base model loaded. Continuing training.")
    else:
        # Create PPO agent
        model = PPO(
            policy="MlpPolicy",
            env=env, 
            learning_rate=0.0003,
            verbose=1
        )
    
    # Train agent
    model.learn(total_timesteps=total_timesteps)
    
    # Save model
    model.save(save_path)
    print(f"Model saved to {save_path}")
    
    # Get replay data if recording was enabled
    replay_data = None
    if record_replay and hasattr(env, 'save_replay'):
        try:
            # Save the replay to file
            actual_path = env.save_replay(replay_path)
            print(f"Replay explicitly saved to {actual_path}")
            
            # Also get the replay data for the API response
            import json
            if os.path.exists(actual_path):
                with open(actual_path, 'r') as f:
                    replay_data = json.load(f)
        except Exception as e:
            print(f"Error handling replay data: {e}")
    
    # Close the environment to ensure resources are released
    env.close()

    # Print the model summary and episode statistics
    print("\n" + "="*50)
    print("MODEL TRAINING SUMMARY")
    print("="*50)
    
    # Extract episode statistics from the model's logger
    if hasattr(model, 'logger') and hasattr(model.logger, 'name_to_value'):
        stats = model.logger.name_to_value
        
        # Print episode rewards
        if 'rollout/ep_rew_mean' in stats:
            print(f"Mean Episode Reward: {stats['rollout/ep_rew_mean']:.4f}")
        
        # Print episode lengths
        if 'rollout/ep_len_mean' in stats:
            print(f"Mean Episode Length: {stats['rollout/ep_len_mean']:.1f} steps")
        
        # Print value and policy loss
        if 'train/value_loss' in stats:
            print(f"Value Loss: {stats['train/value_loss']:.6f}")
        if 'train/policy_loss' in stats:
            print(f"Policy Loss: {stats['train/policy_loss']:.6f}")
        
        # Print other useful metrics    
        if 'train/entropy_loss' in stats:
            print(f"Entropy Loss: {stats['train/entropy_loss']:.6f}")
        if 'train/learning_rate' in stats:
            print(f"Learning Rate: {stats['train/learning_rate']:.6f}")
        if 'train/n_updates' in stats:
            print(f"Number of Updates: {stats['train/n_updates']}")
        if 'time/fps' in stats:
            print(f"Training Speed: {stats['time/fps']:.1f} FPS")
    else:
        print("No training statistics available")
    
    print("="*50)
    if save_path == "dronebase_model.zip":
        save_path = "drone_ft_model.zip"
    print(f"Model saved to: {save_path}")
    print("="*50)
    
    return model, replay_data

def train_agent_with_all_replays(total_timesteps=10000,
                       save_path="drone_agent_model",
                       record_replay=True,
                       max_steps=1000,
                       num_blue_drones=1,
                       num_red_drones=1,
                       base_model_path="dronebase_model"):
    """
    Train a PPO agent on the drone combat environment and save replays for all episodes
    
    Args:
        total_timesteps: Number of timesteps to train for
        save_path: Path to save the trained model
        record_replay: Whether to record replay data
        replay_dir: Directory to save replay data
        max_steps: Maximum steps per episode
        num_blue_drones: Number of blue drones (controlled by the agent)
        num_red_drones: Number of red drones (opponents)
        base_model_path: Path to the base model for fine-tuning
        
    Returns:
        tuple: (trained PPO model, all replay data as dict)
    """
    # No need to create directory for replays as we're keeping them in memory

    # Create environment with replay recording
    env = DroneCombatEnv(
        record_replay=record_replay,
        max_steps=max_steps,
        num_blue_drones=num_blue_drones,
        num_red_drones=num_red_drones
    )

    # Create or load PPO agent
    if base_model_path is not None and os.path.exists(base_model_path):
        print(f"Loading base PPO model from {base_model_path} for fine-tuning...")
        model = PPO.load(base_model_path, env=env)
        print("Base model loaded. Continuing training.")
    else:
        model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=0.0003,
            verbose=1
        )

    # Initialize variables for tracking episodes
    all_replays = {}
    episode_count = 0
    timesteps_so_far = 0

    # Calculate episodes needed based on max_steps and total_timesteps
    estimated_episodes = total_timesteps // max_steps + 1
    print(f"Training for approximately {estimated_episodes} episodes")

    # Train in batches to collect replays after each episode
    while timesteps_so_far < total_timesteps:
        # Calculate remaining timesteps
        remaining = total_timesteps - timesteps_so_far
        # Train for one episode at a time to collect replays
        batch_size = min(max_steps, remaining)

        # Train for one episode
        model.learn(total_timesteps=batch_size, reset_num_timesteps=False)
        timesteps_so_far += batch_size
        episode_count += 1

        # Capture replay data if recording was enabled
        if record_replay and hasattr(env, 'replay_data') and env.replay_data:
            try:
                # Create a deep copy of the replay data
                episode_key = f"episode_{episode_count}"

                # Create metadata for this episode
                metadata = {
                    "timestamp": datetime.now().isoformat(),
                    "episode": episode_count,
                    "timesteps": batch_size,
                    "total_timesteps_so_far": timesteps_so_far
                }

                # Create a replay structure similar to what save_replay would create
                replay_data = {
                    "metadata": {
                        "version": "1.0",
                        "date": datetime.now().isoformat(),
                        "total_steps": len(env.replay_data),
                        "world_size": {
                            "x": float(env.world.size_x),
                            "y": float(env.world.size_y),
                            "z": float(env.world.size_z)
                        },
                        "episode_info": metadata
                    },
                    "buildings": [
                        {
                            "x": float(b.x),
                            "y": float(b.y),
                            "width": float(b.width),
                            "depth": float(b.depth),
                            "height": float(b.height)
                        } for b in env.world.buildings
                    ],
                    "frames": env.replay_data
                }

                # Add to collection of all replays
                all_replays[episode_key] = replay_data
                print(f"Captured replay data for episode {episode_count} with {len(env.replay_data)} frames")

                # Reset the environment to clear replay data for next episode
                env.reset()

            except Exception as e:
                print(f"Error capturing replay for episode {episode_count}: {e}")

        # Print progress
        print(f"Episode {episode_count} completed. Total timesteps: {timesteps_so_far}/{total_timesteps}")

    # Save final model
    if save_path == "dronebase_model.zip":
        save_path = "drone_ft_model.zip"
    model.save(save_path)
    print(f"Final model saved to {save_path}")

    # Keep all replays in memory - no need to save to disk
    print(f"Collected {episode_count} episodes of replay data in memory")

    # Close the environment
    env.close()

    # Print training summary
    print("\n" + "="*50)
    print("MODEL TRAINING SUMMARY")
    print("="*50)
    print(f"Total episodes: {episode_count}")
    print(f"Total timesteps: {timesteps_so_far}")
    print(f"Final model saved to: {save_path}")
    print(f"All {len(all_replays)} episodes collected in memory")
    print("="*50)

    return model, all_replays


def evaluate_agent(
    model_path="dronebase_model", 
    num_episodes=1000, 
    record_replay=True, 
    replay_path="eval_replay.json",
    num_red_drones=1,
    num_blue_drones=1,
    replay_all_path="replay_all.json"
):
    """Evaluate a trained agent over multiple episodes using the dronebase_model by default"""
    # Create environment with replay recording if specified
    env = DroneCombatEnv(
        record_replay=record_replay,
        replay_path=replay_path,
        max_steps=num_episodes,
        num_red_drones=num_red_drones,
        num_blue_drones=num_blue_drones
    )

    # Load model
    model = PPO.load(model_path)

    # Run evaluation
    rewards = []
    steps = []

    for i in range(num_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        episode_steps = 0
        blue_killed_all = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_steps += 1
            done = terminated or truncated

        # Check if blue killed all reds (all red drones are hit at the end)
        if hasattr(env, 'replay_data') and 'red_drones' in env.replay_data[-1]:
            final_reds = env.replay_data[-1]['red_drones']
            if all(d.get('hit', False) for d in final_reds):
                blue_killed_all = True

        rewards.append(episode_reward)

        if blue_killed_all:
            # Save this replay and break
            if record_replay and hasattr(env, 'save_replay'):
                try:
                    actual_path = env.save_replay(replay_all_path)
                    print(f"Blue killed all reds in episode {i+1}! Replay saved to {actual_path}")
                except Exception as e:
                    print(f"Error saving replay: {e}")
            break
        steps.append(episode_steps)
        print(f"Episode {i+1}: Reward = {episode_reward}, Steps = {episode_steps}")
        
    # Explicitly save replay if recording was enabled
    if record_replay and hasattr(env, 'save_replay') and env.blue_hit:
        try:
            actual_path = env.save_replay(replay_path)
            print(f"Replay explicitly saved to {actual_path}")
            
            # Also save a copy to replay1.json if the blue drone successfully killed the red drone
            if any(frame.get('red_hit', False) for frame in env.replay_data):
                success_path = "replay1.json"
                env.save_replay(success_path)
                print(f"Success replay saved to {success_path}")
        except Exception as e:
            print(f"Error saving replay: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test and train drone combat RL environment")
    parser.add_argument("--mode", type=str, default="test", 
                        choices=["test", "train", "visualize", "evaluate"],
                        help="Mode to run (test, train, visualize, evaluate)")
    parser.add_argument("--timesteps", type=int, default=10000,
                        help="Number of timesteps to train for")
    parser.add_argument("--model-path", type=str, default="dronebase_model",
                        help="Path to save/load model")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of episodes for evaluation")
    parser.add_argument("--max-steps", type=int, default=100,
                        help="Maximum number of steps per episode for visualization")
    parser.add_argument("--delay", type=float, default=0.5,
                        help="Delay between steps during visualization (seconds)")
    parser.add_argument("--record-replay", action="store_true",
                        help="Record replay data during execution")
    parser.add_argument("--replay-path", type=str, default=None,
                        help="Path to save replay data (default: replay.json)")
    parser.add_argument("--num-red-drones", type=int, default=1,
                        help="Number of red drones (opponents)", choices=range(1, 6))
    parser.add_argument("--num-blue-drones", type=int, default=1,
                        help="Number of blue drones (controlled by the agent)", choices=range(1, 6))
    
    args = parser.parse_args()
    
    if args.mode == "test":
        print("Testing environment with random actions...")
        test_environment_manually()
    
    elif args.mode == "train":
        print(f"Training agent for {args.timesteps} timesteps...")
        if args.record_replay:
            replay_path = args.replay_path if args.replay_path else 'replay.json'
            print(f"Recording replay data to {replay_path}")
        
        # Set max_steps to be at least as large as timesteps to avoid early truncation
        max_steps = max(args.timesteps // 10, 1000)  # Reasonable default based on timesteps
        print(f"Setting max steps per episode to {max_steps}")
        
        train_agent(
            total_timesteps=args.timesteps, 
            save_path=args.model_path,
            record_replay=args.record_replay,
            replay_path=args.replay_path,
            max_steps=max_steps,
            num_red_drones=args.num_red_drones,
            num_blue_drones=args.num_blue_drones
        )
    
    
    elif args.mode == "evaluate":
        print(f"Evaluating agent over {args.episodes} episodes...")
        if args.record_replay:
            print(f"Recording replay data to {args.replay_path if args.replay_path else 'replay.json'}")
        print(f"Using {args.num_blue_drones} blue drones and {args.num_red_drones} red drones")
        evaluate_agent(
            num_episodes=args.episodes,
            num_red_drones=args.num_red_drones,
            num_blue_drones=args.num_blue_drones
        )
