import gymnasium as gym
import numpy as np
from drone_combat_env import DroneCombatEnv
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import os

"""
example: 
python3 test_drone_rl.py --timesteps 10 --mode test
python test_drone_rl.py --mode train --timesteps 50000 --model-path drone_combat_model --record-replay
python3 test_drone_rl.py --mode visualize --model-path drone_combat_model   
python3 test_drone_rl.py --mode train --timesteps 20000 --model-path drone_combat_model
python3 test_drone_rl.py --mode evaluate --model-path drone_combat_model --episodes 1
"""


def test_environment_manually():
    """Test the environment with random actions"""
    # Create environment with replay recording if specified
    env = DroneCombatEnv(record_replay=args.record_replay, replay_path=args.replay_path)
    
    # Check if environment follows Gym API
    check_env(env)
    print("Environment check passed!")
    
    # Reset environment
    obs, info = env.reset()
    
    # Run a few random steps
    total_reward = 0
    for i in range(20):
        # Sample random action
        action = env.action_space.sample()
        
        # Take step in environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        print(f"Step {i+1}")
        print(f"  Action: {action}")
        print(f"  Observation: {obs}")
        print(f"  Reward: {reward}")
        print(f"  Info: {info}")
        
        if terminated or truncated:
            print("Episode finished early!")
            break
    
    print(f"Total reward: {total_reward}")
    
    # Close environment
    env.close()

def visualize_episode(env, model=None, max_steps=100, step_delay=0.5):
    """Visualize a single episode, stepping through each action and showing drone movements over time"""
    import time
    import matplotlib.pyplot as plt
    from IPython.display import clear_output
    
    # Note: If record_replay is True, the environment will automatically record replay data
    
    # Reset environment
    obs, info = env.reset()
    
    # Run episode
    done = False
    total_reward = 0
    step = 0
    
    print("Starting visualization - press Ctrl+C to stop")
    
    try:
        while not done and step < max_steps:
            # Display current state
            print(f"\nStep {step + 1}/{max_steps}")
            print(f"Red drone: ({env.red_drone.x:.2f}, {env.red_drone.y:.2f}, {env.red_drone.z:.2f})")
            print(f"Blue drone: ({env.blue_drone.x:.2f}, {env.blue_drone.y:.2f}, {env.blue_drone.z:.2f})")
            print(f"Current reward: {total_reward:.2f}")
            
            # Render current state
            env.render()
            
            # Either use model or random actions
            if model:
                action, _ = model.predict(obs, deterministic=True)
                print(f"Model action: {action}")
            else:
                action = env.action_space.sample()
                print(f"Random action: {action}")
            
            # Explain the action
            dx, dy, dz, shoot = action
            print(f"Movement: dx={dx:.2f}, dy={dy:.2f}, dz={dz:.2f}")
            print(f"Shooting: {'Yes' if shoot > 0.5 else 'No'}")
            
            # Take step
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step += 1
            
            # Show step results
            print(f"Step reward: {reward:.2f}")
            if info.get('red_hit'):
                print("Red drone was hit!")
            if info.get('blue_hit'):
                print("Blue drone was hit!")
            
            # Check if done
            done = terminated or truncated
            if done:
                print("\nEpisode terminated!")
                if terminated:
                    if info.get('red_hit'):
                        print("Red drone was destroyed")
                    if info.get('blue_hit'):
                        print("Blue drone was destroyed")
                if truncated:
                    print("Maximum steps reached")
            
            # Pause between steps
            time.sleep(step_delay)
    
    except KeyboardInterrupt:
        print("\nVisualization stopped by user")
    
    print(f"\nEpisode finished after {step} steps with total reward {total_reward:.2f}")
    
    # Render final state
    env.render()

def train_simple_agent(total_timesteps=10000,
                       save_path="drone_agent_model",
                       record_replay=False,
                       replay_path=None,
                       max_steps=10000):
    """Train a simple PPO agent on the environment"""
    # Create environment with replay recording if specified
    env = DroneCombatEnv(
        record_replay=record_replay, 
        replay_path=replay_path, 
        max_steps=max_steps, 
    )
    
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
    
    # Explicitly save replay if recording was enabled
    if record_replay and hasattr(env, 'save_replay'):
        try:
            actual_path = env.save_replay(replay_path)
            print(f"Replay explicitly saved to {actual_path}")
        except Exception as e:
            print(f"Error explicitly saving replay: {e}")
    
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
    print(f"Model saved to: {save_path}")
    print("="*50)
    
    return model

def evaluate_agent(
    model_path="drone_agent_model", 
    num_episodes=1000, 
    record_replay=True, 
    replay_path="eval_replay.json"
):
    """Evaluate a trained agent over multiple episodes"""
    # Create environment with replay recording if specified
    env = DroneCombatEnv(
        record_replay=record_replay,
        replay_path=replay_path,
        max_steps=num_episodes,
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

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_steps += 1
            done = terminated or truncated

        rewards.append(episode_reward)
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
    parser.add_argument("--model-path", type=str, default="drone_agent_model",
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
        
        train_simple_agent(
            total_timesteps=args.timesteps, 
            save_path=args.model_path,
            record_replay=args.record_replay,
            replay_path=args.replay_path,
            max_steps=max_steps
        )
    
    elif args.mode == "visualize":
        print("Visualizing episode...")
        if args.record_replay:
            replay_path = args.replay_path if args.replay_path else 'replay.json'
            print(f"Recording replay data to {replay_path}")
        env = DroneCombatEnv(
            render_mode="human",
            record_replay=args.record_replay,
            replay_path=args.replay_path
        )
        
        # Parse additional visualization parameters
        max_steps = args.max_steps if hasattr(args, 'max_steps') else 100
        step_delay = args.delay if hasattr(args, 'delay') else 0.5
        
        # Check if model exists
        if os.path.exists(f"{args.model_path}.zip"):
            print(f"Using trained model from {args.model_path}")
            model = PPO.load(args.model_path)
            visualize_episode(env, model, max_steps=max_steps, step_delay=step_delay)
        else:
            print("No model found, using random actions")
            visualize_episode(env, max_steps=max_steps, step_delay=step_delay)
    
    elif args.mode == "evaluate":
        print(f"Evaluating agent over {args.episodes} episodes...")
        if args.record_replay:
            print(f"Recording replay data to {args.replay_path if args.replay_path else 'replay.json'}")
        evaluate_agent(
            model_path=args.model_path, 
            num_episodes=args.episodes,
            record_replay=args.record_replay,
            replay_path=args.replay_path
        )
