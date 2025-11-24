# test_full_pipeline.py
"""
Comprehensive full pipeline test for all 4 configurations.
Tests complete training workflow: environment, agent creation, training, saving, evaluation.
"""
import numpy as np
import torch
import os
import tempfile
import shutil
from datetime import datetime
from tqdm import tqdm

from configs.config import Config
from maddpg import MADDPG, ReplayBuffer
from utils.env import get_env_info, create_single_env
from utils.logger import Logger
from utils.utils import evaluate

def test_configuration(actor_type, critic_type, config_name, num_steps=500):
    """
    Test a complete training pipeline for a specific configuration.
    
    Args:
        actor_type: "traditional" or "rotational equivariant"
        critic_type: "traditional" or "permutation invariant"
        config_name: Human-readable name for this configuration
        num_steps: Number of training steps to run
    
    Returns:
        dict: Test results and metrics
    """
    print("\n" + "="*70)
    print(f"Testing Configuration: {config_name}")
    print(f"Actor: {actor_type}, Critic: {critic_type}")
    print("="*70)
    
    results = {
        'config_name': config_name,
        'actor_type': actor_type,
        'critic_type': critic_type,
        'passed': False,
        'errors': []
    }
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Setup config
        cfg = Config()
        cfg.total_timesteps = num_steps
        cfg.warmup_steps = 50
        cfg.update_every = 10
        cfg.eval_interval = 100
        cfg.batch_size = 32
        cfg.actor = actor_type
        cfg.critic = critic_type
        
        # Get environment info
        agents, num_agents, action_sizes, action_low, action_high, state_sizes = get_env_info(
            env_name=cfg.env_name,
            max_steps=cfg.max_steps,
            apply_padding=False
        )
        
        print(f"✓ Environment info retrieved")
        print(f"  Agents: {num_agents}, State sizes: {state_sizes}, Action sizes: {action_sizes}")
        
        # Create environments
        env = create_single_env(
            env_name=cfg.env_name,
            max_steps=cfg.max_steps,
            render_mode=None,
            apply_padding=False
        )
        
        env_eval = create_single_env(
            env_name=cfg.env_name,
            max_steps=cfg.max_steps,
            render_mode=None,
            apply_padding=False
        )
        
        print(f"✓ Environments created")
        
        # Create MADDPG agent
        hidden_sizes = tuple(map(int, cfg.hidden_sizes.split(',')))
        maddpg = MADDPG(
            state_sizes=state_sizes,
            action_sizes=action_sizes,
            hidden_sizes=hidden_sizes,
            actor_lr=cfg.actor_lr,
            critic_lr=cfg.critic_lr,
            gamma=cfg.gamma,
            tau=cfg.tau,
            action_low=action_low,
            action_high=action_high,
            actor=actor_type,
            critic=critic_type
        )
        
        print(f"✓ MADDPG agent created")
        
        # Create replay buffer
        buffer = ReplayBuffer(
            buffer_size=5000,
            batch_size=cfg.batch_size,
            agents=agents,
            state_sizes=state_sizes,
            action_sizes=action_sizes
        )
        
        print(f"✓ Replay buffer created")
        
        # Create logger
        logger = Logger(
            run_name=f"test_{config_name.replace(' ', '_').lower()}",
            folder=temp_dir,
            algo=cfg.algo,
            env=cfg.env_name
        )
        
        print(f"✓ Logger created")
        
        # Initial evaluation
        print(f"\n  Running initial evaluation...")
        initial_rewards = evaluate(
            env_eval, maddpg, logger, 
            record_gif=False, 
            num_eval_episodes=5, 
            global_step=0
        )
        initial_total = np.sum(initial_rewards)
        print(f"  Initial total reward: {initial_total:.2f}")
        
        # Training loop
        print(f"\n  Training for {num_steps} steps...")
        observations, _ = env.reset()
        episode_rewards = np.zeros(len(agents))
        noise_scale = cfg.noise_scale
        noise_decay = (cfg.noise_scale - cfg.min_noise) / max(1, cfg.noise_decay_steps)
        
        training_metrics = {
            'steps_completed': 0,
            'learning_updates': 0,
            'episodes_completed': 0,
            'critic_losses': [],
            'actor_losses': [],
            'episode_rewards': []
        }
        
        for step in range(1, num_steps + 1):
            # Get actions
            states_list = [np.array(observations[agent], dtype=np.float32) for agent in agents]
            actions_list = maddpg.act(states_list, add_noise=True, noise_scale=noise_scale)
            actions = {agent: action for agent, action in zip(agents, actions_list)}
            
            # Step environment
            next_observations, rewards, terminations, truncations, _ = env.step(actions)
            dones = [terminations[agent] or truncations[agent] for agent in agents]
            done = any(dones)
            
            # Store in buffer
            rewards_array = np.array([rewards[agent] for agent in agents], dtype=np.float32)
            next_states_list = [np.array(next_observations[agent], dtype=np.float32) for agent in agents]
            terminations_array = np.array([terminations[agent] for agent in agents], dtype=np.uint8)
            
            buffer.add(states_list, actions_list, rewards_array, next_states_list, terminations_array)
            
            observations = next_observations
            episode_rewards += np.array(list(rewards.values()))
            
            # Learn
            if step > cfg.warmup_steps and step % cfg.update_every == 0 and len(buffer) >= cfg.batch_size:
                for i in range(len(agents)):
                    experiences = buffer.sample()
                    critic_loss, actor_loss = maddpg.learn(experiences, i)
                    
                    training_metrics['critic_losses'].append(critic_loss)
                    training_metrics['actor_losses'].append(actor_loss)
                    training_metrics['learning_updates'] += 1
                    
                    logger.add_scalar(f'{agents[i]}/critic_loss', critic_loss, step)
                    logger.add_scalar(f'{agents[i]}/actor_loss', actor_loss, step)
                
                maddpg.update_targets()
            
            # Update noise
            if cfg.use_noise_decay and step > cfg.warmup_steps:
                noise_scale = max(cfg.min_noise, noise_scale - noise_decay)
            
            # Reset episode
            if done or (step % cfg.max_steps == 0):
                training_metrics['episodes_completed'] += 1
                training_metrics['episode_rewards'].append(np.sum(episode_rewards))
                
                for i, reward in enumerate(episode_rewards):
                    logger.add_scalar(f"{agents[i]}/episode_reward", reward, step)
                logger.add_scalar('train/total_reward', np.sum(episode_rewards), step)
                logger.add_scalar('noise/scale', noise_scale, step)
                
                observations, _ = env.reset()
                episode_rewards = np.zeros(len(agents))
            
            training_metrics['steps_completed'] = step
            
            # Evaluate periodically
            if step % cfg.eval_interval == 0:
                eval_rewards = evaluate(
                    env_eval, maddpg, logger, 
                    record_gif=False, 
                    num_eval_episodes=3, 
                    global_step=step
                )
                print(f"    Step {step}: Eval reward = {np.sum(eval_rewards):.2f}")
        
        # Final evaluation
        print(f"\n  Running final evaluation...")
        final_rewards = evaluate(
            env_eval, maddpg, logger, 
            record_gif=False, 
            num_eval_episodes=10, 
            global_step=num_steps
        )
        final_total = np.sum(final_rewards)
        print(f"  Final total reward: {final_total:.2f}")
        
        # Test model saving
        model_path = os.path.join(temp_dir, f"model_{config_name.replace(' ', '_').lower()}.pt")
        maddpg.save(model_path)
        assert os.path.exists(model_path), "Model file should exist"
        print(f"✓ Model saved successfully ({os.path.getsize(model_path)} bytes)")
        
        # Calculate metrics
        results['initial_reward'] = initial_total
        results['final_reward'] = final_total
        results['reward_improvement'] = final_total - initial_total
        results['avg_critic_loss'] = np.mean(training_metrics['critic_losses']) if training_metrics['critic_losses'] else 0
        results['avg_actor_loss'] = np.mean(training_metrics['actor_losses']) if training_metrics['actor_losses'] else 0
        results['episodes_completed'] = training_metrics['episodes_completed']
        results['learning_updates'] = training_metrics['learning_updates']
        results['steps_completed'] = training_metrics['steps_completed']
        results['model_size'] = os.path.getsize(model_path)
        
        # Cleanup
        env.close()
        env_eval.close()
        logger.close()
        
        results['passed'] = True
        print(f"\n✓ Configuration '{config_name}' PASSED all tests")
        
    except Exception as e:
        results['passed'] = False
        results['errors'].append(str(e))
        print(f"\n❌ Configuration '{config_name}' FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup temp directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    
    return results

def main():
    """Run comprehensive pipeline tests for all configurations"""
    print("\n" + "="*70)
    print("COMPREHENSIVE FULL PIPELINE TEST")
    print("Testing all 4 configurations with complete training workflow")
    print("="*70)
    
    configurations = [
        ("traditional", "traditional", "Baseline"),
        ("rotational equivariant", "traditional", "Rotational Equivariant Actor"),
        ("traditional", "permutation invariant", "Permutation Invariant Critic"),
        ("rotational equivariant", "permutation invariant", "Full Symmetric"),
    ]
    
    all_results = []
    
    for actor_type, critic_type, config_name in configurations:
        results = test_configuration(
            actor_type=actor_type,
            critic_type=critic_type,
            config_name=config_name,
            num_steps=500  # Short training run for testing
        )
        all_results.append(results)
    
    # Print summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    
    print(f"\n{'Configuration':<40} {'Status':<10} {'Initial':<10} {'Final':<10} {'Improvement':<12} {'Updates':<8}")
    print("-" * 100)
    
    for r in all_results:
        status = "✓ PASSED" if r['passed'] else "❌ FAILED"
        initial = f"{r.get('initial_reward', 0):.1f}" if r['passed'] else "N/A"
        final = f"{r.get('final_reward', 0):.1f}" if r['passed'] else "N/A"
        improvement = f"{r.get('reward_improvement', 0):.1f}" if r['passed'] else "N/A"
        updates = f"{r.get('learning_updates', 0)}" if r['passed'] else "N/A"
        
        print(f"{r['config_name']:<40} {status:<10} {initial:<10} {final:<10} {improvement:<12} {updates:<8}")
    
    # Detailed metrics
    print("\n" + "="*70)
    print("DETAILED METRICS")
    print("="*70)
    
    for r in all_results:
        if r['passed']:
            print(f"\n{r['config_name']}:")
            print(f"  Steps completed: {r.get('steps_completed', 0)}")
            print(f"  Episodes completed: {r.get('episodes_completed', 0)}")
            print(f"  Learning updates: {r.get('learning_updates', 0)}")
            print(f"  Avg critic loss: {r.get('avg_critic_loss', 0):.6f}")
            print(f"  Avg actor loss: {r.get('avg_actor_loss', 0):.6f}")
            print(f"  Initial reward: {r.get('initial_reward', 0):.2f}")
            print(f"  Final reward: {r.get('final_reward', 0):.2f}")
            print(f"  Reward improvement: {r.get('reward_improvement', 0):.2f}")
            print(f"  Model size: {r.get('model_size', 0)} bytes")
        else:
            print(f"\n{r['config_name']}: FAILED")
            for error in r.get('errors', []):
                print(f"  Error: {error}")
    
    # Overall status
    all_passed = all(r['passed'] for r in all_results)
    
    print("\n" + "="*70)
    if all_passed:
        print("✅ ALL CONFIGURATIONS PASSED!")
        print("The entire pipeline is ready for full training runs.")
    else:
        print("⚠ SOME CONFIGURATIONS FAILED")
        print("Please fix issues before proceeding with full training.")
    print("="*70)
    
    return all_passed

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

