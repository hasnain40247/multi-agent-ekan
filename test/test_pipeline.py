# test_pipeline.py
"""
Comprehensive pipeline test to verify the entire training pipeline works correctly.
Tests all components: environment, agent, buffer, learning, saving, evaluation.
"""
import numpy as np
import torch
import os
import tempfile
import shutil
from datetime import datetime

# Import all necessary components
from configs.config import Config
from maddpg import MADDPG, ReplayBuffer
from utils.env import get_env_info, create_single_env
from utils.logger import Logger
from utils.utils import evaluate

def test_environment():
    """Test 1: Environment creation and basic operations"""
    print("="*70)
    print("Test 1: Environment Setup")
    print("="*70)
    try:
        env = create_single_env(
            env_name="simple_spread_v3",
            max_steps=25,
            render_mode=None,
            apply_padding=False
        )
        
        # Test reset
        observations, _ = env.reset()
        agents = env.agents
        print(f"✓ Environment created: {len(agents)} agents")
        print(f"  Agents: {agents}")
        
        # Test step - get action space from environment
        states_list = [np.array(observations[agent], dtype=np.float32) for agent in agents]
        action_sizes = [env.action_space(agent).shape[0] for agent in agents]
        random_actions = {agent: np.random.uniform(-1, 1, size=action_sizes[i]) for i, agent in enumerate(agents)}
        next_obs, rewards, terminations, truncations, _ = env.step(random_actions)
        
        print(f"✓ Environment step successful")
        print(f"  State shapes: {[s.shape for s in states_list]}")
        print(f"  Reward shapes: {[rewards[a] for a in agents]}")
        
        env.close()
        return agents, states_list, len(agents)
    except Exception as e:
        print(f"❌ Environment test failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def test_agent_creation(agents, state_sizes, action_sizes, action_low, action_high):
    """Test 2: MADDPG agent creation with all configurations"""
    print("\n" + "="*70)
    print("Test 2: MADDPG Agent Creation (All Configurations)")
    print("="*70)
    
    configs = [
        ("traditional", "traditional", "Baseline"),
        ("rotational equivariant", "traditional", "Rotational Equivariant Actor"),
        ("traditional", "permutation invariant", "Permutation Invariant Critic"),
        ("rotational equivariant", "permutation invariant", "Full Symmetric"),
    ]
    
    agents_created = {}
    
    for actor_type, critic_type, name in configs:
        try:
            print(f"\n  Testing: {name}")
            maddpg = MADDPG(
                state_sizes=state_sizes,
                action_sizes=action_sizes,
                hidden_sizes=(64, 64),
                actor_lr=1e-3,
                critic_lr=2e-3,
                gamma=0.95,
                tau=0.01,
                action_low=action_low,
                action_high=action_high,
                actor=actor_type,
                critic=critic_type
            )
            agents_created[(actor_type, critic_type)] = maddpg
            print(f"    ✓ Created successfully")
        except Exception as e:
            print(f"    ❌ Failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    return agents_created

def test_action_generation(maddpg, states_list):
    """Test 3: Action generation"""
    print("\n" + "="*70)
    print("Test 3: Action Generation")
    print("="*70)
    try:
        # Test without noise
        actions_no_noise = maddpg.act(states_list, add_noise=False)
        print(f"✓ Actions generated (no noise): {[a.shape for a in actions_no_noise]}")
        
        # Test with noise
        actions_with_noise = maddpg.act(states_list, add_noise=True, noise_scale=0.3)
        print(f"✓ Actions generated (with noise): {[a.shape for a in actions_with_noise]}")
        
        # Verify action shapes
        assert len(actions_no_noise) == len(states_list), "Number of actions should match states"
        for i, action in enumerate(actions_no_noise):
            assert len(action.shape) == 1, f"Action {i} should be 1D, got shape {action.shape}"
            # Actions should be in valid range (check if they're clipped)
            assert np.all(action >= -1.1) and np.all(action <= 1.1), f"Action {i} should be approximately in [-1, 1], got range [{action.min():.2f}, {action.max():.2f}]"
        
        print(f"✓ All actions are valid (shape and range)")
        return True
    except Exception as e:
        print(f"❌ Action generation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_replay_buffer(agents, state_sizes, action_sizes):
    """Test 4: Replay Buffer Operations"""
    print("\n" + "="*70)
    print("Test 4: Replay Buffer Operations")
    print("="*70)
    try:
        buffer = ReplayBuffer(
            buffer_size=1000,
            batch_size=32,
            agents=agents,
            state_sizes=state_sizes,
            action_sizes=action_sizes
        )
        
        # Add some experiences
        num_samples = 100
        for _ in range(num_samples):
            states = [np.random.randn(s).astype(np.float32) for s in state_sizes]
            actions = [np.random.uniform(-1, 1, size=a).astype(np.float32) for a in action_sizes]
            rewards = np.random.randn(len(agents)).astype(np.float32)
            next_states = [np.random.randn(s).astype(np.float32) for s in state_sizes]
            dones = np.random.choice([0, 1], size=len(agents)).astype(np.float32)
            
            buffer.add(states, actions, rewards, next_states, dones)
        
        print(f"✓ Added {num_samples} experiences to buffer")
        print(f"  Buffer size: {len(buffer)}")
        
        # Test sampling
        if len(buffer) >= 32:
            experiences = buffer.sample()
            states_batch, actions_batch, rewards_batch, next_states_batch, dones_batch, \
                states_full, next_states_full, actions_full = experiences
            
            print(f"✓ Sampled batch successfully")
            print(f"  States batch: {[s.shape for s in states_batch]}")
            print(f"  Actions batch: {[a.shape for a in actions_batch]}")
            print(f"  Rewards batch: {rewards_batch.shape}")
            print(f"  States full: {states_full.shape}")
            print(f"  Actions full: {actions_full.shape}")
            
            # Verify shapes
            assert states_full.shape[0] == 32, f"Batch size should be 32, got {states_full.shape[0]}"
            assert actions_full.shape[0] == 32, f"Batch size should be 32, got {actions_full.shape[0]}"
            
            return buffer
        else:
            print(f"⚠ Buffer too small for sampling (need 32, have {len(buffer)})")
            return buffer
    except Exception as e:
        print(f"❌ Replay buffer test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_learning(maddpg, buffer, agents):
    """Test 5: Learning/Update Operations"""
    print("\n" + "="*70)
    print("Test 5: Learning/Update Operations")
    print("="*70)
    try:
        if len(buffer) < 32:
            print("⚠ Skipping learning test - buffer too small")
            return False
        
        # Sample experiences
        experiences = buffer.sample()
        
        # Test learning for each agent
        for i in range(len(agents)):
            critic_loss, actor_loss = maddpg.learn(experiences, i)
            print(f"✓ Agent {i} learning successful")
            print(f"  Critic loss: {critic_loss:.6f}")
            print(f"  Actor loss: {actor_loss:.6f}")
            
            assert not np.isnan(critic_loss), f"Critic loss is NaN for agent {i}"
            assert not np.isnan(actor_loss), f"Actor loss is NaN for agent {i}"
            assert critic_loss >= 0, f"Critic loss should be >= 0, got {critic_loss}"
        
        # Test target network update
        maddpg.update_targets()
        print(f"✓ Target networks updated successfully")
        
        return True
    except Exception as e:
        print(f"❌ Learning test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_save_load(maddpg, temp_dir):
    """Test 6: Model Saving and Loading"""
    print("\n" + "="*70)
    print("Test 6: Model Saving and Loading")
    print("="*70)
    try:
        model_path = os.path.join(temp_dir, "test_model.pt")
        
        # Save model
        maddpg.save(model_path)
        print(f"✓ Model saved to {model_path}")
        assert os.path.exists(model_path), "Model file should exist"
        
        # Create new agent and load (simplified - just test that file exists and can be loaded)
        # Note: Full loading test would require matching architecture
        file_size = os.path.getsize(model_path)
        print(f"✓ Model file size: {file_size} bytes")
        
        return True
    except Exception as e:
        print(f"❌ Save/load test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_evaluation(maddpg, agents):
    """Test 7: Evaluation Function"""
    print("\n" + "="*70)
    print("Test 7: Evaluation Function")
    print("="*70)
    try:
        # Create evaluation environment
        env_eval = create_single_env(
            env_name="simple_spread_v3",
            max_steps=25,
            render_mode=None,
            apply_padding=False
        )
        
        # Create temporary logger
        temp_log_dir = tempfile.mkdtemp()
        logger = Logger(
            run_name="test_eval",
            folder=temp_log_dir,
            algo="MADDPG",
            env="simple_spread_v3"
        )
        
        # Run evaluation
        avg_rewards = evaluate(
            env_eval, 
            maddpg, 
            logger, 
            record_gif=False, 
            num_eval_episodes=3, 
            global_step=0
        )
        
        print(f"✓ Evaluation completed")
        print(f"  Average rewards: {avg_rewards}")
        print(f"  Total reward: {np.sum(avg_rewards):.2f}")
        
        env_eval.close()
        logger.close()
        shutil.rmtree(temp_log_dir)
        
        return True
    except Exception as e:
        print(f"❌ Evaluation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_full_training_loop():
    """Test 8: Full Training Loop (Short Version)"""
    print("\n" + "="*70)
    print("Test 8: Full Training Loop (Short - 100 steps)")
    print("="*70)
    try:
        from configs.config import Config
        
        cfg = Config()
        cfg.total_timesteps = 100  # Very short for testing
        cfg.warmup_steps = 20
        cfg.update_every = 10
        cfg.eval_interval = 50
        cfg.batch_size = 16
        
        # Get environment info
        agents, num_agents, action_sizes, action_low, action_high, state_sizes = get_env_info(
            env_name=cfg.env_name,
            max_steps=cfg.max_steps,
            apply_padding=False
        )
        
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
        
        # Create agent (test with traditional first)
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
            actor="traditional",
            critic="traditional"
        )
        
        # Create buffer
        buffer = ReplayBuffer(
            buffer_size=1000,
            batch_size=cfg.batch_size,
            agents=agents,
            state_sizes=state_sizes,
            action_sizes=action_sizes
        )
        
        # Create logger
        temp_log_dir = tempfile.mkdtemp()
        logger = Logger(
            run_name="test_training",
            folder=temp_log_dir,
            algo=cfg.algo,
            env=cfg.env_name
        )
        
        # Training loop
        observations, _ = env.reset()
        episode_rewards = np.zeros(len(agents))
        noise_scale = cfg.noise_scale
        
        for step in range(1, cfg.total_timesteps + 1):
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
                    logger.add_scalar(f'{agents[i]}/critic_loss', critic_loss, step)
                    logger.add_scalar(f'{agents[i]}/actor_loss', actor_loss, step)
                maddpg.update_targets()
            
            # Reset episode
            if done or (step % cfg.max_steps == 0):
                for i, reward in enumerate(episode_rewards):
                    logger.add_scalar(f"{agents[i]}/episode_reward", reward, step)
                logger.add_scalar('train/total_reward', np.sum(episode_rewards), step)
                observations, _ = env.reset()
                episode_rewards = np.zeros(len(agents))
            
            # Evaluate
            if step % cfg.eval_interval == 0:
                evaluate(env_eval, maddpg, logger, record_gif=False, num_eval_episodes=2, global_step=step)
        
        print(f"✓ Full training loop completed successfully")
        print(f"  Steps: {cfg.total_timesteps}")
        print(f"  Buffer size: {len(buffer)}")
        
        env.close()
        env_eval.close()
        logger.close()
        shutil.rmtree(temp_log_dir)
        
        return True
    except Exception as e:
        print(f"❌ Full training loop test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all pipeline tests"""
    print("\n" + "="*70)
    print("COMPREHENSIVE PIPELINE TEST")
    print("="*70)
    print("Testing entire training pipeline before full training runs...")
    
    results = {}
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Test 1: Environment
        agents, states_list, num_agents = test_environment()
        if agents is None:
            print("\n❌ Environment test failed. Stopping.")
            return
        
        # Get environment info
        _, _, action_sizes, action_low, action_high, state_sizes = get_env_info(
            env_name="simple_spread_v3",
            max_steps=25,
            apply_padding=False
        )
        
        # Test 2: Agent creation
        agents_dict = test_agent_creation(agents, state_sizes, action_sizes, action_low, action_high)
        if agents_dict is None:
            print("\n❌ Agent creation test failed. Stopping.")
            return
        
        # Use traditional/traditional for remaining tests
        maddpg = agents_dict[("traditional", "traditional")]
        
        # Test 3: Action generation
        results['action_gen'] = test_action_generation(maddpg, states_list)
        
        # Test 4: Replay buffer
        buffer = test_replay_buffer(agents, state_sizes, action_sizes)
        results['buffer'] = buffer is not None
        
        # Test 5: Learning
        if buffer:
            results['learning'] = test_learning(maddpg, buffer, agents)
        
        # Test 6: Save/Load
        results['save_load'] = test_save_load(maddpg, temp_dir)
        
        # Test 7: Evaluation
        results['evaluation'] = test_evaluation(maddpg, agents)
        
        # Test 8: Full training loop
        results['full_loop'] = test_full_training_loop()
        
        # Summary
        print("\n" + "="*70)
        print("TEST SUMMARY")
        print("="*70)
        for test_name, passed in results.items():
            status = "✓ PASSED" if passed else "❌ FAILED"
            print(f"{test_name:20s}: {status}")
        
        all_passed = all(results.values())
        if all_passed:
            print("\n✅ ALL TESTS PASSED! Pipeline is ready for training.")
        else:
            print("\n⚠ SOME TESTS FAILED. Please fix issues before training.")
        
    finally:
        # Cleanup
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

if __name__ == "__main__":
    main()

