import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math


class InvariantForagingEnv:
    """
    A multi-agent foraging environment with TRUE permutation and rotational invariance.
    
    Key Design Principles:
    1. PERMUTATION INVARIANCE: Observations use set-based aggregations (mean/max pooling)
       rather than ordered lists, so agent ordering doesn't matter.
    2. ROTATIONAL INVARIANCE: Observations use only distances and counts, never raw 
       x,y coordinates, so rotating the world doesn't change observations.
    """
    
    def __init__(self, n_agents=3, n_resources=5, world_size=10.0, dt=0.05):
        self.n_agents = n_agents
        self.n_resources = n_resources
        self.world_size = world_size
        self.dt = dt
        self.max_speed = 2.0  # Increased from 1.0
        
        # Observation space design parameters
        self.n_distance_bins = 8  # For discretizing distances
        self.max_observe_dist = world_size * 1.5
        
        self.reset()

    def reset(self):
        """Initialize environment with random positions."""
        # Agent states
        self.positions = np.random.uniform(-self.world_size, self.world_size, 
                                          (self.n_agents, 2))
        self.velocities = np.zeros((self.n_agents, 2))
        self.carrying = np.zeros(self.n_agents, dtype=int)

        # Resources (randomly placed)
        self.resources = np.random.uniform(-self.world_size, self.world_size, 
                                          (self.n_resources, 2))
        self.resource_active = np.ones(self.n_resources, dtype=bool)

        # Drop-off point at origin
        self.dropoff = np.array([0.0, 0.0])

        return self._get_obs()

    def step(self, actions):
        """Execute one step of the environment."""
        # Clip actions to valid range
        actions = np.clip(actions, -1.0, 1.0)

        # Update velocities (with acceleration from actions)
        self.velocities += actions * self.dt
        
        # Enforce speed limit
        speeds = np.linalg.norm(self.velocities, axis=1, keepdims=True)
        self.velocities = np.where(
            speeds > self.max_speed,
            self.velocities / speeds * self.max_speed,
            self.velocities
        )
        
        # Update positions
        self.positions += self.velocities * self.dt

        reward = 0.0

        # Resource collection
        for i in range(self.n_agents):
            if self.carrying[i] == 0:  # Not carrying anything
                dists = np.linalg.norm(self.resources - self.positions[i], axis=1)
                idx = np.argmin(dists)
                if self.resource_active[idx] and dists[idx] < 0.3:
                    self.carrying[i] = 1
                    self.resource_active[idx] = False

            # Drop-off at central point
            if self.carrying[i] == 1:
                if np.linalg.norm(self.positions[i] - self.dropoff) < 0.4:
                    reward += 1.0  # Success!
                    self.carrying[i] = 0

        # Collision penalty (encourage spreading out)
        for i in range(self.n_agents):
            for j in range(i + 1, self.n_agents):
                if np.linalg.norm(self.positions[i] - self.positions[j]) < 0.25:
                    reward -= 0.05

        # Small time penalty (encourages efficiency)
        reward -= 0.001 * self.n_agents

        return self._get_obs(), reward, False, {}

    def _get_obs(self):
        """
        Construct INVARIANT observations for each agent.
        
        WHY THIS DESIGN:
        ===============
        
        PERMUTATION INVARIANCE:
        - We don't list neighbors in any particular order
        - Instead, we aggregate: count neighbors in distance bins, compute mean distance, etc.
        - Mathematical property: f(set) doesn't depend on element ordering
        
        ROTATIONAL INVARIANCE:
        - We NEVER use raw (x,y) coordinates (they change under rotation)
        - We ONLY use distances: ||p_i - p_j|| (invariant under rotation)
        - We count objects in radial bins (like a radar, but rotation-symmetric)
        
        Observation vector per agent:
        [
            # Self state (1 feature)
            carrying_flag,
            
            # Neighbor information (permutation-invariant aggregations)
            mean_neighbor_distance,
            min_neighbor_distance, 
            num_neighbors_close,    # within 2.0 units
            num_neighbors_medium,   # within 5.0 units
            num_neighbors_far,      # beyond 5.0 units
            
            # Resource information (permutation-invariant aggregations)
            mean_resource_distance,
            min_resource_distance,
            num_resources_close,    # within 2.0 units
            num_resources_medium,   # within 5.0 units
            num_resources_far,      # beyond 5.0 units
            num_active_resources,
            
            # Dropoff information (rotationally invariant)
            distance_to_dropoff,
            
            # Agent's own motion (speed, not velocity direction which would break invariance)
            own_speed
        ]
        
        Total: 14 features per agent
        """
        obs = []
        
        for i in range(self.n_agents):
            # === SELF STATE ===
            carrying_flag = float(self.carrying[i])
            own_speed = np.linalg.norm(self.velocities[i])
            
            # === NEIGHBOR FEATURES (other agents) ===
            neighbor_dists = []
            for j in range(self.n_agents):
                if i != j:
                    dist = np.linalg.norm(self.positions[j] - self.positions[i])
                    neighbor_dists.append(dist)
            
            if len(neighbor_dists) > 0:
                neighbor_dists = np.array(neighbor_dists)
                mean_neighbor_dist = np.mean(neighbor_dists)
                min_neighbor_dist = np.min(neighbor_dists)
                num_neighbors_close = np.sum(neighbor_dists < 2.0)
                num_neighbors_medium = np.sum((neighbor_dists >= 2.0) & (neighbor_dists < 5.0))
                num_neighbors_far = np.sum(neighbor_dists >= 5.0)
            else:
                mean_neighbor_dist = 0.0
                min_neighbor_dist = 0.0
                num_neighbors_close = 0.0
                num_neighbors_medium = 0.0
                num_neighbors_far = 0.0
            
            # === RESOURCE FEATURES ===
            resource_dists = []
            for r in range(self.n_resources):
                if self.resource_active[r]:
                    dist = np.linalg.norm(self.resources[r] - self.positions[i])
                    resource_dists.append(dist)
            
            if len(resource_dists) > 0:
                resource_dists = np.array(resource_dists)
                mean_resource_dist = np.mean(resource_dists)
                min_resource_dist = np.min(resource_dists)
                num_resources_close = np.sum(resource_dists < 2.0)
                num_resources_medium = np.sum((resource_dists >= 2.0) & (resource_dists < 5.0))
                num_resources_far = np.sum(resource_dists >= 5.0)
                num_active = len(resource_dists)
            else:
                mean_resource_dist = 0.0
                min_resource_dist = 0.0
                num_resources_close = 0.0
                num_resources_medium = 0.0
                num_resources_far = 0.0
                num_active = 0.0
            
            # === DROPOFF FEATURES ===
            dist_to_dropoff = np.linalg.norm(self.positions[i] - self.dropoff)
            
            # Construct observation vector (14 features)
            agent_obs = np.array([
                carrying_flag,
                mean_neighbor_dist,
                min_neighbor_dist,
                num_neighbors_close,
                num_neighbors_medium,
                num_neighbors_far,
                mean_resource_dist,
                min_resource_dist,
                num_resources_close,
                num_resources_medium,
                num_resources_far,
                num_active,
                dist_to_dropoff,
                own_speed
            ])
            
            obs.append(agent_obs)

        return np.array(obs)

    def render(self):
        """Visualize the environment using pygame."""
        import pygame
        if not hasattr(self, 'screen'):
            pygame.init()
            self.screen_size = 600
            self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
            self.clock = pygame.time.Clock()

        self.screen.fill((30, 30, 30))

        def world_to_screen(pos):
            scale = self.screen_size / (2 * self.world_size)
            return (pos + self.world_size) * scale

        # Draw dropoff zone
        d = world_to_screen(self.dropoff)
        pygame.draw.circle(self.screen, (255, 255, 0), d.astype(int), 15)

        # Draw active resources
        for r, active in zip(self.resources, self.resource_active):
            if active:
                p = world_to_screen(r)
                pygame.draw.circle(self.screen, (0, 255, 0), p.astype(int), 8)

        # Draw agents (color changes based on carrying state)
        for i in range(self.n_agents):
            p = world_to_screen(self.positions[i])
            color = (0, 150, 255) if self.carrying[i] == 0 else (255, 100, 0)
            pygame.draw.circle(self.screen, color, p.astype(int), 10)

        pygame.display.flip()
        self.clock.tick(30)


def test_invariance():
    """
    Test that observations are truly invariant to permutations and rotations.
    """
    print("="*60)
    print("TESTING PERMUTATION AND ROTATIONAL INVARIANCE")
    print("="*60)
    
    env = InvariantForagingEnv(n_agents=3, n_resources=5)
    
    # Set fixed state for reproducibility
    np.random.seed(42)
    env.positions = np.array([[1.0, 2.0], [3.0, 4.0], [-1.0, -2.0]])
    env.velocities = np.array([[0.1, 0.2], [0.3, 0.1], [-0.1, 0.0]])
    env.resources = np.array([[2.0, 1.0], [-3.0, 2.0], [1.0, -1.0], [4.0, 4.0], [-2.0, -3.0]])
    env.resource_active = np.array([True, True, False, True, True])
    env.carrying = np.array([0, 1, 0])
    
    obs1 = env._get_obs()
    print("\nOriginal observations (agent 0):")
    print(obs1[0])
    
    # TEST 1: Permutation invariance
    print("\n" + "="*60)
    print("TEST 1: PERMUTATION INVARIANCE")
    print("="*60)
    print("Swapping agents 0 and 1...")
    
    # Swap agents 0 and 1
    env.positions[[0, 1]] = env.positions[[1, 0]]
    env.velocities[[0, 1]] = env.velocities[[1, 0]]
    env.carrying[[0, 1]] = env.carrying[[1, 0]]
    
    obs2 = env._get_obs()
    print("Observations after swap (now agent 0 is at old agent 1's position):")
    print(obs2[0])
    print("\nExpected: Should match original agent 1's observation")
    print(obs1[1])
    print(f"\nAre they equal? {np.allclose(obs2[0], obs1[1])}")
    
    # TEST 2: Rotational invariance
    print("\n" + "="*60)
    print("TEST 2: ROTATIONAL INVARIANCE")
    print("="*60)
    print("Rotating entire world by 90 degrees...")
    
    # Reset to original state
    env.positions = np.array([[1.0, 2.0], [3.0, 4.0], [-1.0, -2.0]])
    env.velocities = np.array([[0.1, 0.2], [0.3, 0.1], [-0.1, 0.0]])
    env.resources = np.array([[2.0, 1.0], [-3.0, 2.0], [1.0, -1.0], [4.0, 4.0], [-2.0, -3.0]])
    obs_before = env._get_obs()
    
    # Rotate by 90 degrees
    theta = np.pi / 2
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    
    env.positions = env.positions @ R.T
    env.velocities = env.velocities @ R.T
    env.resources = env.resources @ R.T
    # dropoff stays at origin
    
    obs_after = env._get_obs()
    
    print("Observations before rotation (agent 0):")
    print(obs_before[0])
    print("\nObservations after rotation (agent 0):")
    print(obs_after[0])
    print(f"\nAre they equal? {np.allclose(obs_before[0], obs_after[0], atol=1e-6)}")
    
    if np.allclose(obs_before[0], obs_after[0], atol=1e-6):
        print("\n✓ SUCCESS: Observations are rotationally invariant!")
    else:
        print("\n✗ FAILURE: Observations changed under rotation")
        print("Difference:", np.abs(obs_before[0] - obs_after[0]))


# Example usage
if __name__ == "__main__":
    import sys
    
    # Check if user wants to run tests or render
    if len(sys.argv) > 1 and sys.argv[1] == "--render":
        print("="*60)
        print("RENDERING ENVIRONMENT (Press ESC or close window to exit)")
        print("="*60)
        
        import pygame
        env = InvariantForagingEnv()
        obs = env.reset()
        
        running = True
        total_reward = 0
        step = 0
        
        while running:
            # Random actions (replace with your trained policy)
            actions = np.random.uniform(-1, 1, (env.n_agents, 2))
            obs, reward, done, info = env.step(actions)
            total_reward += reward
            
            # Render
            env.render()
            
            # Print status every 50 steps
            if step % 50 == 0:
                active_resources = np.sum(env.resource_active)
                print(f"Step {step}: Reward = {reward:.4f}, Total = {total_reward:.4f}, "
                      f"Resources left = {active_resources}")
            
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    if event.key == pygame.K_r:  # Press R to reset
                        env.reset()
                        total_reward = 0
                        step = 0
                        print("Environment reset!")
            
            step += 1
            
            # Auto-reset after collecting all resources
            if np.sum(env.resource_active) == 0:
                print(f"All resources collected! Final reward: {total_reward:.2f}")
                env.reset()
                total_reward = 0
                step = 0
        
        pygame.quit()
        print("Rendering closed.")
    
    else:
        # Run invariance tests
        test_invariance()
        
        print("\n" + "="*60)
        print("RUNNING ENVIRONMENT SIMULATION")
        print("="*60)
        print("(Use --render flag to visualize: python script.py --render)")
        
        # Run environment
        env = InvariantForagingEnv()
        obs = env.reset()
        print(f"\nObservation shape: {obs.shape}")
        print(f"Features per agent: {obs.shape[1]}")
        
        total_reward = 0
        for step in range(100):
            actions = np.random.uniform(-1, 1, (env.n_agents, 2))
            obs, reward, done, info = env.step(actions)
            total_reward += reward
            if step % 20 == 0:
                print(f"Step {step}: Reward = {reward:.4f}, Total = {total_reward:.4f}")