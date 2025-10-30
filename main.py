
from environment.mpe import MPEEnvironment
from mappo.mappo import MAPPO
import argparse


if __name__ == "__main__":


    parser = argparse.ArgumentParser(description="Stuff")
    parser.add_argument("--n_agents", type=int, default=3, help="Number of agents")
    parser.add_argument("--max_cycles", type=int, default=25, help="Maximum number of cycles per episode")
    parser.add_argument("--continuous", action="store_true", help="Use continuous actions")
    # parser.add_argument("--render", action="store_true", help="Enable rendering (human mode)") commenting out to test stuff
    parser.add_argument("--steps", type=int, default=50, help="Number of visualization steps")
    parser.add_argument("--delay", type=float, default=0, help="Delay between frames")
    parser.add_argument("--manual",  action="store_true", help="Manually play with the environment")

    args = parser.parse_args()

    # render_mode = "human" if args.render else None

    env = MPEEnvironment(
        n_agents=args.n_agents,
        max_cycles=args.max_cycles,
        continuous_actions=args.continuous,
        render_mode="human"
    )

    specs = env.get_specs()
    print("\nSpecs:")
    for k, v in specs.items():
        print(f"{k}: {v}")

    env.visualize(
        steps=args.steps,
        random_actions=True,
        manual=True if args.manual else False
    )

    env = MPEEnvironment(
        n_agents=args.n_agents,
        max_cycles=args.max_cycles,
        continuous_actions=args.continuous,
        render_mode=None
    )

    mappo = MAPPO(env)

    mappo.collect_trajectories(steps=50)

    env.close()
