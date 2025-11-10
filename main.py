from environment.mpe import MPEEnvironment
from mappo.mappo import MAPPO
import argparse
import logging
import wandb


def build_env(n_agents, max_cycles, continuous, render_mode):
    return MPEEnvironment(
        n_agents=n_agents,
        max_cycles=max_cycles,
        continuous_actions=continuous,
        render_mode=render_mode,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MPE Mappo baseline")

    # environment
    parser.add_argument("--n_agents", type=int, default=3, help="Number of agents")
    parser.add_argument(
        "--max_cycles",
        type=int,
        default=25,
        help="Maximum number of cycles per episode",
    )
    parser.add_argument(
        "--continuous", action="store_false", help="Use continuous actions"
    )
    # parser.add_argument("--render", action="store_true", help="Enable rendering (human mode)") commenting out to test stuff

    # modes
    parser.add_argument(
        "--mode",
        choices=["visualize", "train"],
        default="train",
        help="Run visualization or training",
    )
    parser.add_argument(
        "--steps", type=int, default=50, help="Number of visualization steps"
    )
    parser.add_argument("--delay", type=float, default=0, help="Delay between frames")
    parser.add_argument(
        "--manual", action="store_true", help="Manually play with the environment"
    )

    # training
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument(
        "--steps_per_epoch",
        type=int,
        default=1024,
        help="Env steps per epoch (across agents)",
    )
    parser.add_argument(
        "--value_coef", type=float, default=0.5, help="Value loss weight"
    )
    parser.add_argument(
        "--entropy_coef", type=float, default=0.01, help="Entropy bonus weight"
    )
    parser.add_argument(
        "--max_grad_norm", type=float, default=0.5, help="Grad clip norm"
    )
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    
    # wandb arguments
    parser.add_argument("--wandb_project", type=str, default="multi-agent-ekan", help="W&B project name")
    parser.add_argument("--wandb_name", type=str, default=None, help="W&B run name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="W&B entity/team name")
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")
    
    args = parser.parse_args()

    # render_mode = "human" if args.render else None

    if args.mode == "visualize":
        env = build_env(
            args.n_agents, args.max_cycles, args.continuous, render_mode="human"
        )
        specs = env.get_specs()
        print("\nSpecs:")
        for k, v in specs.items():
            print(f"{k}: {v}")

        env.visualize(
            steps=args.steps,
            random_actions=not args.manual,
            manual=args.manual,
            delay=args.delay,
        )
        env.close()

    else:
        # Initialize wandb
        if not args.no_wandb:
            wandb_config = {
                "n_agents": args.n_agents,
                "max_cycles": args.max_cycles,
                "continuous_actions": args.continuous,
                "epochs": args.epochs,
                "steps_per_epoch": args.steps_per_epoch,
                "value_coef": args.value_coef,
                "entropy_coef": args.entropy_coef,
                "max_grad_norm": args.max_grad_norm,
                "gamma": args.gamma,
                "lr": args.lr,
            }
            
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_name,
                entity=args.wandb_entity,
                config=wandb_config,
            )
        
        env = build_env(
            args.n_agents, args.max_cycles, args.continuous, render_mode=None
        )

        agent = MAPPO(env, gamma=args.gamma, lr=args.lr, use_wandb=not args.no_wandb)
        agent.train(
            epochs=args.epochs,
            steps_per_epoch=args.steps_per_epoch,
            value_coef=args.value_coef,
            entropy_coef=args.entropy_coef,
            max_grad_norm=args.max_grad_norm,
        )

        if not args.no_wandb:
            wandb.finish()
        
        env.close()
