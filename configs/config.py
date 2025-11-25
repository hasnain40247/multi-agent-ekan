class Config:
    def __init__(self):
        # fixed defaults
        self.algo = "MADDPG"
        self.env_name = "simple_spread_v3"

        # training settings
        self.total_timesteps = int(1e5)
        # total_timesteps	1500000

        self.buffer_size = int(1e6)
        # buffer_size	1000000

        self.warmup_steps = 20000
        # warmup_steps	20000

        self.batch_size = 512
        # batch_size	256

        self.max_steps = 25
        # max_steps	25


        # RL hyperparameters
        self.gamma = 0.95
        # gamma	0.95

        self.tau = 0.01
        # tau	0.01

        self.actor_lr = 1e-3
        # actor_lr	0.001

        self.critic_lr = 2e-3
        # critic_lr	0.002

        self.hidden_sizes = "64,64"
        # hidden_sizes	64,64


        # update frequency
        self.update_every = 15
        # update_every	15


        # noise settings
        self.noise_scale = 0.3
        # noise_scale	0.3

        self.min_noise = 0.05
        # min_noise	0.01

        self.noise_decay_steps = int(3e5)
        # noise_decay_steps	500000

        self.use_noise_decay = False
        # use_noise_decay	True


        # evaluation settings
        self.eval_interval = 5000
        # eval_interval	5000


        self.actor = "traditional"
        self.critic = "traditional"

        self.render_mode = None
        self.create_gif	=False


        # render_mode	None


    def apply_cli(self, args):
        """Override config attributes with CLI args if they are not None."""
        for key, val in vars(args).items():
            if hasattr(self, key) and val is not None:
                setattr(self, key, val)
