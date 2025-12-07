from maddpg.ckpt_plot.plot_curve import read_csv, plot_result,plot_multi_results

# Example: read csv
# metrics = read_csv("ckpt_plot/simple_spread_n3_continuous_actor_mlp_lr_0.0001_critic_mlp_lr_0.001_fixed_lr_batch_size_1024_actor_clip_grad_norm_0.5_seed1/train_curve.csv")
# metrics = read_csv("ckpt_plot/simple_spread_n3_continuous_actor_mlp_lr_0.0001_critic_gcn_max_lr_0.001_fixed_lr_batch_size_1024_actor_clip_grad_norm_0.5_seed1/train_curve.csv")

# # Extract data
# t = metrics["steps"]  # Use actual step values from CSV
# reward = metrics["rewards"]

# plot_result(t, reward, "reward_plot", "steps", "reward")





metrics1 = read_csv("ckpt_plot/simple_spread_n3_continuous_actor_mlp_lr_0.0001_critic_gcn_max_lr_0.001_fixed_lr_batch_size_1024_actor_clip_grad_norm_0.5_seed1/train_curve.csv")
metrics2 = read_csv("ckpt_plot/simple_spread_n3_continuous_actor_mlp_lr_0.0001_critic_mlp_lr_0.001_fixed_lr_batch_size_1024_actor_clip_grad_norm_0.5_seed1/train_curve.csv")


results = {
    "mlp + gcn": (metrics1["steps"], metrics1["rewards"]),
    "mlp + mlp": (metrics2["steps"], metrics2["rewards"]),
}

plot_multi_results(results, "reward_compare", "steps", "reward")


# # Plot two curves (train / val)
# if "val_reward" in metrics:
#     plot_result2(t, metrics["reward"], metrics["val_reward"],
#                  "reward_train_val", "steps", "reward")

# # Example of multi-plot
# plot_result_mul(
#     fig_name="multi_plot",
#     x_label="steps",
#     y_label="value",
#     legend=["curve1", "curve2", "curve3"],
#     t1=t, r1=metrics["reward"],
#     t2=t, r2=metrics.get("loss", None),
#     t3=None, r3=None
# )
