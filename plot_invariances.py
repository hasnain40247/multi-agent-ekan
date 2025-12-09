import os
import json
import numpy as np
import matplotlib.pyplot as plt


def main(path="invariance_results.json"):
    with open(path, "r") as f:
        res = json.load(f)

    # make sure output dir exists
    out_dir = "plots/eval"
    os.makedirs(out_dir, exist_ok=True)

    cfgA = "EKAN+PIcritic"
    cfgB = "EKAN+Tradcritic"
    cfgC = "TradActor+Tradcritic"

    # ==============================
    #  Actor SO(2) equivariance
    # ==============================
    actA = res[cfgA]["actor_SO2_equivariance"]
    actB = res[cfgB]["actor_SO2_equivariance"]
    actC = res[cfgC]["actor_SO2_equivariance"]

    theta = actA["theta_deg"]
    per_batch_A = np.array(actA["per_batch_L2"], dtype=float)
    per_batch_B = np.array(actB["per_batch_L2"], dtype=float)
    per_batch_C = np.array(actC["per_batch_L2"], dtype=float)

    labels_actor = [
        "EKAN + PI critic",
        "EKAN + Trad critic",
        "Trad actor + Trad critic",
    ]

    # Figure 1: Distribution of equivariance error for all three runs
    plt.figure(figsize=(7, 4))
    plt.boxplot(
        [per_batch_A, per_batch_B, per_batch_C],
        labels=labels_actor,
        showmeans=True,
    )
    plt.ylabel(r"$\|\pi(R_\theta s) - R_\theta \pi(s)\|_2$")
    plt.title(f"Actor SO(2) equivariance errors (θ = {theta:.1f}°)")
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "actor_equivariance_boxplot.png"), dpi=300)

    # Figure 1b: bar plot of mean equivariance error with std
    plt.figure(figsize=(6, 4))
    means = [actA["mean_L2"], actB["mean_L2"], actC["mean_L2"]]
    stds = [per_batch_A.std(), per_batch_B.std(), per_batch_C.std()]
    x = np.arange(3)
    bars = plt.bar(x, means, yerr=stds, capsize=5)
    plt.xticks(x, labels_actor, rotation=10)
    plt.ylabel("Mean L2 equivariance error")
    plt.title(f"Actor equivariance (θ = {theta:.1f}°)")
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    # annotate bars with values
    for bx, m in zip(bars, means):
        plt.text(
            bx.get_x() + bx.get_width() / 2.0,
            m,
            f"{m:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "actor_equivariance_means.png"), dpi=300)

    # ==============================
    #  Critic permutation invariance
    # ==============================
    critA = res[cfgA]["critic_perm_invariance"]
    critB = res[cfgB]["critic_perm_invariance"]
    critC = res[cfgC]["critic_perm_invariance"]

    per_batch_crit_A = np.array(critA["per_batch_abs_diff"], dtype=float)
    per_batch_crit_B = np.array(critB["per_batch_abs_diff"], dtype=float)
    per_batch_crit_C = np.array(critC["per_batch_abs_diff"], dtype=float)

    perm = critA["perm"]

    labels_critic = [
        "EKAN + PI critic",
        "EKAN + Trad critic",
        "Trad actor + Trad critic",
    ]

    # Figure 2: Distribution of permutation invariance error
    plt.figure(figsize=(7, 4))
    plt.boxplot(
        [per_batch_crit_A, per_batch_crit_B, per_batch_crit_C],
        labels=labels_critic,
        showmeans=True,
    )
    plt.axhline(0.0, linestyle="--", linewidth=1, alpha=0.6)  # highlight perfect invariance
    plt.ylabel(r"$|Q(s,a) - Q(\pi s, \pi a)|$")
    plt.title(f"Critic permutation invariance (perm = {perm})")
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "critic_invariance_boxplot.png"), dpi=300)

    # Figure 2b: bar plot of mean invariance error with std
    plt.figure(figsize=(6, 4))
    crit_means = [
        critA["mean_abs_diff"],
        critB["mean_abs_diff"],
        critC["mean_abs_diff"],
    ]
    crit_stds = [
        per_batch_crit_A.std(),
        per_batch_crit_B.std(),
        per_batch_crit_C.std(),
    ]
    x = np.arange(3)
    bars = plt.bar(x, crit_means, yerr=crit_stds, capsize=5)
    plt.xticks(x, labels_critic, rotation=10)
    plt.ylabel("Mean |ΔQ| under permutation")
    plt.title("Critic permutation invariance (swap first two agents)")
    plt.axhline(0.0, linestyle="--", linewidth=1, alpha=0.6)
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    for bx, m in zip(bars, crit_means):
        plt.text(
            bx.get_x() + bx.get_width() / 2.0,
            m,
            f"{m:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "critic_invariance_means.png"), dpi=300)

    print("Saved:")
    print(f"  {os.path.join(out_dir, 'actor_equivariance_boxplot.png')}")
    print(f"  {os.path.join(out_dir, 'actor_equivariance_means.png')}")
    print(f"  {os.path.join(out_dir, 'critic_invariance_boxplot.png')}")
    print(f"  {os.path.join(out_dir, 'critic_invariance_means.png')}")


if __name__ == "__main__":
    main()
