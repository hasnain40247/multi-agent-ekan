import os
import sys
import numpy as np
import matplotlib as mpl
import csv

mpl.use('Agg')
from matplotlib import pyplot as plt


def avg_list(l, avg_group_size=2):
    ret_l = []
    n = len(l)
    h_size = avg_group_size / 2
    for i in range(n):
        left = int(max(0, i - h_size))
        right = int(min(n, i + h_size))
        ret_l.append(np.mean(l[left:right]))
    return ret_l


def plot_result(t1, r1, fig_name, x_label, y_label):
    plt.close()
    fig, ax = plt.subplots()
    
    # Scale steps by 10^6 and rewards by 10^2
    t1_scaled = [t / 1e6 for t in t1]
    r1_scaled = [r / 1e2 for r in r1]
    
    base, = ax.plot(t1_scaled, avg_list(r1_scaled))

    # Disable scientific notation on axes
    ax.ticklabel_format(style='plain', axis='both')
    ax.grid()
    ax.set_xlabel(x_label + r' ($\times 10^6$)')
    ax.set_ylabel(y_label + r' ($\times 10^2$)')
    ax.set_title(fig_name)

    try:
        plt.savefig(fig_name + '.pdf')
        plt.savefig(fig_name + '.png')
    except:
        print('ERROR:', sys.exc_info()[0])
        print('Terminate Program')
        sys.exit()
    print('INFO: Wrote plot to ' + fig_name + '.pdf')

def plot_multi_results(results, fig_name, x_label, y_label):
    """
    results: {
        "actorA_criticX": (steps_list, reward_list),
        "actorB_criticY": (steps_list, reward_list),
        ...
    }
    """
    plt.close()
    fig, ax = plt.subplots()

    for label, (t, r) in results.items():
        t_scaled = [x / 1e6 for x in t]
        r_scaled = [y / 1e2 for y in r]

        ax.plot(t_scaled, avg_list(r_scaled), label=label)

    ax.grid()
    ax.set_xlabel(x_label + r' ($\times 10^6$)')
    ax.set_ylabel(y_label + r' ($\times 10^2$)')
    ax.set_title(fig_name)
    ax.legend()

    plt.savefig(fig_name + ".pdf")
    plt.savefig(fig_name + ".png")
    print("INFO: wrote", fig_name)

def plot_result2(t1, r1, r2, fig_name, x_label, y_label):
    plt.close()
    
    # Scale steps by 10^6 and values by 10^2
    t1_scaled = [t / 1e6 for t in t1]
    r1_scaled = [r / 1e2 for r in r1]
    r2_scaled = [r / 1e2 for r in r2]
    
    l1, = plt.plot(t1_scaled, r1_scaled)
    l2, = plt.plot(t1_scaled, r2_scaled)

    plt.grid()
    plt.legend([l1, l2], ['train', 'val'])
    plt.xlabel(x_label + r' ($\times 10^6$)')
    plt.ylabel(y_label + r' ($\times 10^2$)')
    plt.title(fig_name)

    try:
        plt.savefig(fig_name + '.pdf')
        plt.savefig(fig_name + '.png')
    except:
        print('ERROR:', sys.exc_info()[0])
        print('Terminate Program')
        sys.exit()
    print('INFO: Wrote plot to ' + fig_name + '.pdf')


def read_csv(csv_path):
    res = {}
    with open(csv_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            key = row[0]
            values = []
            for r in row[1:]:
                r = r.strip()
                if r == "":
                    continue
                values.append(float(r))
            res[key] = values
    return res


# Example usage:
if __name__ == "__main__":
    data = read_csv("your_data.csv")
    steps = data['steps']
    rewards = data['rewards']
    plot_result(steps, rewards, "training_rewards", "Steps", "Rewards")