from info import resource_monitor, log_metrics, display_metrics, plot_confusion_matrix
from info.rl import log_episode, display_episodes
import time
import random
import math
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def test_rl_info():
    reward_bias = 0
    max_steps = 100_000
    step = 0
    while step < max_steps:
        ep_len = 0
        for i in range(4096):
            step+=1
            ep_len += 1
            if random.random() < 0.01:
                log_episode(random.random() + reward_bias, ep_len, step=(step, max_steps))
                ep_len = 0
                reward_bias += math.exp(1 + step / max_steps) * 0.001 * random.random()
    display_episodes()

def test_train_info():
    loss = 10.0
    epochs = 3
    batches = 300

    for it in range(1):
        for epoch in range(epochs):
            for batch in range(batches):
                loss_ =  math.log2(abs(loss))
                acc = loss + random.random()
                log_metrics(
                    name=("loss", "acc"),
                    value=(loss_, acc),
                    epoch_idx=(epoch, epochs),
                    batch_idx=(batch, batches)
                )
                loss -= 1e-2
                time.sleep(0.001)
        print('\n\n\n')
            
    display_metrics()

def test_graph_plotting():
    values = [10, 15, 17, 18, 18.3]
    values2 = [9, 14, 17.5, 19, 20.1]
    values3 = [10, 14.2, 17.7, 19.2, 20.3]
    from info import plot_graph
    plot_graph((values, values2, values3), marker=".")

import torch
import torch.nn.functional as F
from info import benchmark
sil = torch.nn.SiLU()
x = torch.full((2048, 2048), 2.31, device="cpu")
def test_swish():
    x / (1 + torch.exp(-x))

def test_swish2():
    F.silu(x)


def test_rish():
    k = torch.exp(x)
    (x - 1) * k / (1 + k)


def test_conf_matrix():
    y =     [0, 0, 1, 20, 0, 0, 1, 2, 3, 4, 5, 6, 7, 12, 11, 14]
    y_hat = [0, 1, 1, 20, 0, 0, 1, 2, 9, 3, 4, 5, 10, 12, 11, 14]
    plot_confusion_matrix(y_hat, y, normalize=True)


test_conf_matrix()