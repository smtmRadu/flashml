import time
import random
import math
import os


def test_rl_info():
    from tools.rl import log_episode

    reward_bias = 0
    max_steps = 12000
    step = 0
    while step < max_steps:
        ep_len = 0
        for i in range(4096):
            step += 1
            ep_len += 1
            if random.random() < 0.01:
                log_episode(
                    random.random() + reward_bias,
                    ep_len,
                    step=(step, max_steps),
                    other_metrics={"Rand": random.random()},
                )
                ep_len = 0
                reward_bias += math.exp(1 + step / max_steps) * 0.001 * random.random()
            time.sleep(0.001)


def test_train_info():
    from tools import log_metrics

    HYPERPARAMS = {
        "model": "Qwen/Qwen3-0.6B",  # "tiiuae/Falcon-H1-0.5B-Base",
        "continue_from_index": -1,
        "seed": 42,
        "batch_size": 2,
        "gradient_accumulation": 8,
        "epochs": 1,
        "lr": 2e-5,
        "betas": (0.9, 0.999),
        "weight_decay": 0.005,
    }
    loss = 10.0
    epochs = 3
    batches = 300

    for it in range(1):
        for epoch in range(batches * epochs):
            loss_ = math.log2(abs(loss))
            acc = loss + random.random()
            log_metrics(
                {"loss": loss_, "acc": acc},
                step=(epoch, batches * epochs),
                hyperparams=HYPERPARAMS,
            )
            loss -= 1e-2
            time.sleep(0.001)
        print("\n\n\n")


def test_plotting():
    from tools import plot_graph
    # values = [10, 15, 17, 18, 18.3]
    # values2 = [9, 14, 17.5, 19, 20.1]
    # values3 = [10, 14.2, 17.7, 19.2, 20.3]
    #
    # plot_graph((values, values2, values3), marker=".")

    values = [random.random() for _ in range(1000)]
    plot_graph(values, color="black")


def test_conf_matrix():
    from tools import plot_confusion_matrix

    y = [0, 0, 1, 20, 0, 0, 1, 2, 3, 4, 5, 6, 7, 12, 11, 14]
    y_hat = [0, 1, 1, 20, 0, 0, 1, 2, 9, 3, 4, 5, 10, 12, 11, 14]
    plot_confusion_matrix(y_hat, y, average="macro")


def test_scheduler():
    from schedulers import LRConsineAnnealingWithLinearWarmup
    import torch.optim as optim
    import torch

    optimizer = optim.Adam(torch.nn.Linear(10, 10).parameters(), lr=0.001)
    sched = LRConsineAnnealingWithLinearWarmup(
        optimizer, max_steps=10000, warmup_steps_ratio=0.03
    )
    print(sched.warmup_steps)
    print("done")


def test_logging():
    from tools import ansi_of, hex_of
    from tools import log, display_logs

    log("This is red", color="red")  # Entire text in red
    log("This is hex blue", color="#0000FF")  # Entire text in blue (hex)
    log(
        "This is a test message\n with a new line and some random characters:   "
    )  # Entire text in white
    log("This is hex green", color="#00FF00")  # Entire text in green (hex)
    log("Khaki hex", color="khaki")
    display_logs()


def test_rt_plotting():
    from tools import plot_rt_graph
    import matplotlib.pyplot as plt

    for i in range(200):
        plot_rt_graph(
            name="test_graph",
            value=[random.random() * 2 - 1, random.random()],
            step=None,
            x_label="Time",
            y_label="Price",
            color=("lightblue", "red"),
            linestyle="-",
            marker=".",
        )
    plt.show(block=True)
    # exit()
    # plot_rt_graph(
    #     name="test_graph",
    #     value=[random.random() * 10, random.random()],
    #     step = None,
    #     x_label="Time",
    #     y_label="Value",
    #     color=("lightblue", "yellow"),
    #     linestyle=["--", "-"],
    # )


def test_log_session():
    def example_train_function() -> float:
        import random

        time.sleep(random.random())
        return {"score": random.random() * 100}

    def example_train_function2() -> float:
        import random

        time.sleep(random.random())
        return {"acc": random.random() * 100, "f1": random.random() * 99}

    from flashml.tools import log_session

    hyperparams1 = {"learning_rate": 0.001, "dang": 32}
    log_session(hyperparams1, example_train_function, sort_by=None)

    hyperparams2 = {"learning_rate": 0.002, "batch_size": 16, "dropout": 0.2}
    log_session(hyperparams2, example_train_function, sort_by="Score")

    hyperparams3 = {"learning_rate": 0.003, "batch_size": 8}
    log_session(hyperparams3, example_train_function, sort_by="Score")

    hyperparams3 = {"learning_rate": 0.002, "batch_size": 8, "manag": "pick"}
    log_session(hyperparams3, example_train_function, sort_by="Score")

    log_session(None, None, sort_by="learning_rate")

    hyperparams4 = {"learning_rate": 0.001, "batch_size": 16, "dropout": 0.2}
    log_session(hyperparams4, example_train_function, sort_by="Score")

    hyperparams5 = {"learning_rate": 0.003, "batch_size": 8, "lavander": "lander"}
    log_session(hyperparams5, example_train_function, sort_by="Score")

    hyperparams5 = {"learning_rate": 0.003, "batch_size": 8, "lavander": "lander"}
    log_session(hyperparams5, example_train_function2)

    hyperparams6 = {"learning_rate": 0.003, "batch_size": 8}
    log_session(hyperparams6, example_train_function2, sort_by="acc")

    log_session(None, None, sort_by="f1")


from tools.nlp import *


def test_nlp_preprocesssing():
    # Polars
    import polars as pl

    df_pl = pl.DataFrame({"Name": ["Alice♥ ️🐈", "BOB", "Charlie"]})
    df_pl = lowercase(df_pl, "Name")
    print(df_pl)

    data = {
        "text": [
            "I have 2 dogs!.♥ ️🐈",
            "My phone number is 1234567890.",
            "The year is 2025.",
            "I bought 5 a♥ ️🐈pples and 3 bananas.",
            "The answer is 42.",
        ]
    }

    df = pl.DataFrame(data)
    df = replace_emojis(df, "text")
    df = lemmatize(df, "text")
    print(df)


def test_plot_dist():
    from tools import plot_distribution

    # Create a fake frequency dictionary with 100 token IDs
    fake_freqs = {i: random.randint(1, 1000) for i in range(1000)}

    # Run test plot
    plot_distribution(fake_freqs, sort_descending=True, top_n=1000)


def stress_test():
    stress_gpu()


def test_logger():
    from tools import log, load_logs

    print(load_logs(as_df="pl"))


def test_simple_for():
    lst = []
    for i in range(10000):
        lst.append(str(random.random()) * 10)

    for i in range(10000):
        lst[i] = sorted(lst[i])


def test_parallel_for():
    lst = []
    for i in range(10000):
        lst.append(str(random.random()) * 10)

    parallel_foreach(lst, sorted)


def plot_tensorx():
    import torch
    from tools import plot_tensor

    plot_tensor(torch.randn(42, 121))


def test_batch_generation():
    from tools import BatchIterator

    for x in BatchIterator(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], num_epochs=10, batch_size=3, mode="train"
    ):
        print(x)

    import polars as pl

    df = pl.DataFrame({"a": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]})
    for x in BatchIterator(df, num_epochs=10, batch_size=3, mode="train"):
        print(x)


def test_plot_tsne():
    import numpy as np
    from tools import plot_tsne

    data = np.random.rand(1200, 10)
    plot_tsne(data, verbose=1)


def log_metrics_test():
    from tools import log_metrics

    for i in range(100):
        x = random.random()

        if x < 0.3:
            log_metrics(
                {"loss": random.random(), "accuracy": random.random()},
                step=(i, 100),
                hyperparams={"lr": 0.001, "batch_size": 32},
            )
            i -= 3
        else:
            log_metrics(
                {"loss": random.random(), "f1": random.random()},
                step=(i, 100),
            )
        time.sleep(0.1)


from tools import resource_monitor

resource_monitor()
