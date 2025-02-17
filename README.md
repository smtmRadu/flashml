# flashml

**flashml** is a third party tool used in certain ML related tasks.


### Always-on resource monitor
```python
from flashml.info import resource_monitor

resource_monitor()
```

### Fast plot
```python
from flashml.info import plot_graph

plot_graph(([10, 15, 17, 18, 18.3], [8, 14, 17.1, 18.9, 19.02]))
```

### ML train graphs

```python
from flashml.info import log_metrics, display_metrics

for ep in range(epochs):
    for idx, batch in enumerate(batches):    
        # Compute loss and perform update
        # Compute validation
        log_metrics(
            name=("loss", "accuracy"),
            value=(loss_, acc),
            epoch_idx=(ep, epochs),
            batch_idx=(idx, batches))

display_metrics() # Displays matplot graphs at the end, allowing data export as csv
```

```
Output (rt):

[Epoch 2/3]:  75%|████████████████████████████████████████████████████████████████████████                        | 225/300 [00:00<00:00, 2059.44it/s, loss=2.25, acc=5.17]

```


### RL train graphs

```python
from flashml.info.rl import log_episodes, display_episodes

 while step < max_steps:
    for i in range(buffer_size):
        # Step env, collect info
        if episode is done:
            log_episode(
                cumulative_reward=reward,
                episode_length=episode_len,
                step = (step, max_steps))

display_episodes() # Displays matplot graphs at the end (rewards vs step/epoch, ep_len vs step/epoch), allowing data export as csv
```

```
Output (rt):

[Episode 350]:  37%|███████████████████████████▎                                             | 37409/100000 [00:00<00:00, 112028.23it/s, r_max=188, r_avg=160, r_std=14.6]

```

### Architectures common implementations
Enumerate **GQA**, **SwiGLU**, **FFN**, **MinGRU** ...

### Schedulers
Enumerate **LRCosineAnnealingWithLinearWarmup** ...