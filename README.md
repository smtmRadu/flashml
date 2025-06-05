# FlashML

**FlashML** is a lightweight library (with minimal import overhead) dedicated to ML/RL/NLP engineers working with PyTorch. Not published yet on PyPI,
but you can install it by opening this project in VSCode, and with conda env call:

```
pip install -e .
```


and add this line to VSCode's settings.json:
```json
python.analysis.extraPath":["path\\to\\flashml"],
```
***
### Always-on resource monitor
```python
from flashml.tools import resource_monitor

resource_monitor()
```
<img src="https://github.com/smtmRadu/flashml/blob/main/doc/resource_monitor.jpg?raw=true" width="200" height="350" alt="Resource Monitor">

***

### Model inspection
```python
from flashml.tools import inspect_model, inspect_tokenizer

tokenizer = AutoTokenizer.from_pretrained("<any-huggingface-tokenizer>")
model = AutoModelForCausalLM.from_pretrained("<any-pytorch-model>")
inspect_model(model)
inspect_tokenizer(tokenizer)
```
![mi](https://github.com/smtmRadu/flashml/blob/main/doc/model_inspector.jpg?raw=true)
![mi](https://github.com/smtmRadu/flashml/blob/main/doc/tokenizer_inspector.jpg?raw=true)

***
### Tensor ploting
```python
from flashml.tools import plot_tensor
plot_tensor(torch.randn(42, 121))
```
![tp](https://github.com/smtmRadu/flashml/blob/main/doc/tensor_plot.jpg?raw=true)
***
### Fast plot
```python
from flashml.tools import plot_graph

plot_graph(([10, 15, 17, 18, 18.3], [8, 14, 17.1, 18.9, 19.02]))
```
***
### Real-Time plot
```python
from flashml.tools import plot_rt_graph

for i in range(T):
    plot_rt_graph((<priceBTC>, <priceETH>), <timestep>, x_label="Time", y_label="Price", color=["yellow", "purple])
```

***
### Parallel For-Loops
```python
from flashml.tools import parallel_for, parallel_foreach

def my_func(x):
    ...

result = parallel_for(0, 1e6, my_func, num_workers=16)
```
***
### ML train utilities

```python
from flashml.tools import log_metrics, generate_batches, display_metrics, plot_confusion_matrix

data = ...
batches = generate_batches(len(data), num_epochs=8, batch_size=32, mode="train")
for idx, batch_ids in enumerate(batches):
        batch = data[batch_ids]
        # Compute loss and perform update
        # Compute validation
        log_metrics(
            metrics={ "loss": loss , "acc" : acc},
            step=idx)

display_metrics() # Displays matplot graphs at the end, allowing data export as csv
plot_confusion_matrix(yHat, y)
```
```
Output (rt):

[Epoch 2/3]:  75%|████████████████████████████████████████████████████████████████████████                        | 225/300 [00:00<00:00, 2059.44it/s, loss=2.25, acc=5.17]

```
***

### RL train plots

```python
from flashml.tools.rl import log_episodes, display_episodes

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
Cumulated Reward [max: 3.236] [µ: 2.599] [σ: 0.30z]
Episode Length   [max: 685] [µ: 87.310] [σ: 73.73z]
[Episode 350]:  37%|███████████████████████████▎                                             | 37409/100000 [00:00<00:00, 112028.23it/s]

```
***
### NLP Preprocessing
```python
from flashml.tools.nlp import *
df = pl.read_csv(...)
df = lowercase(df, "text_col")
df = lemmatize(df, "text_col")
df = replace_emojis(df, "text_col", "E")
df = remove_double_spacing(df, "text_col")
... and other operations
```
***
### t-SNE visualization
```python
from flashml.tools import plot_tsne
import torch
data = torch.randn(1024, 40)
plot_tsne(data, mode='3d')
```
***
### Architectures common implementations
Enumerate **GQA**, **SwiGLU**, **FFN**, **MinGRU** ...
***
### Schedulers
Enumerate **LRCosineAnnealingWithLinearWarmup** ...
***
## Standalone scripts with GUI
- Color picker
- Image converter, resizer, processor etc.
- File converter
- Local chat (working with ollama)