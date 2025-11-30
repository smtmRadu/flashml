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
from flashml import resource_monitor

resource_monitor()
```
<img src="https://github.com/smtmRadu/flashml/blob/main/doc/resource_monitor.jpg?raw=true" width="200" height="350" alt="Resource Monitor">

***

### Model/Tokenizer inspection
```python
from flashml import inspect_model, inspect_tokenizer

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
from flashml import plot_tensor
plot_tensor(torch.randn(10, 8, device: 'cuda:0'))
```
![tp](https://github.com/smtmRadu/flashml/blob/main/doc/tensor_plot.jpg?raw=true)
***

### LR Scheduler
```python
from flashml.schedulers import LRScheduler

scheduler = LRScheduler(
    optimizer=optim, 
    max_steps=1000, 
    warmup_steps=100, # linear warmup steps
    constant_steps=200, # constant lr steps
    min_lr=1e-8,
    annealing_type='cosine', # ['linear', 'exponential', 'logarithmic']
    num_cycles=3,
    max_lr_decay_factor=0.6, # decay of max lr after each cycle
)
```
![tp](https://github.com/smtmRadu/flashml/blob/main/doc/lr_scheduler.jpg?raw=true)
***
### Fast plot
```python
from flashml import plot_graph

plot_graph(([10, 15, 17, 18, 18.3], [8, 14, 17.1, 18.9, 19.02]))
```
***
### Real-Time plot
```python
from flashml import plot_rt_graph

for i in range(T):
    plot_rt_graph((<priceBTC>, <priceETH>), <timestep>, x_label="Time", y_label="Price", color=["yellow", "purple])
```

***
### Parallel For-Loops
```python
from flashml import parallel_for, parallel_foreach

def my_func(x):
    ...

result = parallel_for(0, 1e6, my_func, num_workers=16)
```
***
### ML train utilities (MLFlow)

```python
from flashml import log_metrics, BatchIterator,log_checkpoint, load_checkpoint
from flashml.classification import plot_confusion_matrix

for batch in BatchIterator(df=train_df, num_epochs=10, batch_size=32, mode="train"):
        # batch.value, batch.index, batch.step, batch.ids
        # Compute loss and perform update
        # Compute validation
        log_metrics(
            metrics={ "loss": loss , "acc" : acc},
            step=idx)

plot_confusion_matrix(yHat, y)
log_checkpoint({"model":model.state_dict(), "optim":optim.state_dict()}) # this will be logged in MLFlow

# later load with load_checkpoint(run_name="charming-seal-738", version=1)
```
```
Output (rt + MLFlow session log):

75%|████████████████████████████████████████████████████████████████████████                        | 225/300 [00:00<00:00, 2059.44it/s, loss=2.25, acc=5.17]

```
***
### Instant Baselines (MLFlow)
```python
from flashml.classification import make_dummy_classification_dataset, run_dummy_classifiers, run_linear_classifier

x, y, a, b = make_dummy_classification_dataset(num_samples=2048, num_features=10, num_classes=2, weights=[0.25, 0.75])

run_dummy_classifiers(x, y, a, b) # dummy classifications (uniform, prior, full zeros/ones, stratified)
run_linear_classifier(x, y, a, b, regularization="l1")
```


***

### RL train plots (MLFlow)

```python
from flashml.rl import log_episodes

 while step < max_steps:
    for i in range(buffer_size):
        # Step env, collect info
        if episode is done:
            log_episode(
                cumulative_reward=reward,
                episode_length=episode_len,
                step = (step, max_steps))

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
from flashml.nlp import *
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
from flashml import plot_tsne
import numpy as np

x1 = np.random.normal(loc=34, scale=3, size=(120, 99))
x2 = np.random.normal(loc=22, scale=5, size=(120, 99))
x3 = np.random.normal(loc=-32, scale=2, size=(200, 99))

x = np.concatenate([x1, x2, x3], axis=0)
plot_tsne(x, labels=["A"] * 120 + ["B"] * 120 + ["C"] * 200)
```
![tp](https://github.com/smtmRadu/flashml/blob/main/doc/plot_tsne.jpg?raw=true)
***
### Architectures common implementations
Enumerate **GQA**, **SwiGLU**, **FFN**, **MinGRU** ...
***
## Standalone scripts with GUI
- Color picker
- Image converter, resizer, processor etc.
- File converter
- Local chat (working with ollama)