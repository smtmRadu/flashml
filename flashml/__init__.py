###

### All scripts from the main_tools folder are imported here.
### So the user can directly: from flashml import *
### They are also fast to import.

###
from .main_tools.benchmark import benchmark, stress_cpu, stress_gpu
from .main_tools.colors import ansi_of, ansi_to_hex, hex_of, hex_to_ansi
from .main_tools.extern import call_cs_kernel
from .main_tools.logging_jsonl import load_jsonl, log_json
from .main_tools.lorem_ipsum import lorem_ipsum, lorem_ipsum_en
from .main_tools.manipulation import (
    BatchIterator,
    sample_from,
    sample_elementwise,
    shuffle_df,
    shuffle_tensor,
    reorder_columns_df
)
from .main_tools.mlflow_logging import (
    get_metrics_at_step,
    host_mlflow,
    load_checkpoint,
    log_checkpoint,
    log_figure,
    log_metrics,
)
from .main_tools.parallel import parallel_for, parallel_foreach
from .main_tools.plot_distribution import plot_dist
from .main_tools.plot_graph import plot_graph
from .main_tools.plot_rt_graph import plot_rt_graph
from .main_tools.plot_tensor import plot_tensor
from .main_tools.plot_tsne import plot_tsne
from .main_tools.resource_monitor import resource_monitor
from .main_tools.sound_effects import bell

__all__ = [
    "BatchIterator",
    "ansi_of",
    "ansi_to_hex",
    "bell",
    "benchmark",
    "call_cs_kernel",
    "get_metrics_at_step",
    "hex_of",
    "hex_to_ansi",
    "host_mlflow",
    "load_checkpoint",
    "load_jsonl",
    "log_checkpoint",
    "log_figure",
    "log_json",
    "log_metrics",
    "lorem_ipsum",
    "lorem_ipsum_en",
    "parallel_for",
    "parallel_foreach",
    "plot_dist",
    "plot_graph",
    "plot_rt_graph",
    "plot_tensor",
    "plot_tsne",
    "reorder_columns_df",
    "resource_monitor",
    "sample_elementwise",
    "sample_from",
    "shuffle_df",
    "shuffle_tensor",
    "stress_cpu",
    "stress_gpu",
]



assert __all__ == sorted(__all__), "Modules are not sorted alphabetically"
