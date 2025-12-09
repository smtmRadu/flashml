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
    sample_from,
    sample_elementwise,
    shuffle_df,
    shuffle_tensor,
    reorder_df_columns,
)
from .main_tools.batching import BatchIterator
from .main_tools.mlflow_logging import (
    get_metrics_at_step,
    host_mlflow,
    load_checkpoint,
    log_checkpoint,
    log_figure,
    log_metrics,
    make_run_name
)
from .main_tools.parallel import parallel_for, parallel_foreach
from .main_tools.plot_distribution import plot_dist
from .main_tools.plot_graph import plot_graph
from .main_tools.plot_rt_graph import plot_rt_graph
from .main_tools.plot_tensor import plot_tensor, plot_image_tensor
from .main_tools.plot_tsne import plot_tsne
from .main_tools.resource_monitor import resource_monitor
from .main_tools.sound_effects import bell
from .main_tools.load_yaml_configs import load_yaml_configs
from .main_tools.send_telegram_notification import send_telegram_notification

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
    "load_run_state",
    "load_yaml_configs",
    "log_checkpoint",
    "log_figure",
    "log_json",
    "log_metrics",
    "lorem_ipsum",
    "lorem_ipsum_en",
    "make_run_name",
    "parallel_for",
    "parallel_foreach",
    "plot_dist",
    "plot_graph",
    "plot_image_tensor",
    "plot_rt_graph",
    "plot_tensor",
    "plot_tsne",
    "reorder_df_columns",
    "resource_monitor",
    "sample_elementwise",
    "sample_from",
    "send_telegram_notification",
    "shuffle_df",
    "shuffle_tensor",
    "stress_cpu",
    "stress_gpu",
]



assert __all__ == sorted(__all__), "Modules are not sorted alphabetically"
