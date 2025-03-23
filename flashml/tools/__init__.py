from .resource_monitor import resource_monitor
from .log_metrics import log_metrics, log_metrics2
from .display_metrics import display_metrics
from .plot_graph import plot_graph
from .benchmark import benchmark
from .plot_confusion_matrix import plot_confusion_matrix
from .data_operations import batch_ranges, shuffle_df
from .log import log, display_logs
from .plot_rt_graph import plot_rt_graph
from .colors import ansi_of, hex_of

# A LITTLE NOTE ON DISPLAY METRICS
# I created a new log_metrics2 that might be used as well for reinforcement learning, that allows logging 
# based only on the step (and excepts batch and epoch). `display_metrics` is used for the same thing.
#
__all__ = [
    "ansi_of",
    "batch_ranges",
    "benchmark",
    "display_logs",
    "display_metrics",
    "hex_of",
    "log",
    "log_metrics",
    "log_metrics2",
    "plot_confusion_matrix"
    "plot_graph",
    "plot_rt_graph",
    "resource_monitor",
    "shuffle_df",     
    ]

assert __all__ == sorted(__all__)