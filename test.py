from info import show_memory_usage
import time
import random
import math
show_memory_usage()
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


from info import log_metrics, display_metrics
loss = 10.0

epochs = 3
batches = 300

for it in range(1):
    for epoch in range(epochs):
        for batch in range(batches):
            loss_ =  math.log2(abs(loss))
            acc = loss + random.random()
            log_metrics(
                name=("loss", "val", "lr"),
                value=(loss_, acc, random.random()),
                epoch_idx=(epoch, epochs),
                batch_idx=(batch, batches),
                retain_logs=True
            )
            loss -= 1e-2
    print('\n\n\n')
        
display_metrics()