import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from tkinter import Tk, filedialog
import csv
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

def _export_values(predictions, targets):
    root = Tk()
    root.withdraw()

    file_path = filedialog.asksaveasfilename(
        defaultextension='.csv',
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )

    if file_path:
        with open(file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Predictions", "Targets"])
            
            for pred, targ in zip(predictions, targets):
                writer.writerow([pred, targ])
                    
        print(f"Data exported to {file_path}")


def _compute_confusion_matrix(pred, targ, labels):
    n_labels = len(labels)
    mapped_true = np.full_like(targ, fill_value=-1)
    mapped_pred = np.full_like(pred, fill_value=-1)

    for i, label in enumerate(labels):
        mapped_true[targ == label] = i
        mapped_pred[pred == label] = i
    
    valid = (mapped_true >= 0) & (mapped_pred >= 0)
    mapped_true = mapped_true[valid]
    mapped_pred = mapped_pred[valid]
    
    indices = mapped_true * n_labels + mapped_pred
    cm = np.bincount(indices.astype(np.int64), minlength=n_labels * n_labels).reshape(n_labels, n_labels)

    # cm = np.bincount(indices, minlength=n_labels * n_labels).reshape(n_labels, n_labels)
    return cm

def _flat(x) -> np.ndarray:
    if isinstance(x, (list, tuple)):
        return np.array(x).flatten()
    elif isinstance(x, np.ndarray):
        return x.flatten()
    else:
        raise ValueError("Unsupported input type.")

def plot_confusion_matrix(
        predicted: np.ndarray | list | tuple, 
        target: np.ndarray | list | tuple, 
        classes: list = None, 
        average: str = "binary",
        normalize: bool = False,
        title: str = "Confusion Matrix", 
        cmap: str = "Blues", 
        blocking: bool = True,
       ):
    '''
    Returns the accuracy, precision, recall, and f1 score of the given predicted and target values.''
    '''

    pred_array = _flat(predicted)
    target_array = _flat(target)

    unique_labels = np.unique(np.concatenate([target_array, pred_array]))
    
    if classes is None:       
        classes = [x for x in unique_labels]
        labels = unique_labels.tolist()
    else:
        assert len(classes) == len(unique_labels), f"Classes list length ({len(classes)}) doesn't equal the total labels found ({len(unique_labels)})."
        labels = list(range(len(classes)))

    classes = [str(x) for x in classes]
    cm = _compute_confusion_matrix(pred_array, target_array, labels)
    cm_raw = cm.copy()

    assert len(unique_labels) < 129, f"Too many classes, your pc will blow up {len(unique_labels)}"

    total = cm_raw.sum()
    accuracy = accuracy_score(y_true=target_array, y_pred=pred_array)
    precision = precision_score(y_true=target_array, y_pred=pred_array, average=average, zero_division=0)
    recall = recall_score(y_true=target_array, y_pred=pred_array, average=average, zero_division=0)
    f1 = f1_score(y_true=target_array, y_pred=pred_array, average=average, zero_division=0)

    if normalize:
        row_sum = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm.astype(float), row_sum, out=np.zeros_like(cm, dtype=float), where=row_sum != 0)

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.canvas.manager.set_window_title("flashml")
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    
    cbar = fig.colorbar(im, ax=ax)
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)

    ax.set_xlabel("Predictions")
    ax.set_ylabel("Targets")
    ax.set_title(title)
    
    # Add numbers to each cell
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f'{cm[i, j]:.2f}' if normalize else f'{int(cm[i, j])}',
                   ha='center', va='center',
                   color='white' if cm[i, j] > thresh else 'black')

    plt.tight_layout(rect=[0, 0.15, 1, 1])
    
    metrics_text = (
        f"Accuracy: {accuracy:.4f}\n"
        f"Precision ({average}): {precision:.4f}\n"
        f"Recall ({average}): {recall:.4f}\n"
        f"F1 Score ({average}): {f1:.4f}"
    )
    fig.text(0.5, 0.05, metrics_text, ha='center', fontsize=10)

    export_ax = plt.axes([0.85, 0.01, 0.12, 0.05])
    export_button = Button(export_ax, 'Export CSV')
    export_button.on_clicked(lambda event: _export_values(predicted, target))

    plt.show(block=blocking)
    return accuracy, precision, recall, f1