import torch
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from tkinter import Tk, filedialog
import csv
from collections.abc import Iterable

def _export_values(x_label: str, y_label: str, x_ticks, values):
    root = Tk()
    root.withdraw()

    file_path = filedialog.asksaveasfilename(
        defaultextension='.csv',
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )

    if file_path:
        with open(file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            if isinstance(values[0], Iterable):
                header = [x_label] + [f"{y_label}_{i+1}" for i in range(len(values))]
                writer.writerow(header)
                for i in range(len(x_ticks)):
                    row = [x_ticks[i]] + [series[i] for series in values]
                    writer.writerow(row)
            else:
                writer.writerow([x_label, y_label])
                for x, y in zip(x_ticks, values):
                    writer.writerow([x, y])
                    
        print(f"Data exported to {file_path}")

def _compute_confusion_matrix(pred, targ,labels):
        n_labels = len(labels)
        mapped_true = torch.full_like(targ, fill_value=-1)
        mapped_pred = torch.full_like(pred, fill_value=-1)

        for i, label in enumerate(labels):
            mapped_true[targ == label] = i
            mapped_pred[pred == label] = i
        valid = (mapped_true >= 0) & (mapped_pred >= 0)
        mapped_true = mapped_true[valid]
        mapped_pred = mapped_pred[valid]
        
        indices = mapped_true * n_labels + mapped_pred
        cm = torch.bincount(indices, minlength=n_labels * n_labels).reshape(n_labels, n_labels)
        return cm

def _flat(x) -> torch.Tensor:
    if isinstance(x, (list, tuple)):
        return torch.tensor(x)
    elif isinstance(x, torch.Tensor):
        return x.view(-1)
    else:
        raise "Wtf is this."
      
def plot_confusion_matrix(
        predicted: torch.Tensor | list | tuple, 
        target: torch.Tensor | list | tuple, 
        classes: list[str] = None, 
        normalize: bool = False,
        title: str = "Confusion Matrix", 
        cmap: str = "Blues", 
        blocking: bool = True) -> None:
    """
    Plot a confusion matrix from predicted and target labels. The inputs are only
    the indices of the prediction and targets.
    
    Args:
        predicted: Predicted labels (torch.Tensor, numpy array, or list/tuple of them).
        target: Ground truth labels (torch.Tensor, numpy array, or list/tuple of them).
        classes (list): List of class names. If None, they will be derived from the data.
        normalize (bool): If True, each row of the confusion matrix is normalized.
        title (str): Title of the plot.
        cmap (str): Colormap used for the heatmap.
        blocking (bool): If True, plt.show() will block execution.
    """

    GRAY_HEX = "#24292e"
    pred_tensor = _flat(predicted)
    target_tensor = _flat(target)

    unique_labels = torch.unique(torch.cat([target_tensor, pred_tensor]))
    
    if classes is None:       
        classes = [str(x.item()) for x in unique_labels]
        labels = unique_labels.tolist()
    else:
        assert len(classes) == len(unique_labels), "Classes list length doesn't equal the total labels found"
        labels = list(range(len(classes)))

    cm = _compute_confusion_matrix(pred_tensor, target_tensor, labels)
    cm_raw = cm.clone()  

    assert len(unique_labels) < 129, f"Too many classes, your pc will blow up {len(unique_labels)}"

    total = cm_raw.sum().item()
    accuracy = torch.diag(cm_raw).sum().item() / total if total > 0 else 0.0

    precision_list = []
    recall_list = []
    f1_list = []
    num_classes = cm_raw.shape[0]
    for i in range(num_classes):
        tp = cm_raw[i, i].item()
        fp = cm_raw[:, i].sum().item() - tp
        fn = cm_raw[i, :].sum().item() - tp

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        precision_list.append(prec)
        recall_list.append(rec)
        f1_list.append(f1_score)

    precision_avg = sum(precision_list) / num_classes if num_classes > 0 else 0.0
    recall_avg = sum(recall_list) / num_classes if num_classes > 0 else 0.0
    f1_avg = sum(f1_list) / num_classes if num_classes > 0 else 0.0

    if normalize:
        row_sum = cm.float().sum(dim=1, keepdim=True)
        cm = torch.where(row_sum == 0, torch.zeros_like(cm), cm / row_sum)

     
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor(GRAY_HEX ) 
    ax.set_facecolor(GRAY_HEX )         
    ax.tick_params(axis='both', colors='white', labelcolor='white')
    ax.spines["top"].set_color('white')
    ax.spines["bottom"].set_color('white')
    ax.spines["left"].set_color('white')
    ax.spines["right"].set_color('white')
    fig.canvas.manager.set_window_title("flashml")
    cm_np = cm.cpu().numpy() if cm.is_cuda else cm.numpy()
    im = ax.imshow(cm_np, interpolation='nearest', cmap=cmap)
    
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.tick_params(axis='both', colors='white', labelcolor='white')
    cbar.ax.spines["top"].set_color('white')
    cbar.ax.spines["bottom"].set_color('white')  
    cbar.ax.spines['left'].set_color('white') 
    cbar.ax.spines['right'].set_color('white') 

    tick_marks = torch.arange(len(classes)).cpu().numpy()
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, rotation=45, ha="right", color='white')
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, color='white')

    ax.set_xlabel("Predictions", color='white')
    ax.set_ylabel("Targets", color='white')
    ax.set_title(title, color='white')
    plt.tight_layout(rect=[0, 0.15, 1, 1])
    
    metrics_text = (
        f"Accuracy: {accuracy:.4f}\n"
        f"Precision (macro): {precision_avg:.4f}\n"
        f"Recall (macro): {recall_avg:.4f}\n"
        f"F1 Score (macro): {f1_avg:.4f}"
    )
    fig.text(0.5, 0.05, metrics_text, ha='center', fontsize=10, color='white')

    export_ax = plt.axes([0.85, 0.01, 0.12, 0.05])
    export_button = Button(export_ax, 'Export CSV')
    export_button.on_clicked(lambda event: _export_values("Predictions", "Targets", classes, cm.cpu().tolist()))

    plt.show(block=blocking)

