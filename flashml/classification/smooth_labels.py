def smooth_labels(targets, epsilon=0.1):
    """_summary_

    Args:
        targets (torch.Tensor/ndarray): Tensor of shape (*, num_classes)
        epsilon (float, optional): Value of epsilon for label smoothing. Defaults to 0.1.

    Examples:
    E.g. if epsilon = 0.1, then (0, 1) will be (0.05, 0.95)
    E.g. if epsilon = 0.1, then (1, 0, 0) will be (0.9333, 0.03333, 0.3333)
    """
    return (1 - epsilon) * targets + epsilon / targets.shape[-1]
