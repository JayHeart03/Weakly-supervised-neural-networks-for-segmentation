import torch


def mean_iou(predictions, targets, num_classes):
    """
    Calculate the mean IoU for a batch of segmentation masks.
    
    Args:
        predictions (torch.Tensor): Predicted masks of shape (B, H, W).
        targets (torch.Tensor): Ground truth masks of shape (B, H, W).
        num_classes (int): Number of classes.
    
    Returns:
        torch.Tensor: Mean IoU as a scalar.
    """
    # Flatten tensors
    pred_flat = predictions.view(-1)
    targ_flat = targets.view(-1)
    
    # Compute confusion matrix
    index = targ_flat * num_classes + pred_flat
    cm = torch.bincount(index, minlength=num_classes**2).reshape(num_classes, num_classes)
    
    # Calculate IoU
    intersection = torch.diag(cm).float()
    union = cm.sum(dim=1).float() + cm.sum(dim=0).float() - intersection
    iou = torch.where(union > 0, intersection / union, torch.zeros_like(intersection))
    
    return iou.mean()
