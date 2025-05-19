import torch
import torch.nn.functional as F
import numpy as np


class GradCAMPlusPlus:
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers

    def __call__(self, input_tensor, targets=None):
        return self.forward(input_tensor, targets)

    def forward(self, input_tensor, targets=None):
        """
        Perform forward and backward passes to compute the GradCAM++ heatmap.

        Args:
            input_tensor (torch.Tensor): Input image tensor with shape (N, 3, H, W).
            targets (optional): Target class index. If None, the predicted class is used.

        Returns:
            list: A list containing the computed CAM as a numpy array (typically for a single sample).
        """
        self.model.eval()
        features = {}
        gradients = {}

        # Select the target layer
        target_layer = self.target_layers[0]

        # Define a forward hook to capture the output (feature map) of the target layer.
        def forward_hook(module, input, output):
            features["value"] = output
        # Define a backward hook to capture the gradients flowing out of the target layer.
        def backward_hook(module, grad_in, grad_out):
            gradients["value"] = grad_out[0]

        # Register the hooks on the target layer.
        hook_forward = target_layer.register_forward_hook(forward_hook)
        hook_backward = target_layer.register_backward_hook(backward_hook)

        # Perform a forward pass through the model.
        outputs = self.model(input_tensor)
        if targets is None:
            target_class = outputs.argmax(dim=1).item()
        else:
            target_class = targets[0] if isinstance(targets, list) else targets

        one_hot = torch.zeros_like(outputs)
        one_hot[0, target_class] = 1
        self.model.zero_grad()
        outputs.backward(gradient=one_hot, retain_graph=True)

        feature_maps = features["value"].detach()
        grads = gradients["value"].detach()
        
        # Compute the squared and cubed gradients for GradCAM++.
        grads_2 = grads ** 2
        grads_3 = grads ** 3

        # GradCAM++
        denominator = 2 * grads_2 + feature_maps * grads_3
        denominator = torch.where(denominator != 0, denominator, torch.ones_like(denominator))
        
        # Compute the local weight alpha.
        alpha = grads_2 / denominator
        
        # Compute the channel-wise weights by summing (alpha * ReLU(gradients)) over the spatial dimensions.
        weights = torch.sum(alpha * torch.relu(grads), dim=(2, 3))[0]
        
        # Initialize an empty CAM (heatmap) with the same spatial dimensions as the feature map.
        cam = torch.zeros(feature_maps.shape[2:], dtype=torch.float32, device=feature_maps.device)
        for i, w in enumerate(weights):
            cam += w * feature_maps[0, i, :, :]

        # Apply ReLU to keep only positive values and normalize the CAM to [0, 1].
        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        hook_forward.remove()
        hook_backward.remove()

        return [cam.cpu().numpy()]


def generate_multiscale_cam(
    model,
    image_tensor,
    target_layer,
    scales=[64, 112, 224, 320, 448],
    target_size=(256,256)
    ) -> np.ndarray:
    """
    Generate a multi-scale GradCAM++ heatmap by computing CAMs at different scales and fusing them.

    Args:
        model (torch.nn.Module): The trained neural network model.
        image_tensor (torch.Tensor): The input image tensor, typically of shape (1, 3, H, W).
        target_layer: The target layer for generating CAM (e.g., model.layer4).
        scales (list): List of scale sizes (resized image dimensions) to compute CAMs.
        target_size (tuple): The desired output heatmap size (H, W) after upsampling.

    Returns:
        np.ndarray: The fused and normalized CAM as a numpy array with shape target_size.
    """
    cam_list = []
    device = image_tensor.device
    
    for s in scales:
        resized_img = F.interpolate(image_tensor, size=(s, s), mode='bicubic', align_corners=False)
        cam_extractor = GradCAMPlusPlus(model=model, target_layers=[target_layer])
        cam_np = cam_extractor(input_tensor=resized_img, targets=None)[0]
        cam_tensor = torch.from_numpy(cam_np).unsqueeze(0).unsqueeze(0).float().to(device)
        cam_upsampled = F.interpolate(cam_tensor, size=target_size, mode='bicubic', align_corners=False)
        cam_list.append(cam_upsampled.squeeze(0).squeeze(0))
        
    fused_cam_tensor = torch.stack(cam_list, dim=0).mean(dim=0)
    fused_cam_tensor = fused_cam_tensor - fused_cam_tensor.min()
    fused_cam_tensor = fused_cam_tensor / (fused_cam_tensor.max() + 1e-8)
    return fused_cam_tensor.cpu().numpy()


def threshold_cam_three(cam, high_threshold=0.6, low_threshold=0.4) -> np.ndarray:
    """
    Threshold the CAM to create a multi-class segmentation map.

    Args:
        cam (np.ndarray): The CAM to be thresholded.
        high_threshold (float): The high threshold for the CAM.
        low_threshold (float): The low threshold for the CAM.
    Returns:
        np.ndarray: The thresholded CAM as a multi-class segmentation map.
    """
    labels = np.zeros_like(cam, dtype=np.uint8)
    labels[cam >= high_threshold] = 2 
    labels[(cam >= low_threshold) & (cam < high_threshold)] = 1 
    return labels
