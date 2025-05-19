import torch


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
