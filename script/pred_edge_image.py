
import os
import argparse
import glob
import numpy as np

import cv2

import torch

# Copy from https://raw.githubusercontent.com/Mikael17125/ViT-GradCAM/refs/heads/main/gradcam.py
class GradCam:
    def __init__(self, model, target):
        self.model = model.eval()  # Set the model to evaluation mode
        self.feature = None  # To store the features from the target layer
        self.gradient = None  # To store the gradients from the target layer
        self.handlers = []  # List to keep track of hooks
        self.target = target  # Target layer for Grad-CAM
        self._get_hook()  # Register hooks to the target layer

    # Hook to get features from the forward pass
    def _get_features_hook(self, module, input, output):
        self.feature = self.reshape_transform(output)  # Store and reshape the output features

    # Hook to get gradients from the backward pass
    def _get_grads_hook(self, module, input_grad, output_grad):
        self.gradient = self.reshape_transform(output_grad)  # Store and reshape the output gradients

        def _store_grad(grad):
            self.gradient = self.reshape_transform(grad)  # Store gradients for later use

        output_grad.register_hook(_store_grad)  # Register hook to store gradients

    # Register forward hooks to the target layer
    def _get_hook(self):
        self.target.register_forward_hook(self._get_features_hook)
        self.target.register_forward_hook(self._get_grads_hook)

    # Function to reshape the tensor for visualization
    def reshape_transform(self, tensor, height=14, width=14):
        result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
        result = result.transpose(2, 3).transpose(1, 2)  # Rearrange dimensions to (C, H, W)
        return result

    # Function to compute the Grad-CAM heatmap
    def __call__(self, inputs):
        self.model.zero_grad()  # Zero the gradients
        output = self.model(inputs)  # Forward pass

        # Get the index of the highest score in the output
        index = np.argmax(output.cpu().data.numpy())
        target = output[0][index]  # Get the target score
        target.backward()  # Backward pass to compute gradients

        # Get the gradients and features
        gradient = self.gradient[0].cpu().data.numpy()
        weight = np.mean(gradient, axis=(1, 2))  # Average the gradients
        feature = self.feature[0].cpu().data.numpy()

        # Compute the weighted sum of the features
        cam = feature * weight[:, np.newaxis, np.newaxis]
        cam = np.sum(cam, axis=0)  # Sum over the channels
        cam = np.maximum(cam, 0)  # Apply ReLU to remove negative values

        # Normalize the heatmap
        cam -= np.min(cam)
        cam /= np.max(cam)
        cam = cv2.resize(cam, (224, 224))  # Resize to match the input image size
        return cam  # Return the Grad-CAM heatmap


if __name__ ==  '__main__':
    parser = argparse.ArgumentParser(
        prog='',  # プログラム名
        usage='',  # プログラムの利用方法
        add_help=True,  # -h/–help オプションの追加
    )
    parser.add_argument('--testdir', type=str, default='dataset/test')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--modelfile', type=str)
    parser.add_argument('--inputsize', type=int, default=224)
    
    args = parser.parse_args()

    device = args.device
    model = torch.load(args.modelfile, map_location=device)
    
    target_layer = model.blocks[-1].norm1  # Specify the target layer for Grad-CAM
    # Initialize Grad-CAM with the model and target layer
    grad_cam = GradCam(model, target_layer)
    
    for test_idx, testfile in enumerate(sorted(glob.glob(os.path.join(args.testdir, '*png'))+glob.glob(os.path.join(args.testdir, '*jpeg'))+glob.glob(os.path.join(args.testdir, '*jpg')))):
        test_img = cv2.imread(testfile)
        if np.mean(test_img) > 128:
            test_img = 255-test_img
        test_input = torch.Tensor(np.transpose(cv2.resize(test_img, (args.inputsize, args.inputsize))/255, (2,0,1))[np.newaxis]).to(device)
        print(testfile, model(test_input).detach().cpu().softmax(dim=1).numpy())    

        mask = grad_cam(test_input) 
        mask_uint8 = (mask*255).astype(np.uint8)
        heatmap = cv2.applyColorMap(mask_uint8, cv2.COLORMAP_JET)
        result_img = cv2.resize(cv2.imread(testfile), (args.inputsize, args.inputsize))//2+heatmap//2
        cv2.imwrite("cam_{}.png".format(test_idx), result_img)