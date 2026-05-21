import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
from train_unet import UNetClassifier

# Initialize Grad-CAM components
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Hooks to capture gradients and activations
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x, class_idx=None):
        self.model.eval()
        
        # Forward pass
        output = self.model(x)
        
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()
            
        # Backward pass
        self.model.zero_grad()
        score = output[0, class_idx]
        score.backward(retain_graph=True)
        
        # Get gradients and activations
        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]
        
        # Pool the gradients across spatial dimensions
        weights = np.mean(gradients, axis=(1, 2))
        
        # Weight the activations
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
            
        # ReLU and normalize
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam, class_idx

def visualize_gradcam(image_path, cam, output_path, predicted_class):
    # Read original image
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    
    # Convert heatmap to RGB
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    
    # Overlay heatmap on original image
    overlay = cv2.addWeighted(img, 0.5, heatmap, 0.5, 0)
    
    # Plotting
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Original MRI")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.title(f"Detection Boundary (Pred: {predicted_class})")
    plt.axis('off')
    
    plt.savefig(output_path)
    print(f"Saved visualization to {output_path}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Classes
    data_dir = 'BrainMRI/Testing'
    classes = sorted(os.listdir(data_dir))
    
    # Load Model
    model = UNetClassifier(n_channels=3, n_classes=len(classes)).to(device)
    model.load_state_dict(torch.load('unet_brain_mri.pth'))
    
    # Initialize GradCAM (Targeting the bottleneck layer of U-Net)
    grad_cam = GradCAM(model, model.down4)
    
    # Transform for the model
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Pick a few sample images to visualize
    samples = [
        os.path.join(data_dir, 'glioma', os.listdir(os.path.join(data_dir, 'glioma'))[0]),
        os.path.join(data_dir, 'meningioma', os.listdir(os.path.join(data_dir, 'meningioma'))[0]),
        os.path.join(data_dir, 'pituitary', os.listdir(os.path.join(data_dir, 'pituitary'))[0])
    ]
    
    for i, img_path in enumerate(samples):
        # Load and transform image
        from PIL import Image
        pil_img = Image.open(img_path).convert('RGB')
        input_tensor = transform(pil_img).unsqueeze(0).to(device)
        
        # Get Heatmap
        cam, pred_idx = grad_cam(input_tensor)
        pred_class = classes[pred_idx]
        
        # Save Visualization
        output_file = f"heatmap_result_{i+1}_{pred_class}.png"
        visualize_gradcam(img_path, cam, output_file, pred_class)
