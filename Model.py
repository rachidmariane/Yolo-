import torch
import numpy as np
import os
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import kagglehub
import json 
# Download latest version
path = kagglehub.dataset_download("kmader/malaria-bounding-boxes")

print("Path to dataset files:", path)


class YOLOModel(nn.Module):
    def __init__(self, num_classes, num_anchors):
        super(YOLOModel, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        # Example layers (you can expand this based on your YOLO architecture)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.fc = nn.Linear(128 * 7 * 7, num_anchors * (5 + num_classes))  # Example for YOLO output

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x

# Example usage
if __name__ == "__main__":
    num_classes = 80  # Example for COCO dataset
    num_anchors = 3
    model = YOLOModel(num_classes, num_anchors)
    dummy_input = torch.randn(1, 3, 224, 224)  # Example input
    output = model(dummy_input)
    print(output.shape)


    class ImageDataset(torch.utils.data.Dataset):
        def __init__(self, image_dir, transform=None):
            self.image_dir = image_dir
            self.image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
            self.transform = transform

        def __len__(self):
            return len(self.image_files)

        def __getitem__(self, idx):
            img_path = os.path.join(self.image_dir, self.image_files[idx])
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, self.image_files[idx]

    # Example usage
    if __name__ == "__main__":
        image_dir = "path/to/your/images"  # Replace with your image directory
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        dataset = ImageDataset(image_dir, transform=transform)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

        for images, filenames in dataloader:
            outputs = model(images)
            # Save outputs to a dataset (example: save as numpy arrays)
            for i, filename in enumerate(filenames):
                output_path = os.path.join("path/to/save/outputs", f"{filename}.npy")
                np.save(output_path, outputs[i].detach().numpy())