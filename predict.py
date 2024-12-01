import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

# Define the colorization model
class ColorizationNet(nn.Module):
    def __init__(self):
        super(ColorizationNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=4, dilation=2)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=4, dilation=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=4, dilation=2)
        self.conv4 = nn.Conv2d(128, 3, kernel_size=5, stride=1, padding=4, dilation=2)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.relu(self.conv3(x))
        x = torch.sigmoid(self.conv4(x))
        return x

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ColorizationNet().to(device)
model.load_state_dict(torch.load("colorization_model.pth"))
model.eval()  # Set model to evaluation mode

# Function to select file using a file dialog
def select_file():
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename(title="Select an Image")
    return file_path

# Select the image file using the file dialog
image_path = select_file()

if image_path:
    # Open the selected image
    img = Image.open(image_path)

    # Convert the image to grayscale
    gray_img = img.convert("L")

    # Define the transformations for the input image
    transform = transforms.Compose([transforms.ToTensor()])

    # Apply transformations to the grayscale image
    img_tensor = transform(gray_img).unsqueeze(0).to(device)

    # Get the model's output
    with torch.no_grad():
        colorized_tensor = model(img_tensor)

    # Convert tensor back to image
    colorized_img = transforms.ToPILImage()(colorized_tensor.squeeze(0).cpu())

    # Optionally, save the colorized image
    colorized_img.save("colorized_output.png")

    # Display original and colorized images
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))  # Create a figure with 1 row and 3 columns

    # Display original color image
    ax[0].imshow(img)
    ax[0].set_title("Original Color Image")
    ax[0].axis('off')  # Hide axes

    # Display grayscale image
    ax[1].imshow(gray_img, cmap='gray')  # Since it's grayscale, use cmap='gray'
    ax[1].set_title("Grayscale Image")
    ax[1].axis('off')  # Hide axes

    # Display colorized image
    ax[2].imshow(colorized_img)
    ax[2].set_title("Colorized Image")
    ax[2].axis('off')  # Hide axes

    plt.tight_layout()  # Adjust spacing
    plt.show()

else:
    print("No file selected!")
