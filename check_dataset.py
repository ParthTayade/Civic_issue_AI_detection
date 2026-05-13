from torchvision import datasets

# Load training dataset
dataset = datasets.ImageFolder("dataset/train")

# Print class names
print("Classes found:", dataset.classes)

# Print number of classes
print("Total classes:", len(dataset.classes))