import torch
from torchvision import transforms
from PIL import Image
import torchvision
import torch.nn as nn

classes = ["crack_road","garbage", "irrelevant","pothole_road","road_damage", "street_light"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = torchvision.models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 6)

model.load_state_dict(torch.load("civic_model.pth", map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)

    return classes[predicted.item()]

img_path = "test.jpg"

print("Prediction:", predict(img_path))