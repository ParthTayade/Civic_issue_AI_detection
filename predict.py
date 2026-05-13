import torch
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models
import torch.nn as nn

classes = ["crack_road","garbage", "irrelevant","pothole_road","road_damage", "street_light"]

# Load model
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 6)
model.load_state_dict(torch.load("civic_model.pth", map_location="cpu"))
model.eval()

# Transform (FIXED)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def predict_image(img_path):
    image = Image.open(img_path).convert("RGB")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        conf, predicted = torch.max(probs, 1)

    return classes[predicted.item()], round(conf.item()*100, 2)

# Test
result, confidence = predict_image("test.jpg")
print("Prediction:", result)
print("Confidence:", confidence, "%")