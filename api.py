from flask import Flask, request, render_template, send_from_directory
from flask_cors import CORS
import torch
from torchvision import transforms, models
from PIL import Image
import os



CLASS_INFO = {
    "crack_road": "Cracks are visible on the road surface. These cracks indicate structural weakening and may develop into potholes if not repaired.",

    "pothole_road": "A pothole is detected on the road. This is a deep cavity formed due to surface wear and water damage, posing a risk to vehicles and pedestrians.",

    "garbage": "Garbage is present on the road or public area, which can affect cleanliness and hygiene.",

    "street_light": "Street light appears to be non-functional or damaged, reducing visibility and safety during night time.",

    "road_damage": "Road is damaged or has potholes.",

    "irrelevant": "No civic issue detected in the image."
}

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model safely
MODEL_PATH = "civic_model.pth"

model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 6)

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
else:
    print("⚠️ Model not found. Running without model.")
    model = None

classes = ["crack_road","garbage", "irrelevant","pothole_road","road_damage", "street_light"]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
])

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    confidence = None
    filename = None
    description = None

    if request.method == 'POST':
        file = request.files.get('file')

        if file and file.filename != "":
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            img = Image.open(filepath).convert('RGB')
            img = transform(img).unsqueeze(0)

            if model is not None:
                with torch.no_grad():
                    outputs = model(img)
                    probs = torch.nn.functional.softmax(outputs, dim=1)
                    conf, pred = torch.max(probs, 1)

                prediction = classes[pred.item()]
                confidence = round(conf.item() * 100, 2)

                if confidence < 60:
                         description = "Low confidence prediction. Please upload clearer image."
                else:
                         if confidence < 60:
                            description = "Low confidence prediction. Please upload a clearer image."

                         else:
                            if prediction == "road_damage":
                                description = "The road shows visible structural damage such as surface cracks or small potholes that may affect vehicle movement and safety."

                            elif prediction == "crack_road":
                                 description = "Cracks are visible on the road surface. These may expand over time and lead to serious road damage."

                            elif prediction == "pothole_road":
                                  description = "Pothole detected on the road. This can cause vehicle damage and is a safety hazard."


                            elif prediction == "street_light":
                                description = "Street light is not functioning properly. The light appears to be off or damaged, causing reduced visibility in the area."

                            elif prediction == "garbage":
                                 description = "Garbage is present in the area, including waste materials dumped on roads or public space, which may affect hygiene and cleanliness."

                            elif prediction == "irrelevant":
                                 description = "No civic issue detected in the image. The content does not relate to public infrastructure or civic problems."

                        
                            else:
                                 description = "Civic issue detected but cannot be clearly classified."

            else:
                prediction = "Model not loaded"
                confidence = 0
                description = "Model not loaded"

            filename = file.filename

    return render_template("index.html",
                       prediction=prediction,
                       confidence=confidence,
                       description=description,
                       filename=filename)

if __name__ == "__main__":
    print("🚀 Starting Flask...")
    app.run(host="127.0.0.1", port=7063, debug=False)