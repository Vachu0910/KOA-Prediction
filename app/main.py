import os
import io
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torchvision.models as models
from torchvision.models import ResNet50_Weights
# Define severity levels and corresponding colors
severity_levels = ["Normal", "Doubtful", "Mild", "Moderate", "Severe"]
severity_solutions = {
    "Normal": "No signs of osteoarthritis. Maintain a healthy lifestyle.",
    "Doubtful": "Mild symptoms detected. Regular exercise and diet control advised.",
    "Mild": "Consider physiotherapy and mild pain relievers.",
    "Moderate": "Consult a doctor for potential medication and lifestyle adjustments.",
    "Severe": "Immediate medical attention required. Surgery may be necessary."
}
severity_colors = {
    "Normal": (0, 255, 0, 100),       # Green
    "Doubtful": (0, 255, 255, 100),   # Cyan
    "Mild": (255, 255, 0, 100),       # Yellow
    "Moderate": (255, 165, 0, 100),   # Orange
    "Severe": (255, 0, 0, 100)        # Red
}

# Image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Model definition
class ACHLModel(nn.Module):
    def __init__(self, num_classes=5):
        super(ACHLModel, self).__init__()
        self.base_model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.base_model.fc = nn.Identity()  # Keep backbone intact
        self.fc_contrastive = nn.Linear(2048, 128)
        self.fc_class = nn.Linear(128, num_classes)

    def forward(self, x):
        features = self.base_model(x)
        contrastive_features = self.fc_contrastive(features)
        contrastive_features = F.normalize(contrastive_features, p=2, dim=1)
        logits = self.fc_class(contrastive_features)
        return logits


# Load Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ACHLModel(num_classes=5).to(device)

try:
    model.load_state_dict(torch.load("app/koa1.pth", map_location=device))
    model.eval()
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {str(e)}")



# Initialize FastAPI
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read and preprocess image
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)

        # Get prediction
        with torch.no_grad():
            output = model(image_tensor)
            predicted_class = torch.argmax(output, dim=1).item()

        prediction_label = severity_levels[predicted_class]
        solution_text = severity_solutions[prediction_label]

        # Apply severity-based coloring
        overlay_color = severity_colors[prediction_label]
        image_resized = image.resize((224, 224))
        overlay = Image.new("RGBA", image_resized.size, overlay_color)
        blended = Image.blend(image_resized.convert("RGBA"), overlay, alpha=0.4)

        # Save processed image
        unique_suffix = str(int(time.time()))
        severity_image_filename = f"severity_{unique_suffix}.png"
        severity_image_path = os.path.join("static", severity_image_filename)
        blended.convert("RGB").save(severity_image_path)

        return JSONResponse(content={
            "prediction": prediction_label,
            "solution": solution_text,
            "severity_image": f"/static/{severity_image_filename}"
        })

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
