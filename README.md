# KOA-Prediction  

## Project Overview  
KOA-Prediction is an AI-powered system for automated detection and severity classification of Knee Osteoarthritis (KOA) using deep learning and computer vision.  
It processes X-ray images to classify KOA into five severity levels and provides fast, reliable diagnostics while reducing dependency on manual grading.  

## Features  
- Deep Learning Model (ResNet-50) for accurate KOA severity classification  
- Pseudo-labeling for improved semi-supervised learning  
- FastAPI Web Interface for real-time predictions  
- Treatment Recommendations based on severity  

---

## Installation Guide  

### Prerequisites  
- Python 3.8+  
- Virtual environment (recommended)  
- Dependencies from `requirements.txt`  

### Steps to Install and Run  

#### 1. Clone the repository:  
```bash
git clone https://github.com/your-username/KOA-Prediction.git
cd KOA-Prediction
```

#### 2. Set up a virtual environment:  

For Windows:  
```bash
python -m venv env
env\Scripts\activate
```

For Linux/macOS:  
```bash
python -m venv env
source env/bin/activate
```

#### 3. Install dependencies:  
```bash
pip install -r requirements.txt
```

#### 4. Run the application:  
```bash
uvicorn app.main:app --reload
```

#### 5. Access the Web Interface:  
Open your browser and visit:  
**http://127.0.0.1:8000**  

---

## Model Training & Evaluation  

### Training  
- Dataset Size: 15,000+ labeled X-ray images  
- Architecture: ResNet-50  
- Training Strategy:  
  - Supervised training on labeled data  
  - Pseudo-labeling for improved performance on unlabeled data  
  - Data augmentation to prevent overfitting  
- Evaluation Metrics: Accuracy, Precision, Recall, F1-score  

### Running Model Evaluation  
After training is complete, evaluate the model using:  
```bash
python evaluate.py
```

---

## How It Works  
1. Upload an X-ray Image through the Web UI  
2. AI Classifies KOA Severity into one of five categories:  
   - Normal  
   - Doubtful  
   - Mild  
   - Moderate  
   - Severe  
3. Get Treatment Recommendations based on severity  

---

## Future Enhancements  
- Multi-modal imaging support (MRI & CT scans)  
- Mobile App integration (Android/iOS)  
- Clinical validation in hospitals  

---

## Contributions  
We welcome contributions! Follow these steps:  

1. **Fork the repository**  
2. **Create a feature branch:**  
   ```bash
   git checkout -b feature-name
   ```  
3. **Commit changes:**  
   ```bash
   git commit -m "Added new feature"
   ```  
4. **Push to GitHub:**  
   ```bash
   git push origin feature-name
   ```  
5. **Open a Pull Request**  

---

## Contact  
- **GitHub Repository:** [KOA-Prediction]((https://github.com/Vachu0910/KOA-Prediction))  
- **Email:** varshith.0910@gmail.com

---

