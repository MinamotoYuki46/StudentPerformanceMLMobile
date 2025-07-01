# StudentPerformanceMLMobile  
**Project Name: Intellectus Wert**  
*"Intellectus Wert"* combines Latin and German, meaning **"Intelligence Value"** or **"Understanding Index"**. It symbolises a smart, data-driven way to assess and understand student performance.

## Overview  
This is my final project for Mobile Programming.  
It is an Android application integrated with **Lightweight Deep Learning (LWDL)** via a FastAPI backend to predict a **Student Performance Index**.

The prediction is based on a set of simple, real-life academic and behavioural inputs.

---

## Dataset & Features  
The model uses a regression approach trained on a student dataset from Kaggle.

### **Independent Variables (Inputs):**
- Hours Studied  
- Previous Score  
- Participation in Extracurricular Activities  
- Number of Sample Papers Practised  
- Sleep Hours

### **Dependent Variable (Output):**
- **Performance Index** (numeric score prediction)

---

## Python Backend (API)  
The backend is built using **Python 3.12**, **TensorFlow**, and **FastAPI**.

### Setup Instructions

1. Navigate to the API directory:
```bash
cd MeineAPIundML
```

2. Activate the virtual environment:

### For **Bash** or **Zsh** (Linux/macOS):
```bash
source .venv/bin/activate
```

### For **CMD** (Windows Command Prompt):
```cmd
.venv\Scripts\activate.bat
```

### For **PowerShell** (Windows):
```powershell
.venv\Scripts\Activate.ps1
```

> If you're using WSL or Git Bash on Windows, use the Bash/Zsh version.


3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure the IP address:
   - **Step 1:** Edit `allow_origins` in `MeineAPIundML/main.py` with your local IP.
   - **Step 2:** Update the corresponding IP in your Android app at:  
     `app/src/main/java/.../di/AppModule.kt`

---

## Android Frontend  
The mobile app is built using **Kotlin + Jetpack Compose**, following **Clean Architecture**.

### Architecture Layers:
- `data`: API and local model access  
- `domain`: Use cases and business logic  
- `presentation`: Jetpack Compose UI, ViewModels, and Navigation
- `di` : Dependency Injection 

# The app includes:
- A prediction screen
- History storage & dashboard
- Optional save for predictions
- Bottom navigation with AppBar
- About 
---

## Model Type  
This project implements **Lightweight Deep Learning**:
- Two dense layers (e.g., 1 and 8 neurons)
- Optimised for mobile speed and size
- TensorFlow Lite (TFLite) (2 KB)
---

## Technologies Used
- Kotlin + Jetpack Compose  
- Python 3.12 + FastAPI  
- TensorFlow & TFLite  
- MVVM + Clean Architecture  
- Local database (Room)