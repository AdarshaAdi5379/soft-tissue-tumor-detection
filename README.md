# 🧠 Soft Tissue Tumor Detection using GM-UNet & EfficientNet  
A Flask-based machine learning application for automated soft tissue tumor segmentation and classification from MRI images.

---

## 🚀 Project Overview
This project performs **soft tissue tumor detection** in two stages:

1. **Segmentation** – GM-UNet predicts the tumor region from an MRI scan.  
2. **Classification** – EfficientNetB0 classifies the tumor ROI as **Benign** or **Malignant**.

The system is deployed using a **Flask backend**, with user authentication (MySQL) and a clean web interface for image upload and result visualization.

---

## 🧩 Key Features
- ✔ MRI Upload Interface  
- ✔ Automatic preprocessing (resize 256×256, normalization)  
- ✔ GM-UNet-based tumor segmentation  
- ✔ EfficientNetB0-based classification  
- ✔ ROI extraction from segmentation mask  
- ✔ Probability score output  
- ✔ Downloadable mini-report / summary  
- ✔ Secure user login (MySQL)  
- ✔ Clean Flask API  
- ✔ Easily deployable on Railway / Render / AWS  

---

## 📂 Tech Stack
### **Backend**
- Flask  
- MySQL  
- python-dotenv  
- Gunicorn (for production)

### **Machine Learning**
- TensorFlow / Keras  
- GM-UNet  
- EfficientNetB0  
- OpenCV  
- Pillow  

### **Dataset**
- Kaggle Soft-Tissue Sarcoma MRI Dataset  
(Approximately 700+ MRI images with segmentation masks)

---

## 📁 Folder Structure
project-root/
│── app.py # Flask application entry point
│── models/
│ ├── gmunet_model.h5 # Segmentation model
│ ├── effnet_model.h5 # Classification model
│
│── static/
│ └── (CSS/JS/Images)
│
│── templates/
│ └── (HTML files)
│
│── utils/
│ ├── preprocess.py # Preprocessing functions
│ ├── segmentation.py # GM-UNet inference
│ ├── classification.py # EfficientNet inference
│
│── requirements.txt
│── README.md
│── .env.example


🖼 How It Works (Pipeline)

User uploads an MRI image

Flask server preprocesses image

GM-UNet generates tumor segmentation mask

Mask is used to extract tumor region

EfficientNet predicts Benign or Malignant

System displays:

segmentation mask

classification result

confidence score

downloadable report

📊 Model Performance
Model	Task	Metric	Score
GM-UNet	Segmentation	Dice Coefficient	~0.90
EfficientNetB0	Classification	Accuracy	~94%

(Replace with your actual metrics.)

📜 License

This project is for educational and research purposes.

👨‍💻 Author

Adarsha K K
Computer Science Engineering
Contact: adarshakk1234@gmail.com

GitHub: https://github.com/AdarshaAdi5379

⭐ If you found this useful

Consider giving the repo a star to support the project!
