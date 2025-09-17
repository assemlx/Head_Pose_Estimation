🎯 Head Pose Estimation using MediaPipe + SVR

This project predicts head pose angles (pitch, yaw, roll) from facial landmarks using MediaPipe FaceMesh and a trained Support Vector Regressor (SVR) model.

It supports both images and videos and exposes a FastAPI app for easy usage.

📂 Project Structure
head_pose_estimation/
│── app.py               # FastAPI app (single endpoint for image/video prediction)
│── utils.py             # Processing functions (image & video)
│──pose_prediction.ipynb   # Data preparation + model training
│── AFLW2000/                # Dataset
│── requirements.txt     # Python dependencies
│── .gitignore           # Ignore venv, cache, datasets, etc.
│── README.md            # Project documentation

⚙️ Installation

Clone the repository:

git clone https://github.com/yourusername/head_pose_estimation.git
cd head_pose_estimation


Create and activate a virtual environment:

python -m venv .venv
source .venv/bin/activate    # On Linux/Mac
.venv\Scripts\activate       # On Windows


Install dependencies:

pip install -r requirements.txt

🚀 Usage
1. Run the FastAPI server
uvicorn app:app --reload


Server will be available at:

http://127.0.0.1:8000/docs

2. Send an image or video for prediction

Go to the Swagger UI (/docs) and use the /predict/ endpoint:

Upload an image (.jpg, .png) → output is a processed image with pose axes.

Upload a video (.mp4, .avi) → output is a processed video saved as output_video.mp4.

🧠 Model Training

The notebook in notebooks/model_training.ipynb prepares data from AFLW2000 dataset, extracts facial landmarks, and trains an SVR model for predicting (pitch, yaw, roll).

After training, save the model:

import joblib
joblib.dump(model, "svr_model.pkl")


Place the model file in the project root so app.py and utils.py can use it.

📊 Example Outputs

Image
Input: face image
Output: face with red, green, blue axes showing head orientation

Video
Input: video file
Output: output_video.mp4 with overlaid axes

🛠️ Requirements

Python 3.8+

OpenCV

MediaPipe

NumPy

scikit-learn

FastAPI

Uvicorn

Matplotlib

(Already listed in requirements.txt)

📌 To-Do

 Improve real-time video streaming in FastAPI

 Experiment with deep learning models (CNN, Vision Transformers)

 Deploy on cloud (Streamlit / Gradio / Hugging Face Spaces)

✍️ Author: Asem Aly
