# CCTV Face Detection and Recognition System

## Overview

This project is an AI-based surveillance system developed to detect, recognize, and track missing persons using facial recognition techniques. The system combines MTCNN for face detection, FaceNet for feature extraction, and Euclidean Distance-based similarity matching for person identification. It supports real-time webcam recognition as well as multi-camera tracking using CCTV-style video feeds.

The project was developed as part of a Mini Project focusing on intelligent surveillance systems for missing person detection.

---

## Features

* Face Detection using MTCNN
* Face Recognition using FaceNet
* Similarity-Based Face Matching using Euclidean Distance
* Multi-Camera Person Tracking
* Low-Light Face Recognition Enhancement using CLAHE
* Real-Time Audio Alert Generation
* Timestamp and Camera-Based Tracking Logs
* Performance Evaluation using Accuracy, Precision, Recall, and F1-Score
* Support for CCTV-Style Surveillance Videos

---

## Technologies Used

* Python
* OpenCV
* PyTorch
* FaceNet
* MTCNN
* NumPy
* Scikit-Learn
* Matplotlib
* Seaborn

---

## Project Structure

```plaintext
faces_clustered/
│
├── person_1
├── person_2
├── person_3
└── ...

build_dataset.py
build_embeddings.py
cluster_faces.py
extract_faces.py
recognize.py
multi_cam_recognize.py
test_images.py
webcam_capture.py

lowlight1.mp4
lowlight2.mp4
lowlight3.mp4
```

---

## Installation

### Clone Repository

```bash
git clone https://github.com/Ani2708/CCTV-Face-Detection-and-Recognition-System.git
cd CCTV-Face-Detection-and-Recognition-System
```

### Create Virtual Environment

```bash
py -3.10 -m venv venv
```

Activate:

```bash
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install numpy==1.26.4
pip install torch==2.2.2 torchvision==0.17.2
pip install facenet-pytorch
pip install opencv-python
pip install scikit-learn
pip install matplotlib
pip install seaborn
pip install pillow
pip install scipy
```

---

## Generating Face Embeddings

Before running the recognition system, generate facial embeddings:

```bash
python build_embeddings.py
```

This creates:

```plaintext
embeddings.pkl
```

which contains the FaceNet embeddings used for recognition.

---

## Running Webcam Face Recognition

To detect and recognize registered individuals using a webcam:

```bash
python recognize.py
```

Features:

* Real-time face detection
* Face recognition
* Audio alert on detection
* Missing person identification

---

## Running Multi-Camera Tracking

The project includes three CCTV-style video feeds:

```plaintext
lowlight1.mp4
lowlight2.mp4
lowlight3.mp4
```

Run:

```bash
python multi_cam_recognize.py
```

Features:

* Three-camera support
* Person tracking across cameras
* Detection timestamps
* Camera movement logging
* Audio alerts

Example Output:

```plaintext
[ALERT] MISSING PERSON DETECTED: Anirudh at CAM 1
[MOVE] Anirudh moved from CAM 1 → CAM 2
[MOVE] Anirudh moved from CAM 2 → CAM 3
```

---

## Performance Evaluation

Run:

```bash
python test_images.py
```

This generates:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix
* Performance Metrics Graph

---

## Dataset

This project uses:

### ChokePoint Dataset

The ChokePoint dataset contains surveillance-style images and video sequences of individuals passing through multiple camera viewpoints and is widely used for face recognition research.

---

## Methodology

1. Face Detection using MTCNN
2. Face Preprocessing and Enhancement
3. Feature Extraction using FaceNet
4. Embedding Generation
5. Similarity Matching using Euclidean Distance
6. Missing Person Identification
7. Multi-Camera Tracking
8. Alert Generation and Logging

---

## Future Enhancements

* Integration with real CCTV camera networks
* Cloud-based centralized monitoring system
* Mobile notification support
* Database integration for missing person records
* Support for large-scale surveillance deployments

---

