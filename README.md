# Emotion-Based Music Recommendation System

Project date: 18 April, 2022 | ECE 560: Intelligent Internet-of-Things

This project uses deep learning to detect facial emotion from a webcam feed and recommends music accordingly.

## Features
- Real-time face detection with OpenCV and Haar cascade
- Emotion classification using CNN trained on FER-2013
- Music recommendations via YouTube based on detected emotion

## Project Structure
- `src/`: Training dataset and model code
- `train.py`: Trains the CNN
- `evaluate.py`: Evaluates model on FER-2013
- `live_demo.py`: Webcam-based live emotion detection and music trigger
- `data/fer2013.csv`: Download file from the link below and place it in this folder

## Getting Started
```bash
pip install -r requirements.txt
python train.py
python evaluate.py
python live_demo.py
```
## Download Dataset

FER-2013 on Kaggle: https://www.kaggle.com/datasets/deadskull7/fer2013. Download and place .csv file in folder `/data` in the root directory.
