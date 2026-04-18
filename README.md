# Social-Media-Analysis-System
A full end-to-end ML pipeline that predicts Pinterest post popularity and classifies topic, combining text features, computer vision (YOLOv8), and a deep learning LSTM model, served via a Flask web app with an interactive UI.

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-LSTM-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Object%20Detection-00BFFF?style=flat)
![Flask](https://img.shields.io/badge/Flask-Web%20App-000000?style=flat&logo=flask&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-RandomForest-F7931E?style=flat&logo=scikit-learn&logoColor=white)

## Overview

Given a Pinterest post (caption + image), the system:
1. **Cleans and encodes** the text — hashtags, length, emoji detection
2. **Detects objects** in the image using YOLOv8 (people, food, etc.)
3. **Classifies the topic** using an LSTM model: fashion / food / art / home / travel / beauty / other
4. **Predicts popularity** using a Random Forest trained on combined text + image features
5. **Returns results** in a clean interactive web interface

## Demo

Run locally and open `http://localhost`:

```bash
python app.py
```

Upload any image, type a caption — get topic, popularity prediction, probability score, and detected objects instantly.

## Pipeline

```
pinterest.csv + images/
        │
        ▼
  Data Cleaning          - drop nulls, deduplicate, filter missing images
        │
        ▼
  Text Processing        - lowercase, remove URLs, extract hashtags, emoji flag
        │
        ▼
  Image Preprocessing    - resize to 640×640, convert to RGB
        │
        ▼
  Labeling               - is_popular (median repins), topic (keyword rules)
        │
        ├──► Random Forest v1   - text features only (baseline)
        │
        ▼
  YOLOv8 Inference       - detect objects - has_person, has_food, object_count
        │
        ├──► Random Forest v2   - text + image features (final model)
        │
        ▼
  LSTM Training          - Embedding - LSTM - Dense - topic classification
        │
        ▼
  Flask Web App          - interactive UI at /  |  REST API at POST /analyze
```

---

## Models

| Model | Task | Input features |
|-------|------|----------------|
| Random Forest v1 | Popularity classification | text_len, n_hashtags, has_emoji, topic |
| YOLOv8n | Object detection | Raw images |
| Random Forest v2 | Popularity classification | v1 + has_person, has_food, object_count |
| LSTM | Topic classification | Tokenized text sequences |

## API

The web app also exposes a REST endpoint for programmatic use:

**`POST /analyze`** — multipart/form-data with `text` and `image` fields

```bash
curl -X POST http://localhost:5000/analyze \
  -F "text=Beautiful summer dress #fashion #style" \
  -F "image=@post.jpg"
```

```json
{
  "topic": "fashion",
  "popularity_predicted": 1,
  "popularity_probability": 0.823,
  "detected_objects": ["person", "handbag"],
  "image_object_count": 2
}
```

## Project Structure

```
├── app.py                       # Flask web app + REST API (with built-in UI)
├── final project.ipynb          # Full ML pipeline notebook
├── popularity_model_v2.joblib   # Trained Random Forest (text + image features)
├── topic_lstm_model.h5          # Trained LSTM topic classifier
├── topic_tokenizer.json         # Keras tokenizer vocabulary
├── topic_label_encoder.csv      # Topic label decoder
├── yolov8n.pt                   # YOLOv8 nano weights
└── requirements.txt
```


## Dataset

Source: **[Pinterest Analysis using NLP and Image Analysis](https://www.kaggle.com/datasets/andreacombette/pinterest-analysis-using-nlp-and-image-analysis)** - Kaggle (Andrea Combette)

Images and raw CSV are not included in this repository due to size and licensing.
To reproduce the full pipeline:
1. Download the dataset from Kaggle
2. Place `pinterest.csv` in `data/raw/`
3. Place images in `data/raw/images/images/`
4. Run all cells in `final project.ipynb`

## How to Run

```bash
# Clone
git clone https://github.com/assiyabaubekova/social-media-analysis.git
cd social-media-analysis

# Install dependencies
pip install -r requirements.txt

# Start the web app
python app.py
# Open http://localhost:5000
```

**`requirements.txt`:**
```
flask
pandas
numpy
scikit-learn
Pillow
tensorflow
ultralytics
joblib
```

## Stack

Python · scikit-learn · TensorFlow/Keras · YOLOv8 (Ultralytics) · Flask · pandas · Pillow

*Project built as part of a data science portfolio*
