from flask import Flask, request, jsonify, render_template_string
import re
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import json
from PIL import Image
from io import BytesIO
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from ultralytics import YOLO

app = Flask(__name__)

# Load models once at startup
popularity_model = joblib.load('popularity_model_v2.joblib')
yolo_model       = YOLO('yolov8n.pt')
topic_model      = tf.keras.models.load_model('topic_lstm_model.h5')

with open('topic_tokenizer.json', 'r') as f:
    tokenizer = tokenizer_from_json(f.read())

le_df = pd.read_csv('topic_label_encoder.csv')
label_decoder = dict(zip(le_df['index'], le_df['class']))

MAX_SEQUENCE_LENGTH = 100
FOOD_OBJECTS = {'banana', 'apple', 'pizza', 'cake', 'sandwich', 'hot dog',
                 'broccoli', 'carrot', 'orange', 'donut'}

TOPIC_EMOJI = {
    'fashion': '👗', 'food': '🍕', 'art': '🎨',
    'home': '🏠', 'travel': '✈️', 'beauty': '💄', 'other': '📌'
}

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Pinterest Post Analyzer</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">
<style>
  :root {
    --cream: #F5F0E8;
    --dark: #1A1612;
    --red: #E8341A;
    --red-light: #FDF0ED;
    --warm-gray: #8C8480;
    --border: #D4CBC0;
    --card: #FDFAF6;
    --success: #2D6A4F;
    --success-bg: #ECF5F0;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--cream);
    color: var(--dark);
    font-family: 'DM Sans', sans-serif;
    min-height: 100vh;
  }

  /* ── Header ── */
  header {
    background: var(--dark);
    padding: 20px 40px;
    display: flex;
    align-items: center;
    gap: 16px;
    border-bottom: 2px solid var(--red);
  }
  .logo {
    width: 32px; height: 32px;
    background: var(--red);
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px;
  }
  header h1 {
    font-family: 'Instrument Serif', serif;
    font-size: 22px;
    color: var(--cream);
    letter-spacing: -0.3px;
  }
  header span {
    margin-left: auto;
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    color: var(--warm-gray);
    letter-spacing: 0.08em;
    text-transform: uppercase;
  }

  /* ── Layout ── */
  main {
    max-width: 900px;
    margin: 0 auto;
    padding: 48px 24px;
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 24px;
    align-items: start;
  }
  @media (max-width: 680px) { main { grid-template-columns: 1fr; } }

  /* ── Cards ── */
  .card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 28px;
  }
  .card-title {
    font-family: 'Instrument Serif', serif;
    font-size: 20px;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 10px;
  }
  .card-title .num {
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    color: var(--warm-gray);
    background: var(--cream);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 3px 8px;
  }

  /* ── Form ── */
  label {
    display: block;
    font-size: 12px;
    font-weight: 500;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--warm-gray);
    margin-bottom: 8px;
  }
  textarea {
    width: 100%;
    background: var(--cream);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 12px 14px;
    font-family: 'DM Sans', sans-serif;
    font-size: 14px;
    color: var(--dark);
    resize: vertical;
    min-height: 100px;
    transition: border-color 0.2s;
    outline: none;
  }
  textarea:focus { border-color: var(--red); }

  /* ── Image drop zone ── */
  .drop-zone {
    border: 2px dashed var(--border);
    border-radius: 10px;
    padding: 28px 16px;
    text-align: center;
    cursor: pointer;
    transition: all 0.2s;
    margin-top: 16px;
    position: relative;
    background: var(--cream);
  }
  .drop-zone:hover, .drop-zone.dragover {
    border-color: var(--red);
    background: var(--red-light);
  }
  .drop-zone input[type=file] {
    position: absolute; inset: 0;
    opacity: 0; cursor: pointer; width: 100%; height: 100%;
  }
  .drop-icon { font-size: 28px; margin-bottom: 8px; }
  .drop-text { font-size: 13px; color: var(--warm-gray); line-height: 1.5; }
  .drop-text strong { color: var(--dark); }
  #preview-wrap {
    display: none;
    margin-top: 14px;
    border-radius: 8px;
    overflow: hidden;
    border: 1px solid var(--border);
  }
  #preview-wrap img {
    width: 100%; height: 160px;
    object-fit: cover; display: block;
  }

  /* ── Button ── */
  .btn {
    width: 100%;
    margin-top: 20px;
    padding: 14px;
    background: var(--red);
    color: white;
    border: none;
    border-radius: 10px;
    font-family: 'DM Sans', sans-serif;
    font-size: 15px;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.2s;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
  }
  .btn:hover { background: #C42A14; transform: translateY(-1px); }
  .btn:active { transform: translateY(0); }
  .btn:disabled { background: var(--border); cursor: not-allowed; transform: none; }

  /* ── Spinner ── */
  .spinner {
    width: 16px; height: 16px;
    border: 2px solid rgba(255,255,255,0.4);
    border-top-color: white;
    border-radius: 50%;
    animation: spin 0.7s linear infinite;
    display: none;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* ── Results ── */
  #results { display: none; }
  #results.visible { display: block; }

  .result-hero {
    background: var(--dark);
    border-radius: 14px;
    padding: 24px;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 20px;
  }
  .result-icon {
    font-size: 40px;
    flex-shrink: 0;
  }
  .result-label {
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--warm-gray);
    margin-bottom: 4px;
  }
  .result-topic {
    font-family: 'Instrument Serif', serif;
    font-size: 28px;
    color: var(--cream);
    text-transform: capitalize;
  }

  .prob-bar-wrap { margin-top: 12px; }
  .prob-label {
    display: flex;
    justify-content: space-between;
    font-size: 12px;
    color: var(--warm-gray);
    margin-bottom: 6px;
  }
  .prob-label strong { color: var(--cream); font-size: 14px; }
  .prob-bar {
    height: 6px;
    background: rgba(255,255,255,0.12);
    border-radius: 3px;
    overflow: hidden;
  }
  .prob-fill {
    height: 100%;
    background: var(--red);
    border-radius: 3px;
    transition: width 0.8s cubic-bezier(0.4,0,0.2,1);
    width: 0%;
  }

  .badge-row {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin-bottom: 16px;
  }
  .badge {
    font-size: 12px;
    font-family: 'DM Mono', monospace;
    padding: 5px 12px;
    border-radius: 20px;
    border: 1px solid var(--border);
    color: var(--dark);
    background: var(--cream);
  }
  .badge.popular {
    background: var(--success-bg);
    border-color: #A8D5BD;
    color: var(--success);
  }
  .badge.not-popular {
    background: var(--red-light);
    border-color: #F5C4BC;
    color: var(--red);
  }

  .objects-list {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
  }
  .obj-tag {
    font-size: 12px;
    font-family: 'DM Mono', monospace;
    padding: 4px 10px;
    background: var(--cream);
    border: 1px solid var(--border);
    border-radius: 6px;
    color: var(--dark);
  }
  .section-label {
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    color: var(--warm-gray);
    margin-bottom: 10px;
  }

  .error-box {
    background: var(--red-light);
    border: 1px solid #F5C4BC;
    border-radius: 10px;
    padding: 14px 16px;
    font-size: 13px;
    color: var(--red);
    display: none;
  }
  .error-box.visible { display: block; }

  .divider { height: 1px; background: var(--border); margin: 16px 0; }
</style>
</head>
<body>

<header>
  <div class="logo">📌</div>
  <h1>Pinterest Post Analyzer</h1>
  <span>ML · LSTM · YOLOv8</span>
</header>

<main>
  <!-- INPUT CARD -->
  <div class="card">
    <div class="card-title">
      <span class="num">01</span>
      Analyze a Post
    </div>

    <label>Post caption</label>
    <textarea id="text-input" placeholder="Beautiful summer dress perfect for vacation #fashion #style #ootd"></textarea>

    <label style="margin-top:16px;">Image</label>
    <div class="drop-zone" id="drop-zone">
      <input type="file" id="image-input" accept="image/*">
      <div class="drop-icon">🖼</div>
      <div class="drop-text"><strong>Click to upload</strong> or drag & drop<br>JPG, PNG, WEBP</div>
    </div>
    <div id="preview-wrap"><img id="preview" src="" alt="preview"></div>

    <div class="error-box" id="error-box"></div>

    <button class="btn" id="analyze-btn" onclick="analyze()">
      <div class="spinner" id="spinner"></div>
      <span id="btn-text">Analyze Post</span>
    </button>
  </div>

  <!-- RESULTS CARD -->
  <div class="card" id="results">
    <div class="card-title">
      <span class="num">02</span>
      Results
    </div>

    <div class="result-hero">
      <div class="result-icon" id="res-icon">📌</div>
      <div>
        <div class="result-label">Detected topic</div>
        <div class="result-topic" id="res-topic">—</div>
        <div class="prob-bar-wrap">
          <div class="prob-label">
            <span>Popularity probability</span>
            <strong id="res-prob">—</strong>
          </div>
          <div class="prob-bar"><div class="prob-fill" id="prob-fill"></div></div>
        </div>
      </div>
    </div>

    <div class="badge-row">
      <div class="badge" id="pop-badge">—</div>
    </div>

    <div class="divider"></div>

    <div class="section-label">Detected objects in image</div>
    <div class="objects-list" id="objects-list"></div>
  </div>
</main>

<script>
const TOPIC_EMOJI = {
  fashion:'👗', food:'🍕', art:'🎨',
  home:'🏠', travel:'✈️', beauty:'💄', other:'📌'
};

// Drag & drop
const dz = document.getElementById('drop-zone');
const fi = document.getElementById('image-input');
dz.addEventListener('dragover', e => { e.preventDefault(); dz.classList.add('dragover'); });
dz.addEventListener('dragleave', () => dz.classList.remove('dragover'));
dz.addEventListener('drop', e => {
  e.preventDefault(); dz.classList.remove('dragover');
  const file = e.dataTransfer.files[0];
  if (file) { fi.files = e.dataTransfer.files; showPreview(file); }
});
fi.addEventListener('change', () => { if (fi.files[0]) showPreview(fi.files[0]); });

function showPreview(file) {
  const reader = new FileReader();
  reader.onload = e => {
    document.getElementById('preview').src = e.target.result;
    document.getElementById('preview-wrap').style.display = 'block';
  };
  reader.readAsDataURL(file);
}

function setLoading(on) {
  const btn = document.getElementById('analyze-btn');
  const sp  = document.getElementById('spinner');
  const tx  = document.getElementById('btn-text');
  btn.disabled = on;
  sp.style.display  = on ? 'block' : 'none';
  tx.textContent = on ? 'Analyzing…' : 'Analyze Post';
}

function showError(msg) {
  const eb = document.getElementById('error-box');
  eb.textContent = msg;
  eb.classList.add('visible');
}
function clearError() {
  document.getElementById('error-box').classList.remove('visible');
}

async function analyze() {
  clearError();
  const text  = document.getElementById('text-input').value.trim();
  const image = document.getElementById('image-input').files[0];

  if (!text)  { showError('Please enter a post caption.'); return; }
  if (!image) { showError('Please upload an image.'); return; }

  setLoading(true);
  const form = new FormData();
  form.append('text',  text);
  form.append('image', image);

  try {
    const res  = await fetch('/analyze', { method: 'POST', body: form });
    const data = await res.json();

    if (!res.ok) { showError(data.error || 'Something went wrong.'); return; }

    // Populate results
    const topic = data.topic || 'other';
    document.getElementById('res-icon').textContent  = TOPIC_EMOJI[topic] || '📌';
    document.getElementById('res-topic').textContent = topic;

    const prob = data.popularity_probability || 0;
    document.getElementById('res-prob').textContent  = (prob * 100).toFixed(1) + '%';
    setTimeout(() => {
      document.getElementById('prob-fill').style.width = (prob * 100) + '%';
    }, 100);

    const popBadge = document.getElementById('pop-badge');
    popBadge.textContent = data.popularity_predicted
      ? '✓ Likely popular' : '✗ Unlikely to go viral';
    popBadge.className = 'badge ' + (data.popularity_predicted ? 'popular' : 'not-popular');

    const ol = document.getElementById('objects-list');
    ol.innerHTML = '';
    const objs = data.detected_objects || [];
    if (objs.length === 0) {
      ol.innerHTML = '<span class="obj-tag">No objects detected</span>';
    } else {
      [...new Set(objs)].forEach(o => {
        const tag = document.createElement('span');
        tag.className = 'obj-tag';
        tag.textContent = o;
        ol.appendChild(tag);
      });
    }

    document.getElementById('results').classList.add('visible');
  } catch (e) {
    showError('Network error — is the server running?');
  } finally {
    setLoading(false);
  }
}
</script>
</body>
</html>"""


def extract_text_features(text):
    return len(text), text.count('#'), int(any(ord(c) > 10000 for c in text))


def detect_image_objects(image):
    results      = yolo_model(image, verbose=False)[0]
    names        = results.names
    detected     = [names[int(cls)] for cls in results.boxes.cls]
    object_count = len(detected)
    has_person   = int('person' in detected)
    has_food     = int(any(obj in detected for obj in FOOD_OBJECTS))
    return detected, object_count, has_person, has_food


def predict_topic(text):
    seq  = tokenizer.texts_to_sequences([text])
    pad  = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
    pred = topic_model.predict(pad, verbose=0)
    return label_decoder[np.argmax(pred)]


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/analyze', methods=['POST'])
def analyze():
    text       = request.form.get('text', '').strip()
    image_file = request.files.get('image', None)

    if not text:
        return jsonify({'error': 'text field is required and cannot be empty.'}), 400
    if image_file is None:
        return jsonify({'error': 'image file is required.'}), 400

    try:
        topic                                                = predict_topic(text)
        text_len, n_hashtags, has_emoji                      = extract_text_features(text)
        image                                                = Image.open(BytesIO(image_file.read())).convert('RGB')
        detected_objects, object_count, has_person, has_food = detect_image_objects(image)

        features = pd.DataFrame([{
            'text_len':     text_len,
            'n_hashtags':   n_hashtags,
            'has_emoji':    has_emoji,
            'topic':        topic,
            'has_person':   has_person,
            'has_food':     has_food,
            'object_count': object_count,
        }])

        popularity = int(popularity_model.predict(features)[0])
        prob       = float(popularity_model.predict_proba(features)[0][1])

        return jsonify({
            'topic':                  topic,
            'popularity_predicted':   popularity,
            'popularity_probability': round(prob, 3),
            'detected_objects':       detected_objects,
            'image_object_count':     object_count,
        })

    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
