# 🔍 Image Tampering Detection

An advanced **Digital Image Forensics** tool built with Python and Streamlit. Upload any image and get a full AI-powered forensic analysis.

## Features

- **Error Level Analysis (ELA)** — Detects inconsistencies in JPEG compression artifacts
- **ML Classification** — Random Forest model predicts if an image is *Likely Real* or *Likely Tampered*
- **Suspicious Region Localization** — Heatmap and bounding boxes highlighting anomalous areas
- **EXIF Metadata Analysis** — Extracts camera, date, and software information
- **Forensic Report** — Downloadable full evidence report

## Tech Stack

- Python 3.10+
- Streamlit
- OpenCV
- Pillow
- scikit-learn
- NumPy

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Project Structure

```
├── app.py            # Streamlit UI
├── forensics.py      # Core analysis engine
├── train_model.py    # Model training script
├── models/           # Trained ML model artifacts
│   ├── model.joblib
│   ├── scaler.joblib
│   └── features.json
├── .streamlit/
│   └── config.toml   # Dark UI theme config
└── requirements.txt
```

## Disclaimer

This tool provides forensic **indicators** only and should not be treated as definitive proof of image manipulation.
