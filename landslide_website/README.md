# 🌧️ Landslide Monitoring System – Purok 3, Santol

A web application for predicting rainfall‑induced landslides using a Logistic Regression model.  
Built with Flask, Tailwind CSS, and OpenWeatherMap API.

## Features
- User authentication (signup/login)
- Real‑time risk prediction based on:
  - Slope Angle (user input)
  - Rainfall Infiltration (user input)
  - Rainfall Intensity (fetched from weather API)
- Historical incident dashboard
- Live weather monitoring
- RTSP camera placeholder

## Tech Stack
- Python 3.9+
- Flask
- scikit-learn / XGBoost / Random Forest (for comparison)
- Tailwind CSS
- SQLite (for users)

## Setup
1. Clone the repo  
   `git clone https://github.com/yourusername/landslide-monitoring.git`

2. Create a virtual environment  
   `python -m venv venv`  
   `source venv/bin/activate` (Linux/Mac) or `venv\Scripts\activate` (Windows)

3. Install dependencies  
   `pip install -r requirements.txt`

4. Place your trained model files (`logistic_model.pkl`, `scaler.pkl`, `label_encoder.pkl`) inside the `models/` folder (or adjust paths in `app.py`).

5. Create a `.env` file (see `.env.example`) and add your OpenWeatherMap API key.

6. Run the app  
   `python app.py`

7. Open `http://localhost:5000` in your browser.