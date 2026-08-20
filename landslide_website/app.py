import os
import sqlite3
import requests
import joblib
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash, session, g
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# ---------- Configuration ----------
app = Flask(__landslide__)
app.secret_key = 'SECRET_KEY'  # CHANGE THIS in production

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths to ML artifacts (adjust if needed)
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'logistic_model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'models', 'scaler.pkl')
ENCODER_PATH = os.path.join(BASE_DIR, 'models', 'label_encoder.pkl')

# OpenWeatherMap API
WEATHER_API_KEY = 'WEATHER_API_KEY'   # Replace with your key
WEATHER_CITY = 'Silang,PH'

# ---------- Database functions ----------
def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        # Use app.instance_path (defaults to 'instance' folder in app root)
        db_path = os.path.join(app.instance_path, 'users.db')
        # Ensure the instance folder exists
        os.makedirs(app.instance_path, exist_ok=True)
        db = g._database = sqlite3.connect(db_path)
        db.row_factory = sqlite3.Row
    return db

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

def init_db():
    with app.app_context():
        db = get_db()
        db.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        db.commit()

# ---------- Load ML models ----------
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
encoder = joblib.load(ENCODER_PATH)

# ---------- Weather helper ----------
def get_weather():
    """Fetch current weather from OpenWeatherMap, return safe numeric values."""
    try:
        url = f'https://api.openweathermap.org/data/2.5/weather?q={WEATHER_CITY}&appid={WEATHER_API_KEY}&units=metric'
        resp = requests.get(url, timeout=5)
        data = resp.json()
        if resp.status_code == 200:
            rain = data.get('rain', {}).get('1h', 0.0)
            return {
                'temp': float(data['main']['temp']),
                'humidity': int(data['main']['humidity']),
                'rainfall': float(rain),
                'description': data['weather'][0]['description'],
                'city': data['name']
            }
    except Exception as e:
        print('Weather API error:', e)
    # Fallback with numeric defaults
    return {'temp': 0.0, 'humidity': 0, 'rainfall': 0.0, 'description': 'N/A', 'city': 'Silang'}

# ---------- Historical data (placeholder) ----------
def get_historical_data():
    return [
        {'date': '2025-01-15', 'location': 'Purok 3, Barangay Santol', 'slope': 78, 'intensity': 65, 'infiltration': 35, 'risk': 'High'},
        {'date': '2025-01-10', 'location': 'Purok 3, Barangay Santol', 'slope': 82, 'intensity': 90, 'infiltration': 20, 'risk': 'High'},
        {'date': '2025-01-05', 'location': 'Purok 3, Barangay Santol', 'slope': 65, 'intensity': 25, 'infiltration': 95, 'risk': 'Low'},
        {'date': '2024-12-28', 'location': 'Purok 3, Barangay Santol', 'slope': 70, 'intensity': 45, 'infiltration': 60, 'risk': 'Medium'},
        {'date': '2024-12-20', 'location': 'Purok 3, Barangay Santol', 'slope': 85, 'intensity': 80, 'infiltration': 30, 'risk': 'High'},
        {'date': '2024-12-15', 'location': 'Purok 3, Barangay Santol', 'slope': 62, 'intensity': 15, 'infiltration': 100, 'risk': 'Low'},
        {'date': '2024-12-10', 'location': 'Purok 3, Barangay Santol', 'slope': 75, 'intensity': 50, 'infiltration': 45, 'risk': 'Medium'},
        {'date': '2024-12-05', 'location': 'Purok 3, Barangay Santol', 'slope': 88, 'intensity': 95, 'infiltration': 15, 'risk': 'High'},
    ]

# ---------- Routes ----------
@app.route('/')
def index():
    if not session.get('user_id'):
        return redirect(url_for('login'))
    weather = get_weather()
    return render_template('index.html', weather=weather)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        db = get_db()
        user = db.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
        if user and check_password_hash(user['password'], password):
            session['user_id'] = user['id']
            session['username'] = user['username']
            flash('Logged in successfully!', 'success')
            return redirect(url_for('index'))
        flash('Invalid username or password', 'danger')
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        hashed = generate_password_hash(password)
        db = get_db()
        try:
            db.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, hashed))
            db.commit()
            flash('Account created! Please log in.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username already exists', 'danger')
    return render_template('signup.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out', 'info')
    return redirect(url_for('login'))

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if not session.get('user_id'):
        return redirect(url_for('login'))
    
    weather = get_weather()
    rainfall_intensity = weather['rainfall']
    
    prediction_result = None
    proba_dict = None
    
    if request.method == 'POST':
        try:
            slope = float(request.form['slope'])
            infiltration = float(request.form['infiltration'])
            intensity = rainfall_intensity if not request.form.get('manual_intensity') else float(request.form['manual_intensity'])
            
            features = np.array([[slope, intensity, infiltration]])
            features_scaled = scaler.transform(features)
            pred_encoded = model.predict(features_scaled)[0]
            risk_label = encoder.inverse_transform([pred_encoded])[0]
            proba = model.predict_proba(features_scaled)[0]
            proba_dict = dict(zip(encoder.classes_, proba))
            
            prediction_result = {
                'slope': slope,
                'intensity': intensity,
                'infiltration': infiltration,
                'risk': risk_label,
                'probabilities': proba_dict
            }
        except Exception as e:
            flash(f'Error: {e}', 'danger')
    
    return render_template('predict.html', 
                           weather=weather, 
                           rainfall_intensity=rainfall_intensity,
                           prediction=prediction_result,
                           now=datetime.now())

@app.route('/history')
def history():
    if not session.get('user_id'):
        return redirect(url_for('login'))
    data = get_historical_data()
    return render_template('history.html', history=data)

@app.route('/weather')
def weather_page():
    if not session.get('user_id'):
        return redirect(url_for('login'))
    weather = get_weather()
    return render_template('weather.html', weather=weather)

@app.route('/camera')
def camera():
    if not session.get('user_id'):
        return redirect(url_for('login'))
    return render_template('camera.html')

# ---------- Run the app ----------
if __name__ == '__main__':
    init_db()
    app.run(host='0.0.0.0', port=5000, debug=True)