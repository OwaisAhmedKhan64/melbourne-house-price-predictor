# Melbourne House Price Predictor
A machine learning regressor model which predicts the price of a house in Melbourne based upon features like location, number of rooms, etc.

### Model Performance
* **Accuracy:** 82.37%
* **Mean Absolute Percentage Error:** 17.63%
* **Mean Absolute Error:** $189,550

### Model Training
The model was trained in **scikit-learn** using the **Random Forest Algorithm**. The following features were used to train the model:

* `Rooms` (The number of rooms)
* `Bathroom` (The number of bathrooms)
* `Distance` (Distance to Central Business District)
* `Landsize`
* `Type` (Type of house)
* `Regionname` (The 8 Melbourne regions)

### Local Setup Instructions

#### Prerequisites

* **Python** (Developed with 3.12.7): [Download here](https://www.python.org/downloads/release/python-3127/)
* **Node.js** (Developed with 24.11.0) & **NPM** (Developed with 11.6.1): [Download here](https://nodejs.org/en/download)
* **Git**: [Download here](https://git-scm.com/downloads)

#### 1. Repository Initialization
Clone the repository and navigate to the root directory:
```bash
git clone https://github.com/OwaisAhmedKhan64/melbourne-house-price-predictor.git
cd melbourne-house-price-predictor
```

#### 2. Backend Configuration (Django)
Navigate to the backend folder and initialize the Python environment:
```bash
cd backend
python -m venv .venv
```

**Activate the virtual environment:**
* Windows: `.venv\Scripts\activate`
* Mac/Linux: `source .venv/bin/activate`

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Run the server:**
```bash
python manage.py runserver
```

#### 3. Frontend Configuration (Vue.js)
Open a new terminal window and navigate to the frontend folder:
```bash
cd frontend
npm install
```

**Run the development server:**
```bash
npm run dev
```

#### 4. Usage
Once both servers are running:
1. Open your browser to the local URL provided by Vite (typically `http://localhost:5173`).
2. Input the property details in the form.
3. Click "Predict Price" to receive the valuation from the Django API.


### Inspiration
This project was inspired from the course on Kaggle - Intro to Machine Learning. Link: https://www.kaggle.com/learn/intro-to-machine-learning.
The dataset was also a part of this course.