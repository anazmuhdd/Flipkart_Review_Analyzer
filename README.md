# Flipkart Review Sentiment Analyzer

A complete machine learning and deep learning pipeline for Flipkart product review sentiment analysis. This project performs data cleaning, EDA, sentiment labeling using VADER and rating, fine-tunes a BERT model, and integrates a React + Flask web app for user interaction.

---

## ✨ Features

- Clean and preprocess raw Flipkart reviews
- Generate sentiment labels from star ratings and text summaries
- Train a BERT-based classifier using Hugging Face Transformers
- Evaluate model with validation and test accuracy
- Fully working frontend in React with Bootstrap
- Flask backend API for sentiment prediction
- GPU support for faster model training

---

## 🥛 Tech Stack

### Machine Learning & NLP

- Python 3.10+
- Pandas, Numpy, Regex
- NLTK (VADER Sentiment Intensity)
- Scikit-learn (for splitting and evaluation)
- HuggingFace Transformers (BERT)
- PyTorch

### Web App

- React (with Vite)
- Bootstrap 5
- Axios (for HTTP requests)
- Flask (Python Backend API)
- CORS middleware for cross-origin support

---

## 📊 Project Pipeline

### 1. Data Preprocessing

- Load Flipkart product dataset (CSV format)
- Clean invalid price values (e.g., text or symbols)
- Handle nulls in reviews, summaries, and ratings
- Remove punctuation and lowercase the text
- Generate `Sentiment_from_rate` (based on rating)
- Generate `Sentiment_from_summary` (using VADER scores)

### 2. Exploratory Data Analysis

- Class-wise sentiment count plots
- Word clouds for positive, neutral, and negative summaries
- Crosstab analysis: summary vs. rating sentiment agreement

### 3. Model Building (BERT)

- Preprocess text using BERT tokenizer
- Split into train/validation/test (70/15/15)
- Create custom PyTorch Dataset
- Train `BertForSequenceClassification` (num_labels=3)
- Evaluate accuracy on train, validation, and test sets

### 4. Model Deployment

- Save model files: `model.safetensors`, tokenizer, config
- Setup Flask backend to:

  - Load model
  - Accept review text via `/predict` endpoint
  - Return predicted sentiment (positive/neutral/negative)

### 5. Frontend with React

- Textarea input for user reviews
- Submit review to Flask API using Axios
- Display sentiment result in styled alert
- Responsive and styled using Bootstrap 5

---

## ⚡ How to Run Locally

### 1. Train the Model

```bash
cd backend
python train_model.py  # Replace with your training script
```

### 2. Run the Flask Backend

```bash
cd backend
pip install -r requirements.txt
python app.py
```

### 3. Run React Frontend

```bash
cd frontend
npm install
npm run dev
```

Visit: `http://localhost:5173`

---

## 📊 Sample Results

- Classification accuracy: \~85-90% on test set
- Real-time sentiment analysis works smoothly with fast BERT inference

---

## 📄 Folder Structure

```
Flipkart_Review_Analyzer/
├── backend/
│   ├── app.py (Flask app)
│   ├── model/ (saved model folder)
│   └── train_model.py
├── frontend/
│   ├── src/
│   │   ├── Components/
│   │   └── App.jsx
│   └── public/
└── flipkart_product.csv
```

---

## 🙌 Author

**Mohammed Anas A R**
B.Tech CSE @ Mar Baselios College of Engineering, Trivandrum
Portfolio: \[Coming Soon]

---

## ✨ Credits

- HuggingFace Transformers
- Bootstrap 5
- React + Vite
- NLTK VADER

---

## 🛡️ License

This project is for academic and learning purposes only.
