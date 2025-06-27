# Flipkart Review Sentiment Analyzer (BERT-based)

This project builds a robust sentiment classification system using product review data scraped from Flipkart. The main goal is to predict whether a review expresses a positive, negative, or neutral sentiment using a BERT deep learning model.

---

## 🔍 Overview

* Cleans and preprocesses Flipkart product review dataset
* Performs EDA (exploratory data analysis) and visualizations
* Uses BERT (via HuggingFace Transformers) to train a sentiment classifier
* Evaluates model on test data
* Supports custom prediction on user review text

---

## 📁 Dataset

The dataset contains:

* ProductName
* Price
* Rate (Rating)
* Summary (short review)
* Review (detailed review)

⬆️ Primary text used for training is the "Summary" column
⬆️ Labels are inferred from VADER sentiment scores and mapped to 3 classes:

* `0` → Negative
* `1` → Neutral
* `2` → Positive

---

## 🔧 Preprocessing Steps

* Cleaned price values (remove ₹, commas)
* Removed null or corrupt rows
* Cleaned special characters from text
* Converted all text to lowercase
* Generated sentiment labels from both ratings and VADER compound scores
* Saved final dataset to CSV

---

## 📊 EDA & Visualization

* Sentiment distribution plots using seaborn
* Top product counts per sentiment category
* Word clouds for positive, neutral, and negative reviews

---

## 🚀 Model Training (BERT)

* Used HuggingFace Transformers (BertTokenizer, BertForSequenceClassification)
* Tokenized summaries to max length 128
* Used PyTorch datasets and Trainer API
* Split into train, validation, and test sets (stratified)
* Trained for 5 epochs with Adam optimizer and softmax loss

---

## 🔬 Evaluation

* Evaluated on training, validation, and test sets
* Used sklearn's `classification_report` to display precision, recall, f1-score
* Tracked overfitting or underfitting through accuracy/loss comparison

---

## 🤔 Prediction

* Model and tokenizer are saved to disk
* You can load the model and run predictions on new summaries using:

  ```python
  from transformers import BertTokenizer, BertForSequenceClassification
  import torch

  tokenizer = BertTokenizer.from_pretrained("path/to/saved/model")
  model = BertForSequenceClassification.from_pretrained("path/to/saved/model")

  text = "The product quality is awesome"
  inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
  outputs = model(**inputs)
  predicted_class = torch.argmax(outputs.logits).item()
  ```

---

## 💡 Tech Stack

* Python
* Pandas, NumPy
* Matplotlib, Seaborn, WordCloud
* PyTorch, HuggingFace Transformers
* Scikit-learn

---

## 🚪 Future Work

* Extend model to classify full reviews (not just summaries)
* Add product category to improve prediction
* Convert model to ONNX/TFLite for lightweight inference

---

## 📅 Author

Mohammed Anas A R
B.Tech CSE, Mar Baselios College of Engineering & Technology, Trivandrum
