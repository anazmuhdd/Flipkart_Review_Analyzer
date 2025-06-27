# Flipkart Review Analyzer using BERT
# This script preprocesses Flipkart product reviews, cleans the data, and trains a BERT model to classify sentiments based on the reviews.
import pandas as pd
import numpy as np

"""Data Preprocessing"""

data=pd.read_csv("flipkart_product.csv", encoding='latin1')
print(data.head(10))

data.info()

data.isnull().sum()

print(data['Price'])

data.nunique()

def check_price(price):
    try:
        float(price.replace('₹', '').replace(',', ''))
        return True
    except:
        return False
problematic_prices = data[~(data['Price'].apply(check_price))]
print(problematic_prices['Price'])

import re
import time
def clean_price(price):
    if(pd.isna(price)):
        return None
    else:
        price=str(price).replace('₹', '').replace(',', '')
        cleaned_price=re.search(r'\d+\.?\d*', price)
        cleaned_price = cleaned_price.group(0) if cleaned_price else None
        return cleaned_price if cleaned_price else None

data['Cleaned_Price'] = data['Price'].apply(clean_price)
print(data['Cleaned_Price'].head(10))

data['Cleaned_Price'].to_csv("cleaned_prices.csv", index=False)

data.head()

data['Cleaned_Price'].info()

data.isnull().sum()

data['Cleaned_Price'].isnull()

data

data['Cleaned_Price']=pd.to_numeric(data['Cleaned_Price'])

print("Rows with Nan after cleaning:", data['Cleaned_Price'].isnull().sum())
data['Cleaned_Price'].describe()

data=data.dropna(subset=['Cleaned_Price'])

data['Price']=data['Cleaned_Price']

data.head(10)

data.drop('Cleaned_Price', axis=1, inplace=True)

data.head(10)

data.isnull().sum()

data=data.dropna(subset=['Summary'])
data['Review']=data['Review'].fillna('No Review found in this')

data['Rate']=pd.to_numeric(data['Rate'],errors='coerce')
data.dropna(subset=['Rate'], inplace=True)
data['Rate'] = data['Rate'].astype(int)
data['Rate'].describe()

data.isnull().sum()

import re
def clean_text(text):
    return re.sub(r'[^a-zA-Z0-9\s]', '', text)
data['Review'] = data['Review'].apply(clean_text)
data['Review'].head(10)

data['ProductName'] = data['ProductName'].apply(clean_text)
data['ProductName'].head(10)
data['Summary'] = data['Summary'].apply(clean_text)
data['Summary'].head(10)

def lowercase_text(text):
    return text.lower() if isinstance(text, str) else text

data['ProductName'] = data['ProductName'].apply(lowercase_text)
data['Summary'] = data['Summary'].apply(lowercase_text)
data['Review'] = data['Review'].apply(lowercase_text)
data.head(10)

def rate_to_category(rate):
    if rate >= 4:
        return 'positive'
    elif rate == 3:
        return 'neutral'
    else:
        return 'negative'
data['Sentiment_from_rate'] = data['Rate'].apply(rate_to_category)
data['Sentiment_from_rate'].value_counts()

data.head()

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')

sia= SentimentIntensityAnalyzer()

data['Summary_Sentiment'] = data['Summary'].apply(lambda x: sia.polarity_scores(x)['compound'])

def classify_sentiment(score):
    if score >= 0.05:
        return 'positive'
    elif score <= -0.05:
        return 'negative'
    else:
        return 'neutral'
data['Sentiment_from_summary'] = data['Summary_Sentiment'].apply(classify_sentiment)
data['Sentiment_from_summary'].value_counts()

data.head(10)

compare=pd.crosstab(data['Sentiment_from_rate'], data['Sentiment_from_summary'])
print(compare)

from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(10, 6))
sns.countplot(x='Sentiment_from_summary', data=data)
plt.title('Sentiment Distribution from Summary')
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(x='Sentiment_from_rate', data=data)
plt.title('Sentiment Distribution from Rating')
plt.show()

plt.figure(figsize=(12, 8))
top_products=data['ProductName'].value_counts().index[:10]
top_products_data=data[data['ProductName'].isin(top_products)]
sns.countplot(y='ProductName', hue='Sentiment_from_summary', data=top_products_data, order=top_products_data['ProductName'].value_counts().index)
plt.title('Top 10 Products by Count')
plt.xlabel('Count')
plt.ylabel('Product Name')
plt.show()

positive_reviews=' '.join(data[data['Sentiment_from_summary']=='positive']['Summary'])
neutral_reviews=' '.join(data[data['Sentiment_from_summary']=='neutral']['Summary'])
negative_reviews=' '.join(data[data['Sentiment_from_summary']=='negative']['Summary'])

plt.figure(figsize=(10, 6))
wordcloud_positive = WordCloud(width=800, height=400, background_color='black').generate(positive_reviews)
plt.imshow(wordcloud_positive, interpolation='bilinear')
plt.axis('off')
plt.title('Positive Reviews Word Cloud')
plt.show()

plt.figure(figsize=(10, 6))
wordcloud_neutral = WordCloud(width=800, height=400, background_color='black').generate(neutral_reviews)
plt.imshow(wordcloud_neutral, interpolation='bilinear')
plt.axis('off')
plt.title('Neutral Reviews Word Cloud')
plt.show()

plt.figure(figsize=(10, 6))
wordcloud_negative = WordCloud(width=800, height=400, background_color='black').generate(negative_reviews)
plt.imshow(wordcloud_negative, interpolation='bilinear')
plt.axis('off')
plt.title('Negative Reviews Word Cloud')
plt.show()

data.head(10)

data['Sentiment_from_summary'].value_counts()

from sklearn.model_selection import train_test_split
df=data[['Summary', 'Sentiment_from_summary']].copy()
label_mapping = {'positive': 2, 'neutral': 1, 'negative': 0}
df['Sentiment_Label'] = df['Sentiment_from_summary'].map(label_mapping)

train_texts,temp_texts,train_labels,temp_labels=train_test_split(
    df['Summary'],df['Sentiment_Label'],test_size=0.3,stratify=df['Sentiment_Label'],random_state=42)

val_texts,test_texts,val_labels,test_labels=train_test_split(
    temp_texts,temp_labels,test_size=0.5,stratify=temp_labels,random_state=42)
print("Training set size:", len(train_texts))
print("Validation set size:", len(val_texts))
print("Test set size:", len(test_texts))

from transformers import BertTokenizer

tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')

train_encodings=tokenizer(list(train_texts),truncation=True,padding=True,max_length=128)
val_encodings=tokenizer(list(val_texts),truncation=True,padding=True,max_length=128)
test_encodings=tokenizer(list(test_texts),truncation=True,padding=True,max_length=128)

import torch

class SentimeDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = SentimeDataset(train_encodings, list(train_labels))
val_dataset = SentimeDataset(val_encodings, list(val_labels))
test_dataset = SentimeDataset(test_encodings, list(test_labels))

from transformers import BertForSequenceClassification, Trainer, TrainingArguments
device=torch.cuda.is_available()
if device:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("Using device:", device)
model= BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
model.to(device)

from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",  # <-- corrected here
    save_strategy="epoch",
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    logging_dir="./logs",
    logging_steps=10,
)


trainer=Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=None  # You can define a custom metric function if needed
)

trainer.train()

import torch

print(torch.cuda.is_available())     # Should print: True
print(torch.cuda.device_count())     # Number of GPUs available
print(torch.cuda.get_device_name(0)) # Name of your GPU

trainer.evaluate()

# Get predictions
predictions = trainer.predict(test_dataset)
preds = torch.argmax(torch.tensor(predictions.predictions), dim=1)

# Classification Report
from sklearn.metrics import classification_report
print(classification_report(test_labels, preds))

# Evaluate on training set
train_metrics = trainer.evaluate(train_dataset)
print("Training metrics:", train_metrics)

# Evaluate on validation set (you've already done this)
val_metrics = trainer.evaluate(val_dataset)
print("Validation metrics:", val_metrics)

# Evaluate on test set (you've already done this)
test_metrics = trainer.evaluate(test_dataset)
print("Test metrics:", test_metrics)

# Now, compare the metrics (e.g., loss and accuracy)
# If training loss is much lower than validation and test loss, and accuracy is much higher on training, it's overfitting.
# If training loss is high and accuracy is low, it might be underfitting.