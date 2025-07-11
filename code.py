import pandas as pd
# Data Preprocessing
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


data.to_csv("cleaned_flipkart_product.csv", index=False)
print("Data cleaning and preprocessing completed successfully.")
data.info()
data.describe()


data1 = pd.read_csv("cleaned_flipkart_product.csv")
data1['Summary']=data1['Summary'].astype(str)


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
wordcloud_positive = WordCloud(width=800, height=400, background_color='white').generate(positive_reviews)
plt.imshow(wordcloud_positive, interpolation='bilinear')
plt.axis('off')
plt.title('Positive Reviews Word Cloud')
plt.show()

plt.figure(figsize=(10, 6))
wordcloud_neutral = WordCloud(width=800, height=400, background_color='white').generate(neutral_reviews)
plt.imshow(wordcloud_neutral, interpolation='bilinear')
plt.axis('off')
plt.title('Neutral Reviews Word Cloud')
plt.show()

plt.figure(figsize=(10, 6))
wordcloud_negative = WordCloud(width=800, height=400, background_color='white').generate(negative_reviews)
plt.imshow(wordcloud_negative, interpolation='bilinear')
plt.axis('off')
plt.title('Negative Reviews Word Cloud')
plt.show()




