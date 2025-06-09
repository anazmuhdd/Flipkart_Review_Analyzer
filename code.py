import pandas as pd
import numpy as np
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
data.head(0)
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





