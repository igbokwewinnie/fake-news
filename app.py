import pickle
import nltk
import streamlit as st
nltk.data.path = ["nltk_data"]
from nltk.corpus import stopwords
nltk.download('stopwords', download_dir="nltk_data")

import streamlit as st
import numpy as np
import pandas as pd
import re

from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

cv = pickle.load(open('news-transformer.pkl','rb'))
model = pickle.load(open('pred-model.pkl','rb'))

#loading dataset
df_real=pd.read_csv('fake.csv')
df_fake=pd.read_csv('real.csv')

df_real['label']=0
df_fake['label']=1
df=pd.concat([df_real,df_fake],ignore_index=True)
df.reset_index(inplace=True)

df = df.dropna()
X = df.drop('label', axis=1)
y = df['label']

#data preprocessing and stemming
ps = PorterStemmer()
def stemming(tweet):
    stemmed_content = re.sub('[^a-zA-Z]',' ',tweet)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [ps.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

df['tweet'] = df['tweet'].apply(stemming)

#vectorizing the data
X = df['tweet']
y = df['label']

X = cv.transform(X)

#splitting our datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
model.fit(X_train,y_train)

#building our website
st.title('Football Fake News Detector')
input_text = st.text_input('Enter Article')

def prediction(input_text):
    input_data = cv.transform([input_text])
    prediction = model.predict(input_data)
    return prediction[0]

if input_text:
    pred = prediction(input_text)
    if pred == 1:
        st.write('This is a Fake News')

    else:
        st.write('This is a Real News')