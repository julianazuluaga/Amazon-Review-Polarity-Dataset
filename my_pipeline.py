# my_pipeline.py

import pandas as pd
import re
import string
import os
import kagglehub
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk

# Inicializar componentes
nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
analyzer = SentimentIntensityAnalyzer()
vectorizer = TfidfVectorizer(max_features=1000)

# Función para cargar datos de Kaggle con muestra limitada
def load_data():
    path = kagglehub.dataset_download("kritanjalijain/amazon-reviews")
    train_file = os.path.join(path, 'train.csv')
    test_file = os.path.join(path, 'test.csv')

    columns = ['polaridad', 'titulo', 'texto']
    df_train = pd.read_csv(train_file, names=columns).head(100000)
    df_test = pd.read_csv(test_file, names=columns).head(10000)

    df_train = concat_columns(df_train, 'texto', 'titulo', 'texto')
    df_test = concat_columns(df_test, 'texto', 'titulo', 'texto')
    df_train['polaridad'] = df_train['polaridad'].map({1:0, 2:1})
    df_test['polaridad'] = df_test['polaridad'].map({1:0, 2:1})

    return df_train, df_test

# Funciones auxiliares
def concat_columns(df, col1, col2, new_col):
    df[new_col] = df[col1].apply(str) + ' ' + df[col2].apply(str)
    df.drop(col2, axis=1, inplace=True)
    return df

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

def tokenize_and_stem(text):
    tokens = [stemmer.stem(word) for word in word_tokenize(text) if word not in stop_words]
    return tokens

def preprocess_data(df):
    df['texto_limpio'] = df['texto'].apply(clean_text)
    df['tokens'] = df['texto_limpio'].apply(tokenize_and_stem)
    return df

def sentiment_analysis(df):
    df['compound'] = df['texto_limpio'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
    df['sentiment'] = df['compound'].apply(lambda score: 'Positivo' if score > 0.05 else 'Negativo' if score < -0.05 else 'Neutral')
    return df
