# my_pipeline.py 

import pandas as pd
import re
import string
import numpy as np
import os
import kagglehub
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_score, recall_score, f1_score
import nltk
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping

# Inicializar componentes necesarios
nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
analyzer = SentimentIntensityAnalyzer()
vectorizer = TfidfVectorizer(max_features=1000)

# Descargar y cargar datos de Kaggle con una muestra limitada
def load_data():
    path = kagglehub.dataset_download("kritanjalijain/amazon-reviews")
    print("Path to dataset files:", path)

    files = os.listdir(path)
    print("Archivos descargados:", files)
    train_file = os.path.join(path, 'train.csv')
    test_file = os.path.join(path, 'test.csv')

    columns = ['polaridad', 'titulo', 'texto']
    df_train = pd.read_csv(train_file, names=columns).head(100000)  # Muestra de 100,000 filas
    df_test = pd.read_csv(test_file, names=columns).head(10000)     # Muestra de 10,000 filas

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
    text = re.sub(r'http\S+|www\S+', '', text)  # Eliminar URLs
    text = re.sub(r'@\w+', '', text)  # Eliminar menciones
    text = text.translate(str.maketrans('', '', string.punctuation))  # Eliminar signos de puntuación
    return text.strip()  # Eliminar espacios en los extremos

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

# Función para vectorizar el texto
def vectorize_text(df_train, df_test):
    X_train = vectorizer.fit_transform(df_train['texto_limpio'])
    X_test = vectorizer.transform(df_test['texto_limpio'])
    return X_train, X_test

# Función para configurar, entrenar y evaluar el modelo
def train_model(df_train, df_test):
    max_words = 10000
    max_len = 200

    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(df_train['texto_limpio'])

    X_train = pad_sequences(tokenizer.texts_to_sequences(df_train['texto_limpio']), maxlen=max_len)
    X_test = pad_sequences(tokenizer.texts_to_sequences(df_test['texto_limpio']), maxlen=max_len)

    y_train = df_train['polaridad'].values
    y_test = df_test['polaridad'].values

    embedding_dim = 16
    model = Sequential([
        Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=max_len),
        GlobalAveragePooling1D(),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, callbacks=[early_stop])

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Loss: {loss:.4f}")

    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Guardar el modelo entrenado
    model.save("trained_model.h5")
    print("Modelo guardado como 'trained_model.h5'")

# Ejecutar pipeline de entrenamiento
if __name__ == "__main__":
    df_train, df_test = load_data()
    df_train = preprocess_data(df_train)
    df_test = preprocess_data(df_test)
    df_train = sentiment_analysis(df_train)
    df_test = sentiment_analysis(df_test)
    train_model(df_train, df_test)

