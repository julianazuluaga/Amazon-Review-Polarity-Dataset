# test_pipeline.py

import pytest
from my_pipeline import load_data, clean_text, tokenize_and_stem, preprocess_data, sentiment_analysis
import pandas as pd

# Prueba de carga de datos
def test_load_data():
    df_train, df_test = load_data()
    assert not df_train.empty, "El DataFrame de entrenamiento está vacío."
    assert not df_test.empty, "El DataFrame de prueba está vacío."

# Prueba de limpieza de texto
def test_clean_text():
    raw_text = "This is a test! Check out https://example.com @user"
    expected = "this is a test check out "
    assert clean_text(raw_text) == expected, "La función clean_text no limpia correctamente el texto."

# Prueba de tokenización y stemming
def test_tokenize_and_stem():
    text = "this is a simple text example"
    expected_tokens = ["simpl", "text", "exampl"]
    assert tokenize_and_stem(text) == expected_tokens, "La función tokenize_and_stem no funciona correctamente."

# Prueba de preprocesamiento
def test_preprocess_data():
    df_train, _ = load_data()
    df_processed = preprocess_data(df_train)
    assert 'texto_limpio' in df_processed.columns, "La columna 'texto_limpio' falta en el DataFrame procesado."
    assert 'tokens' in df_processed.columns, "La columna 'tokens' falta en el DataFrame procesado."

# Prueba de análisis de sentimiento
def test_sentiment_analysis():
    data = {'texto_limpio': ["I love this!", "I hate this!", "It's ok."]}
    df = pd.DataFrame(data)
    df = sentiment_analysis(df)
    assert set(df['sentiment']) == {"Positivo", "Negativo", "Neutral"}, "La función sentiment_analysis no clasifica correctamente los sentimientos."

if __name__ == "__main__":
    pytest.main()
