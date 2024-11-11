import pytest
from my_pipeline import load_data, clean_text, tokenize_and_stem, preprocess_data, sentiment_analysis

def test_load_data():
    df_train, df_test = load_data()
    assert not df_train.empty, "El DataFrame de entrenamiento está vacío."
    assert not df_test.empty, "El DataFrame de prueba está vacío."

def test_clean_text():
    raw_text = "This is a test! Check out https://example.com @user"
    expected = "this is a test check out"
    result = clean_text(raw_text)
    assert result == expected, f"La función clean_text no limpia correctamente el texto. Resultado: {result}"

def test_tokenize_and_stem():
    text = "this is a simple text example"
    expected_tokens = ["simpl", "text", "exampl"]
    assert tokenize_and_stem(text) == expected_tokens, "La función tokenize_and_stem no funciona correctamente."

def test_preprocess_data():
    df_train, _ = load_data()
    df_processed = preprocess_data(df_train)
    assert 'texto_limpio' in df_processed.columns, "La columna 'texto_limpio' falta en el DataFrame procesado."
    assert 'tokens' in df_processed.columns, "La columna 'tokens' falta en el DataFrame procesado."

def test_sentiment_analysis():
    data = {'texto_limpio': ["I love this!", "I hate this!", "It's ok."]}
    df = pd.DataFrame(data)
    df = sentiment_analysis(df)
    assert set(df['sentiment']) == {"Positivo", "Negativo", "Neutral"}, "La función sentiment_analysis no clasifica correctamente los sentimientos."

if __name__ == "__main__":
    pytest.main()
