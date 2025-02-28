import string
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pymorphy2
from translate import Translator

# Установка необходимых ресурсов NLTK
nltk.download("punkt")
nltk.download("stopwords")

# Инициализация лемматизатора и списка стоп-слов
morph = pymorphy2.MorphAnalyzer()
stop_words = set(stopwords.words("russian"))

def preprocess_text(text: str) -> str:
    """
    Лемматизация и стемминг текста с удалением стоп-слов.
    """
    words = word_tokenize(text, language="russian")
    processed_words = [morph.parse(word)[0].normal_form for word in words if word.isalpha() and word.lower() not in stop_words]
    return " ".join(processed_words)

def translate_text(text: str) -> str:
    """
    Перевод текста с русского на английский.
    """
    translator = Translator(to_lang="en")
    return translator.translate(text)

def tokenize_ascii(text: str) -> list:
    """
    Токенизация текста посимвольно (только ASCII-символы).
    """
    return [char for char in text if char in string.printable]

def vectorize_ascii(tokens: list) -> np.ndarray:
    """
    Векторизация ASCII-символов с помощью их порядковых номеров.
    """
    return np.array([ord(char) for char in tokens])

if __name__ == "__main__":
    sample_text = "В столовой уже стояли два мальчика и два пидора, сыновья Манилова, которые были в тех летах, когда сажают уже детей за стол, но еще на высоких стульях. При них стоял учитель, поклонившийся вежливо и с улыбкою. Хозяйка села за свою суповую чашку; гость был посажен между хозяином и хозяйкою, слуга завязал детям на шею салфетки."
    
    # Лемматизация и стемминг
    processed_text = preprocess_text(sample_text)
    print("Лемматизированный текст:", processed_text)
    
    # Перевод на английский
    translated_text = translate_text(processed_text)
    print("Переведённый текст:", translated_text)
    
    # Токенизация ASCII
    tokens = tokenize_ascii(translated_text)
    print("Токены ASCII:", tokens)
    
    # Векторизация ASCII
    vectors = vectorize_ascii(tokens)
    print("Векторы ASCII:", vectors)
