"""
This module is used to translate manga from one language to another.
"""
import os
from deep_translator import GoogleTranslator, MicrosoftTranslator
from dotenv import load_dotenv
from .text_detection_utils import is_romanized_text


load_dotenv()


def translate_manga(text: str, source_lang: str = "auto", target_lang: str = "pt") -> str:
    """
    Translate manga from one language to another.
    """

    if source_lang == target_lang or is_romanized_text(text):
        return text

    translated_text = MicrosoftTranslator(api_key=os.environ['MICROSOFT_API_KEY'],
        target=target_lang, region='brazilsouth').translate(text)
    print("Original text:", text)
    print("Translated text:", translated_text)

    return translated_text