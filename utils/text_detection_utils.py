"""Módulo para detecção de texto romanizado e caracteres asiáticos."""
import re

def is_romanized_text(text: str) -> bool:
    """Verifica se o texto está em caracteres ocidentais (romanizados).

    Args:
        text (str): O texto a ser verificado.

    Returns:
        bool: True se o texto estiver em caracteres ocidentais, False caso contrário.
    """
    # Padrão para caracteres japoneses (Hiragana, Katakana, Kanji)
    japanese_pattern = re.compile(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]')
    
    # Padrão para caracteres coreanos (Hangul)
    korean_pattern = re.compile(r'[\uAC00-\uD7AF\u1100-\u11FF]')
    
    # Padrão para caracteres chineses simplificados e tradicionais
    chinese_pattern = re.compile(r'[\u4E00-\u9FFF\u3400-\u4DBF]')
    
    # Verifica se há caracteres asiáticos no texto
    has_japanese = bool(japanese_pattern.search(text))
    has_korean = bool(korean_pattern.search(text))
    has_chinese = bool(chinese_pattern.search(text))
    
    # Retorna True se o texto não contiver caracteres asiáticos
    return not (has_japanese or has_korean or has_chinese)