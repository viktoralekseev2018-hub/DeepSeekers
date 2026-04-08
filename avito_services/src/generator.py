"""
Модуль генерации текстов черновиков объявлений.
"""

import re
from typing import List, Dict, Any

from .detector import normalize_text
from .microcatalog import MC_DICT


def generate_draft_text(description: str, mc_id: int, source_mc_id: int) -> str:
    """
    Генерирует текст черновика объявления для указанной микрокатегории.
    
    Args:
        description: Исходный текст объявления
        mc_id: ID микрокатегории для черновика
        source_mc_id: ID исходной микрокатегории объявления
        
    Returns:
        Сгенерированный текст черновика
    """
    mc = MC_DICT[mc_id]
    normalized = normalize_text(description)
    
    # Извлекаем релевантные предложения/фразы для данной микрокатегории
    relevant_parts = []
    
    # Разбиваем на предложения (упрощенно)
    sentences = re.split(r'[.!?]', description)
    
    for sentence in sentences:
        sent_lower = sentence.lower().strip()
        if not sent_lower:
            continue
            
        # Проверяем, относится ли предложение к данной микрокатегории
        for phrase in mc.keyPhrases:
            if normalize_text(phrase) in sent_lower:
                # Добавляем предложение, очищая от лишнего
                clean_sentence = sentence.strip()
                if clean_sentence and clean_sentence not in relevant_parts:
                    relevant_parts.append(clean_sentence)
                break
    
    # Если не нашли конкретных предложений, используем ключевые фразы
    if not relevant_parts:
        # Берем первые 2-3 ключевые фразы как основу
        for phrase in mc.keyPhrases[:3]:
            relevant_parts.append(f"Выполняем: {phrase}")
    
    # Формируем итоговый текст
    if len(relevant_parts) >= 2:
        draft_text = ". ".join(relevant_parts[:4]) + "."
    else:
        draft_text = relevant_parts[0] if relevant_parts else f"Выполняем услуги по направлению: {mc.mcTitle}"
    
    # Добавляем информацию о том, что услуга предоставляется отдельно
    if "отдельно" not in draft_text.lower():
        draft_text = f"Отдельно выполняем: {draft_text.lower()}"
    
    return draft_text


def create_draft(description: str, mc_id: int, source_mc_id: int) -> Dict[str, Any]:
    """
    Создает черновик объявления для указанной микрокатегории.
    
    Args:
        description: Исходный текст объявления
        mc_id: ID микрокатегории для черновика
        source_mc_id: ID исходной микрокатегории объявления
        
    Returns:
        Словарь с данными черновика
    """
    mc = MC_DICT.get(mc_id)
    if not mc:
        raise ValueError(f"Микрокатегория с ID {mc_id} не найдена")
    
    draft_text = generate_draft_text(description, mc_id, source_mc_id)
    
    return {
        "mcId": mc_id,
        "mcTitle": mc.mcTitle,
        "text": draft_text
    }


def create_drafts(description: str, split_mc_ids: set[int], source_mc_id: int) -> List[Dict[str, Any]]:
    """
    Создает список черновиков для указанных микрокатегорий.
    
    Args:
        description: Исходный текст объявления
        split_mc_ids: Множество ID микрокатегорий для создания черновиков
        source_mc_id: ID исходной микрокатегории объявления
        
    Returns:
        Список словарей с данными черновиков
    """
    drafts = []
    for mc_id in sorted(split_mc_ids):
        draft = create_draft(description, mc_id, source_mc_id)
        drafts.append(draft)
    
    return drafts
