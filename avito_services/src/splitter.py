"""
Модуль определения необходимости сплита (создания дополнительных черновиков).
"""

import re
from typing import Tuple, Set

from .detector import normalize_text
from .microcatalog import MC_DICT


def is_service_offered_separately(description: str, mc_id: int, source_mc_id: int) -> bool:
    """
    Определяет, предлагается ли услуга данной микрокатегории отдельно.
    
    Args:
        description: Текст объявления
        mc_id: ID микрокатегории
        source_mc_id: ID исходной микрокатегории объявления
        
    Returns:
        True, если услуга предлагается отдельно
    """
    normalized = normalize_text(description)
    
    # Ключевые индикаторы отдельного предложения услуги
    separate_indicators = [
        "отдельно", "как самостоятельную работу", "как самостоятельную услугу",
        "можно заказать отдельно", "берем отдельно", "делаем отдельно",
        "выполняем отдельно", "предлагаем отдельно", "заказывайте отдельно",
    ]
    
    has_separate = any(indicator in normalized for indicator in separate_indicators)
    
    # Проверяем контекст вокруг ключевых фраз микрокатегории
    mc = MC_DICT.get(mc_id)
    if not mc:
        return False
    
    # Ищем паттерны с "отдельно" рядом с ключевыми фразами
    for phrase in mc.keyPhrases[:10]:  # Проверяем первые 10 фраз
        norm_phrase = normalize_text(phrase)
        
        # Паттерн: "отдельно" + фраза или фраза + "отдельно"
        pattern1 = rf"отдельно\s+(?:\w+\s+)?{re.escape(norm_phrase)}"
        pattern2 = rf"{re.escape(norm_phrase)}(?:\s+\w+)?\s+отдельно"
        
        if re.search(pattern1, normalized) or re.search(pattern2, normalized):
            return True
        
        # Паттерн: "можем/делаем/выполняем" + фраза
        pattern3 = rf"(?:можем|делаем|выполняем)\s+{re.escape(norm_phrase)}"
        if re.search(pattern3, normalized):
            return True
    
    # Если это исходная микрокатегория и нет явных признаков отдельной услуги - не сплитовать
    if mc_id == source_mc_id:
        return False
    
    # Для non-turnkey категорий (не 101) - если услуга найдена и не является частью комплекса
    if mc_id != 101 and source_mc_id != 101:
        # Проверяем, не является ли упоминание частью перечисления в комплексе
        if "как часть ремонта" in normalized or "в составе" in normalized:
            return False
        # Если есть явное перечисление через "/" или запятую - может быть отдельной
        if "/" in description or "также" in normalized:
            return True
    
    return has_separate


def should_split_announcement(
    description: str,
    detected_mc_ids: Set[int],
    source_mc_id: int
) -> Tuple[bool, Set[int]]:
    """
    Определяет, нужно ли создавать дополнительные черновики объявлений.
    
    Args:
        description: Текст объявления
        detected_mc_ids: Множество ID обнаруженных микрокатегорий
        source_mc_id: ID исходной микрокатегории объявления
        
    Returns:
        Кортеж (shouldSplit, split_mc_ids)
    """
    normalized = normalize_text(description)
    
    # Исходная микрокатегория не включается в сплит
    candidate_ids = detected_mc_ids - {source_mc_id}
    
    if not candidate_ids:
        return False, set()
    
    # Проверяем явные индикаторы работы только в комплексе
    complex_only_phrases = [
        "по отдельным видам работ не выезжаю",
        "ищу заказы именно на комплекс",
        "работаем только в комплексе",
        "без дробления на этапы",
        "под ключ без дробления",
        "все этапы выполняем как часть ремонта",
        "выполняем в составе ремонта",
        "другие этапы выполняем как часть ремонта",
        "этапы выполняем все работы одной бригадой",
    ]
    
    is_complex_only = any(phrase in normalized for phrase in complex_only_phrases)
    
    # Если явно указано, что работают только в комплексе - не сплитовать
    if is_complex_only:
        # Но проверяем, есть ли явные указания на отдельные услуги
        has_explicit_separate = False
        for mc_id in candidate_ids:
            if is_service_offered_separately(description, mc_id, source_mc_id):
                has_explicit_separate = True
                break
        
        if not has_explicit_separate:
            return False, set()
    
    # Определяем, какие микрокатегории предлагаются отдельно
    split_ids = set()
    for mc_id in candidate_ids:
        if is_service_offered_separately(description, mc_id, source_mc_id):
            split_ids.add(mc_id)
    
    # Если есть хотя бы одна микрокатегория для сплита
    if split_ids:
        return True, split_ids
    
    return False, set()
