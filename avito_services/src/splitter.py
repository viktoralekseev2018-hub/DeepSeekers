"""
Модуль определения необходимости сплита (создания дополнительных черновиков).
Использует ML-подход с классификацией на основе признаков контекста.
"""

import re
from typing import Tuple, Set, List, Dict
from collections import defaultdict

from .detector import normalize_text
from .microcatalog import MC_DICT


def extract_split_features(description: str, mc_id: int, source_mc_id: int) -> Dict[str, float]:
    """
    Извлекает признаки для принятия решения о сплите.

    Args:
        description: Текст объявления
        mc_id: ID микрокатегории
        source_mc_id: ID исходной микрокатегории объявления

    Returns:
        Словарь с признаками и их значениями
    """
    normalized = normalize_text(description)
    features = {}

    # Признак 1: Наличие индикаторов отдельной услуги
    separate_indicators = [
        "отдельно", "как самостоятельную работу", "как самостоятельную услугу",
        "можно заказать отдельно", "берем отдельно", "делаем отдельно",
        "выполняем отдельно", "предлагаем отдельно", "заказывайте отдельно",
        "выполняем как отдельную", "предоставляем отдельно",
    ]
    features['has_separate_indicator'] = float(any(ind in normalized for ind in separate_indicators))

    # Признак 2: Контекст рядом с ключевыми фразами
    mc = MC_DICT.get(mc_id)
    nearby_separate = 0.0
    if mc:
        for phrase in mc.keyPhrases[:15]:
            norm_phrase = normalize_text(phrase)
            # Паттерн: "отдельно" + фраза или фраза + "отдельно" (в пределах 5 слов)
            pattern1 = rf"(?:отдельно|самостоятельно)\s+(?:\w+\s+)?(?:\w+\s+)?{re.escape(norm_phrase)}"
            pattern2 = rf"{re.escape(norm_phrase)}(?:\s+\w+)?(?:\s+\w+)?\s+(?:отдельно|самостоятельно)"

            if re.search(pattern1, normalized) or re.search(pattern2, normalized):
                nearby_separate = 1.0
                break

            # Паттерн: модальные глаголы + фраза
            pattern3 = rf"(?:можем|делаем|выполняем|предлагаем)\s+(?:также\s+)?{re.escape(norm_phrase)}"
            if re.search(pattern3, normalized) and features.get('has_separate_indicator', 0) > 0:
                nearby_separate = max(nearby_separate, 0.8)

    features['nearby_separate_context'] = nearby_separate

    # Признак 3: Индикаторы работы только в комплексе
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
        "только комплексно", "комплексно и точка",
    ]
    features['is_complex_only'] = float(any(phrase in normalized for phrase in complex_only_phrases))

    # Признак 4: Структура перечисления (слэши, запятые, маркированные списки)
    has_slash_list = float("/" in description and len(description.split("/")) > 2)
    has_bullet_list = float(any(marker in description for marker in ["\n-", "\n*", "\n•"]))
    has_numbered_list = float(bool(re.search(r'\d+[\.)]', description)))

    features['has_slash_list'] = has_slash_list
    features['has_bullet_list'] = has_bullet_list
    features['has_numbered_list'] = has_numbered_list

    # Признак 5: Длина объявления
    desc_length = len(description)
    features['is_short'] = float(desc_length < 100)
    features['is_medium'] = float(100 <= desc_length <= 300)
    features['is_long'] = float(desc_length > 300)

    # Признак 6: Это исходная микрокатегория или нет
    features['is_source_mc'] = float(mc_id == source_mc_id)

    # Признак 7: Является ли категория "ремонт под ключ"
    features['is_turnkey'] = float(source_mc_id == 101)

    # Признак 8: Наличие явных маркеров отдельных услуг в списке
    if "/" in description or has_bullet_list:
        parts = re.split(r'[/\n]', description)
        service_count = 0
        for part in parts:
            part_norm = normalize_text(part)
            # Проверяем, содержит ли часть ключевые фразы данной МК
            if mc:
                for phrase in mc.keyPhrases[:10]:
                    if normalize_text(phrase) in part_norm:
                        service_count += 1
                        break
        features['service_in_list_count'] = min(service_count / 3.0, 1.0)  # Нормализуем
    else:
        features['service_in_list_count'] = 0.0

    return features


def predict_split_probability(features: Dict[str, float]) -> float:
    """
    Вычисляет вероятность необходимости сплита на основе признаков.
    Использует взвешенную комбинацию признаков (упрощенная логистическая регрессия).

    Args:
        features: Словарь с признаками

    Returns:
        Вероятность сплита (0.0 - 1.0)
    """
    # Веса признаков (обучаются на данных, здесь заданы эвристически)
    weights = {
        'has_separate_indicator': 2.5,
        'nearby_separate_context': 2.0,
        'is_complex_only': -3.0,
        'has_slash_list': 0.8,
        'has_bullet_list': 1.0,
        'has_numbered_list': 0.5,
        'is_short': -0.5,
        'is_medium': 0.2,
        'is_long': 0.3,
        'is_source_mc': -1.5,
        'is_turnkey': 0.3,
        'service_in_list_count': 1.2,
    }

    bias = -1.0  # Смещение

    # Вычисляем взвешенную сумму
    score = bias
    for feature_name, value in features.items():
        weight = weights.get(feature_name, 0.0)
        score += weight * value

    # Применяем сигмоиду для получения вероятности
    probability = 1.0 / (1.0 + (2.71828 ** (-score)))

    return probability


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
    features = extract_split_features(description, mc_id, source_mc_id)
    probability = predict_split_probability(features)

    # Порог для принятия решения
    threshold = 0.5

    # Если это исходная микрокатегория - повышаем порог
    if mc_id == source_mc_id:
        threshold = 0.7

    # Если явно указано "только комплекс" - понижаем вероятность
    if features.get('is_complex_only', 0) > 0 and features.get('has_separate_indicator', 0) == 0:
        return False

    return probability >= threshold


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

    # Определяем, какие микрокатегории предлагаются отдельно
    split_ids = set()
    for mc_id in candidate_ids:
        # Если явно указано "только комплекс", проверяем только при наличии явных признаков отдельных услуг
        if is_complex_only:
            features = extract_split_features(description, mc_id, source_mc_id)
            if features.get('has_separate_indicator', 0) > 0:
                if is_service_offered_separately(description, mc_id, source_mc_id):
                    split_ids.add(mc_id)
        else:
            if is_service_offered_separately(description, mc_id, source_mc_id):
                split_ids.add(mc_id)

    # Если есть хотя бы одна микрокатегория для сплита
    if split_ids:
        return True, split_ids

    return False, set()