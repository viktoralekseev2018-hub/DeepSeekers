"""
Модуль генерации текстов черновиков объявлений.
Использует ML-подход для извлечения релевантных фрагментов текста.
"""

import re
from typing import List, Dict, Any, Tuple
from collections import defaultdict

from .detector import normalize_text, extract_sentences
from .microcatalog import MC_DICT


def calculate_relevance_score(sentence: str, mc_id: int, description: str) -> float:
    """
    Вычисляет релевантность предложения для данной микрокатегории.

    Args:
        sentence: Предложение/фраза из текста
        mc_id: ID микрокатегории
        description: Полный текст объявления

    Returns:
        Оценка релевантности (0.0 - 1.0)
    """
    mc = MC_DICT.get(mc_id)
    if not mc:
        return 0.0

    normalized_sentence = normalize_text(sentence)
    normalized_description = normalize_text(description)

    score = 0.0
    max_possible_score = 0.0

    # Проверка наличия ключевых фраз в предложении
    for phrase in mc.keyPhrases:
        norm_phrase = normalize_text(phrase)

        # Вес фразы зависит от её специфичности
        phrase_weight = 1.0
        if len(norm_phrase.split()) >= 3:
            phrase_weight = 1.5  # Длинные фразы важнее
        elif len(norm_phrase.split()) == 2:
            phrase_weight = 1.2

        if norm_phrase in normalized_sentence:
            score += phrase_weight
            max_possible_score += phrase_weight

            # Бонус за точное совпадение с важными фразами
            important_phrases = [
                "отдельно", "самостоятельно", "как отдельную",
                "выполняем", "делаем", "предлагаем", "можем",
            ]
            for imp_phrase in important_phrases:
                if imp_phrase in normalized_sentence:
                    score += 0.3
                    break

    # Нормализуем оценку
    if max_possible_score > 0:
        score = min(score / max_possible_score, 1.0)
    else:
        score = 0.0

    return score


def extract_relevant_fragments(description: str, mc_id: int) -> List[Tuple[str, float]]:
    """
    Извлекает релевантные фрагменты текста для данной микрокатегории.

    Args:
        description: Исходный текст объявления
        mc_id: ID микрокатегории

    Returns:
        Список кортежей (фрагмент, оценка релевантности)
    """
    mc = MC_DICT.get(mc_id)
    if not mc:
        return []

    relevant_fragments = []

    # Разбиваем на предложения и маркированные элементы
    sentences = extract_sentences(description)

    # Также обрабатываем маркированные списки
    bullet_lines = []
    if any(marker in description for marker in ["\n-", "\n*", "\n•"]):
        lines = description.split("\n")
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("-") or stripped.startswith("*") or stripped.startswith("•"):
                bullet_lines.append(stripped[1:].strip())

    all_fragments = sentences + bullet_lines

    # Оцениваем каждый фрагмент
    for fragment in all_fragments:
        if not fragment.strip():
            continue

        relevance = calculate_relevance_score(fragment, mc_id, description)

        if relevance > 0.2:  # Порог релевантности
            relevant_fragments.append((fragment.strip(), relevance))

    # Сортируем по релевантности
    relevant_fragments.sort(key=lambda x: x[1], reverse=True)

    return relevant_fragments


def generate_draft_text(description: str, mc_id: int, source_mc_id: int) -> str:
    """
    Генерирует текст черновика объявления для указанной микрокатегории.
    Использует ML-подход для выбора наиболее релевантных фрагментов.

    Args:
        description: Исходный текст объявления
        mc_id: ID микрокатегории для черновика
        source_mc_id: ID исходной микрокатегории объявления

    Returns:
        Сгенерированный текст черновика
    """
    mc = MC_DICT[mc_id]

    # Извлекаем релевантные фрагменты
    relevant_fragments = extract_relevant_fragments(description, mc_id)

    if not relevant_fragments:
        # Если не нашли релевантных фрагментов, используем ключевые фразы
        key_phrases = mc.keyPhrases[:3]
        draft_text = f"Выполняем услуги: {', '.join(key_phrases)}."
    else:
        # 3 наиболее релевантных фрагмента
        top_fragments = [f[0] for f in relevant_fragments[:3]]

        # Объединяем фрагменты в связный текст
        if len(top_fragments) >= 2:
            draft_text = ". ".join(top_fragments)
            if not draft_text.endswith("."):
                draft_text += "."
        else:
            draft_text = top_fragments[0]
            if not draft_text.endswith("."):
                draft_text += "."

    # Проверяем, нужно ли добавить информацию о том, что услуга предоставляется отдельно
    normalized_draft = normalize_text(draft_text)
    if "отдельно" not in normalized_draft and "самостоятельн" not in normalized_draft:
        # Добавляем префикс
        draft_text = "Отдельно выполняем: " + draft_text.lower()

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