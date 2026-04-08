"""
Основной процессор объявлений.
Объединяет детекцию, определение сплита и генерацию черновиков.
"""

from typing import Dict, Any, List

from .detector import detect_microcategories
from .splitter import should_split_announcement
from .generator import create_drafts


def process_advertisement(
    item_id: int,
    source_mc_id: int,
    source_mc_title: str,
    description: str
) -> Dict[str, Any]:
    """
    Обрабатывает одно объявление и возвращает результат.
    
    Args:
        item_id: Уникальный идентификатор объявления
        source_mc_id: ID исходной микрокатегории объявления
        source_mc_title: Название исходной микрокатегории
        description: Текст объявления
        
    Returns:
        Словарь с результатами обработки:
        - itemId: ID объявления
        - detectedMcIds: Список ID обнаруженных микрокатегорий
        - shouldSplit: Флаг необходимости создания черновиков
        - drafts: Список черновиков объявлений
    """
    # Шаг 1: Определяем все микрокатегории в тексте
    detected_ids = detect_microcategories(description)
    
    # Добавляем исходную микрокатегорию, если она не была обнаружена
    # (она может не иметь явных ключевых фраз в тексте)
    detected_ids.add(source_mc_id)
    
    # Шаг 2: Определяем, нужно ли сплитовать
    should_split, split_ids = should_split_announcement(description, detected_ids, source_mc_id)
    
    # Шаг 3: Генерируем черновики
    drafts = []
    if should_split:
        drafts = create_drafts(description, split_ids, source_mc_id)
    
    return {
        "itemId": item_id,
        "detectedMcIds": sorted(list(detected_ids)),
        "shouldSplit": should_split,
        "drafts": drafts
    }


def evaluate_predictions(predictions: List[Dict], ground_truth: List[Dict]) -> Dict[str, float]:
    """
    Вычисляет метрики качества: Precision, Recall, F1-score (micro) и Accuracy по shouldSplit.
    
    Args:
        predictions: Список предсказаний модели
        ground_truth: Список эталонных значений
        
    Returns:
        Словарь с метриками:
        - precision_micro
        - recall_micro
        - f1_score_micro
        - accuracy_should_split
    """
    # Для micro-averaging собираем все TP, FP, FN
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    correct_should_split = 0
    total_samples = len(predictions)
    
    for pred, truth in zip(predictions, ground_truth):
        pred_ids = set(pred.get('detectedMcIds', []))
        truth_detected = set(truth.get('targetDetectedMcIds', []))
        truth_split = set(truth.get('targetSplitMcIds', []))
        
        # Предсказанные для сплита (из черновиков)
        pred_split_ids = set(d['mcId'] for d in pred.get('drafts', []))
        
        # TP: правильно предсказанные микрокатегории для сплита
        tp = len(pred_split_ids & truth_split)
        # FP: предсказанные, но не в эталоне
        fp = len(pred_split_ids - truth_split)
        # FN: в эталоне, но не предсказанные
        fn = len(truth_split - pred_split_ids)
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
        
        # Accuracy для shouldSplit
        if pred['shouldSplit'] == truth['shouldSplit']:
            correct_should_split += 1
    
    # Вычисляем метрики
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = correct_should_split / total_samples if total_samples > 0 else 0.0
    
    return {
        "precision_micro": precision,
        "recall_micro": recall,
        "f1_score_micro": f1_score,
        "accuracy_should_split": accuracy
    }
