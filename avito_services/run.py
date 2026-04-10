#!/usr/bin/env python3
"""
Скрипт запуска обработки датасета и оценки качества модели.
"""

import json
import ast
import os
import sys

# Добавляем родительскую директорию в path для импорта src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.processor import process_advertisement, evaluate_predictions


def load_dataset(filepath: str) -> list[dict]:
    """Загружает датасет из CSV файла."""
    df = pd.read_csv(filepath)
    
    records = []
    for _, row in df.iterrows():
        record = {
            'itemId': int(row['itemId']),
            'sourceMcId': int(row['sourceMcId']),
            'sourceMcTitle': row['sourceMcTitle'],
            'description': row['description'],
            'targetDetectedMcIds': ast.literal_eval(row['targetDetectedMcIds']) if isinstance(row['targetDetectedMcIds'], str) else row['targetDetectedMcIds'],
            'targetSplitMcIds': ast.literal_eval(row['targetSplitMcIds']) if isinstance(row['targetSplitMcIds'], str) else row['targetSplitMcIds'],
            'shouldSplit': bool(row['shouldSplit']),
            'caseType': row['caseType'],
            'split': row['split']
        }
        records.append(record)
    
    return records


def main():
    """Основная функция для обработки датасета и оценки качества."""
    # Определяем путь к датасету
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'data')
    
    dataset_path = os.path.join(data_dir, 'rnc_dataset.csv')
    
    if not os.path.exists(dataset_path):
        print(f"Ошибка: Датасет не найден по пути {dataset_path}")
        print("Пожалуйста, поместите файл rnc_dataset.csv в папку data/")
        sys.exit(1)
    
    print("Загрузка датасета...")
    data = load_dataset(dataset_path)
    print(f"Загружено {len(data)} записей")
    
    # Обработка всех объявлений
    print("\nОбработка объявлений...")
    predictions = []
    for i, record in enumerate(data):
        result = process_advertisement(
            item_id=record['itemId'],
            source_mc_id=record['sourceMcId'],
            source_mc_title=record['sourceMcTitle'],
            description=record['description']
        )
        predictions.append(result)
        
        if (i + 1) % 500 == 0:
            print(f"  Обработано {i + 1}/{len(data)}")
    
    # Оценка качества
    print("\nОценка качества...")
    metrics = evaluate_predictions(predictions, data)
    
    print("\n" + "="*50)
    print("РЕЗУЛЬТАТЫ:")
    print("="*50)
    print(f"Precision (micro): {metrics['precision_micro']:.4f}")
    print(f"Recall (micro):    {metrics['recall_micro']:.4f}")
    print(f"F1-score (micro):  {metrics['f1_score_micro']:.4f}")
    print(f"Accuracy shouldSplit: {metrics['accuracy_should_split']:.4f}")
    print("="*50)
    
    # Примеры результатов
    print("\nПримеры обработки:")
    for i in range(min(5, len(predictions))):
        pred = predictions[i]
        truth = data[i]
        print(f"\n--- Объявление #{pred['itemId']} ---")
        print(f"Исходная МК: {truth['sourceMcTitle']}")
        print(f"Текст: {truth['description'][:100]}...")
        print(f"detectedMcIds: {pred['detectedMcIds']}")
        print(f"Эталон detected: {truth['targetDetectedMcIds']}")
        print(f"shouldSplit: {pred['shouldSplit']} (эталон: {truth['shouldSplit']})")
        if pred['drafts']:
            print("Черновики:")
            for draft in pred['drafts']:
                print(f"  - {draft['mcTitle']}: {draft['text'][:80]}...")
    
    # Сохранение результатов
    output_path = os.path.join(script_dir, 'predictions.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    print(f"\nРезультаты сохранены в {output_path}")


if __name__ == "__main__":
    main()
