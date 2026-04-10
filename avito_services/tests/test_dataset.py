"""
Тесты для процессора объявлений на основе тестовой выборки датасета rnc_dataset.
Запускает обработку объявлений из test split и сравнивает результаты с ground truth.
"""

import unittest
import sys
import os
import csv
import ast
from typing import List, Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.processor import process_advertisement, evaluate_predictions


def load_test_dataset(csv_path: str) -> List[Dict[str, Any]]:
    """
    Загружает тестовую выборку из CSV файла.

    Args:
        csv_path: Путь к CSV файлу с датасетом

    Returns:
        Список словарей с данными объявлений из test split
    """
    test_data = []

    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Фильтруем только test split
            if row['split'] == 'test':
                item = {
                    'itemId': int(row['itemId']),
                    'sourceMcId': int(row['sourceMcId']),
                    'sourceMcTitle': row['sourceMcTitle'],
                    'description': row['description'],
                    'targetDetectedMcIds': ast.literal_eval(row['targetDetectedMcIds']),
                    'targetSplitMcIds': ast.literal_eval(row['targetSplitMcIds']),
                    'shouldSplit': row['shouldSplit'] == 'True',
                    'caseType': row['caseType']
                }
                test_data.append(item)

    return test_data


def get_ground_truth_path() -> str:
    """Определяет путь к датасету."""
    # Проверяем возможные пути
    possible_paths = [
        os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'rnc_dataset.csv'),
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'case 3', 'rnc_dataset.csv'),
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'case3', 'rnc_dataset.csv'),
    ]

    for path in possible_paths:
        if os.path.exists(path):
            return path

    # Если ни один путь не найден, возвращаем первый (будет ошибка)
    return possible_paths[0]


class TestDatasetProcessor(unittest.TestCase):
    """Тесты процессора на реальных данных из датасета."""

    @classmethod
    def setUpClass(cls):
        """Загружает тестовый датасет один раз для всех тестов."""
        dataset_path = get_ground_truth_path()
        cls.test_data = load_test_dataset(dataset_path)
        print(f"\nЗагружено {len(cls.test_data)} объявлений из test split")

        # Статистика по caseType
        case_types = {}
        for item in cls.test_data:
            ct = item['caseType']
            case_types[ct] = case_types.get(ct, 0) + 1
        print(f"Распределение caseType: {case_types}")

    def test_all_test_samples(self):
        """
        Интеграционный тест: обрабатывает все объявления из test split
        и вычисляет метрики качества.
        """
        predictions = []
        ground_truth = []

        # Примеры для вывода (первые 5 объявлений разных типов)
        examples_shown = set()
        max_examples = 5

        for item in self.test_data:
            # Обрабатываем объявление
            result = process_advertisement(
                item_id=item['itemId'],
                source_mc_id=item['sourceMcId'],
                source_mc_title=item['sourceMcTitle'],
                description=item['description']
            )

            # Выводим примеры объявлений и черновиков
            if len(examples_shown) < max_examples and item['caseType'] not in examples_shown:
                examples_shown.add(item['caseType'])
                print(f"\n{'='*80}")
                print(f"ПРИМЕР ОБЪЯВЛЕНИЯ (caseType: {item['caseType']})")
                print(f"{'='*80}")
                print(f"Item ID: {item['itemId']}")
                print(f"Source Microcategory: {item['sourceMcTitle']} (ID: {item['sourceMcId']})")
                print(f"\nОписание:")
                print(f"{item['description'][:500]}{'...' if len(item['description']) > 500 else ''}")
                print(f"\nGround Truth:")
                print(f"  shouldSplit: {item['shouldSplit']}")
                print(f"  targetDetectedMcIds: {item['targetDetectedMcIds']}")
                print(f"  targetSplitMcIds: {item['targetSplitMcIds']}")

                print(f"\nРезультат обработки:")
                print(f"  shouldSplit: {result['shouldSplit']}")
                print(f"  detectedMcIds: {result['detectedMcIds']}")
                # splitMcIds извлекаем из черновиков
                split_ids = [d['mcId'] for d in result.get('drafts', [])]
                print(f"  splitMcIds (из черновиков): {split_ids}")

                if result['shouldSplit'] and result['drafts']:
                    print(f"\nЧерновики ({len(result['drafts'])} шт.):")
                    for i, draft in enumerate(result['drafts'][:3], 1):  # Показываем первые 3
                        print(f"\n  Черновик #{i}:")
                        print(f"    mcId: {draft['mcId']}")
                        print(f"    mcTitle: {draft['mcTitle']}")
                        print(f"    text: {draft['text'][:200]}{'...' if len(draft['text']) > 200 else ''}")
                    if len(result['drafts']) > 3:
                        print(f"\n  ... и ещё {len(result['drafts']) - 3} черновиков")
                print(f"{'='*80}")

            predictions.append(result)
            ground_truth.append({
                'itemId': item['itemId'],
                'targetDetectedMcIds': item['targetDetectedMcIds'],
                'targetSplitMcIds': item['targetSplitMcIds'],
                'shouldSplit': item['shouldSplit']
            })

        # Вычисляем метрики
        metrics = evaluate_predictions(predictions, ground_truth)

        print(f"\nМетрики на test выборке:")
        print(f"  Precision (micro): {metrics['precision_micro']:.4f}")
        print(f"  Recall (micro): {metrics['recall_micro']:.4f}")
        print(f"  F1-score (micro): {metrics['f1_score_micro']:.4f}")
        print(f"  Accuracy shouldSplit: {metrics['accuracy_should_split']:.4f}")

        # Asserts для базового качества
        # F1-score выше 0.3 (базовый порог)
        self.assertGreater(metrics['f1_score_micro'], 0.3,
                          "F1-score слишком низкий")

        self.assertGreater(metrics['accuracy_should_split'], 0.6,
                          "Accuracy по shouldSplit слишком низкий")


class TestCaseTypeSpecific(unittest.TestCase):
    """Специфические тесты для разных типов кейсов."""

    @classmethod
    def setUpClass(cls):
        """Загружает тестовый датасет."""
        dataset_path = get_ground_truth_path()
        all_test_data = load_test_dataset(dataset_path)

        # Группируем по caseType
        cls.by_case_type = {}
        for item in all_test_data:
            ct = item['caseType']
            if ct not in cls.by_case_type:
                cls.by_case_type[ct] = []
            cls.by_case_type[ct].append(item)

        print(f"\nДоступные caseType: {list(cls.by_case_type.keys())}")

    def test_single_direct(self):
        """Тест:单一 услуга без сплита (single_direct)."""
        if 'single_direct' not in self.by_case_type:
            self.skipTest("Нет данных single_direct в test split")

        correct = 0
        total = len(self.by_case_type['single_direct'])

        # Показываем первый пример
        example_shown = False

        for item in self.by_case_type['single_direct'][:20]:  # Первые 20
            result = process_advertisement(
                item_id=item['itemId'],
                source_mc_id=item['sourceMcId'],
                source_mc_title=item['sourceMcTitle'],
                description=item['description']
            )

            # Показываем первый пример
            if not example_shown:
                example_shown = True
                print(f"\n{'='*60}")
                print(f"ПРИМЕР single_direct:")
                print(f"  Описание: {item['description'][:150]}...")
                print(f"  shouldSplit: {result['shouldSplit']} (ожидание: False)")
                print(f"  detectedMcIds: {result['detectedMcIds']}")
                print(f"{'='*60}")

            # Для single_direct не должно быть сплита
            if not result['shouldSplit']:
                correct += 1

        accuracy = correct / min(total, 20)
        print(f"\nsingle_direct accuracy (no split): {accuracy:.2f} ({correct}/{min(total, 20)})")
        self.assertGreater(accuracy, 0.7, "Низкая точность для single_direct")

    def test_turnkey_split(self):
        """Тест: ремонт под ключ со сплитом (turnkey_split)."""
        if 'turnkey_split' not in self.by_case_type:
            self.skipTest("Нет данных turnkey_split в test split")

        correct = 0
        total = len(self.by_case_type['turnkey_split'])

        # Показываем первый пример с черновиками
        example_shown = False

        for item in self.by_case_type['turnkey_split'][:20]:
            result = process_advertisement(
                item_id=item['itemId'],
                source_mc_id=item['sourceMcId'],
                source_mc_title=item['sourceMcTitle'],
                description=item['description']
            )

            # Показываем первый пример
            if not example_shown:
                example_shown = True
                print(f"\n{'='*60}")
                print(f"ПРИМЕР turnkey_split:")
                print(f"  Описание: {item['description'][:150]}...")
                print(f"  shouldSplit: {result['shouldSplit']} (ожидание: True)")
                if result['drafts']:
                    print(f"  Черновики ({len(result['drafts'])} шт.):")
                    for i, draft in enumerate(result['drafts'][:2], 1):
                        print(f"    #{i}: mcId={draft['mcId']}, mcTitle={draft['mcTitle']}")
                print(f"{'='*60}")

            if result['shouldSplit']:
                correct += 1

        accuracy = correct / min(total, 20)
        print(f"\nturnkey_split accuracy (should split): {accuracy:.2f} ({correct}/{min(total, 20)})")
        self.assertGreater(accuracy, 0.5, "Низкая точность для turnkey_split")

    def test_turnkey_no_split(self):
        """Тест: ремонт под ключ без сплита (turnkey_no_split)."""
        if 'turnkey_no_split' not in self.by_case_type:
            self.skipTest("Нет данных turnkey_no_split в test split")

        correct = 0
        total = len(self.by_case_type['turnkey_no_split'])

        for item in self.by_case_type['turnkey_no_split'][:20]:
            result = process_advertisement(
                item_id=item['itemId'],
                source_mc_id=item['sourceMcId'],
                source_mc_title=item['sourceMcTitle'],
                description=item['description']
            )

            # Для turnkey_no_split не должно быть сплита
            if not result['shouldSplit']:
                correct += 1

        accuracy = correct / min(total, 20)
        print(f"\nturnkey_no_split accuracy (no split): {accuracy:.2f} ({correct}/{min(total, 20)})")
        self.assertGreater(accuracy, 0.5, "Низкая точность для turnkey_no_split")

    def test_multi_service_split(self):
        """Тест: несколько услуг со сплитом (multi_service_split)."""
        if 'multi_service_split' not in self.by_case_type:
            self.skipTest("Нет данных multi_service_split в test split")

        correct = 0
        total = len(self.by_case_type['multi_service_split'])

        for item in self.by_case_type['multi_service_split'][:20]:
            result = process_advertisement(
                item_id=item['itemId'],
                source_mc_id=item['sourceMcId'],
                source_mc_title=item['sourceMcTitle'],
                description=item['description']
            )

            # Для multi_service_split должен быть сплит
            if result['shouldSplit']:
                correct += 1

        accuracy = correct / min(total, 20)
        print(f"\nmulti_service_split accuracy (should split): {accuracy:.2f} ({correct}/{min(total, 20)})")
        self.assertGreater(accuracy, 0.5, "Низкая точность для multi_service_split")


class TestDetectorOnDataset(unittest.TestCase):
    """Тесты детектора на данных из датасета."""

    @classmethod
    def setUpClass(cls):
        """Загружает тестовый датасет."""
        dataset_path = get_ground_truth_path()
        cls.test_data = load_test_dataset(dataset_path)

    def test_detection_recall(self):
        """Тест: полнота обнаружения микрокатегорий."""
        from src.detector import detect_microcategories

        total_target = 0
        total_detected = 0

        for item in self.test_data[:50]:  # Первые 50 объявлений
            detected = detect_microcategories(item['description'])
            # Добавляем исходную микрокатегорию
            detected.add(item['sourceMcId'])

            target = set(item['targetDetectedMcIds'])

            total_target += len(target)
            total_detected += len(detected & target)

        recall = total_detected / total_target if total_target > 0 else 0
        print(f"\nDetection recall на 50 примерах: {recall:.4f}")

        # Recall хотя бы 0.5
        self.assertGreater(recall, 0.5, "Низкая полнота обнаружения")


class TestDraftGeneration(unittest.TestCase):
    """Тесты генерации черновиков."""

    def test_draft_has_required_fields(self):
        """Тест: черновики имеют все требуемые поля."""
        result = process_advertisement(
            item_id=1,
            source_mc_id=101,
            source_mc_title="Ремонт квартир и домов под ключ",
            description="Ремонт под ключ. Отдельно сантехника, отдельно электрика."
        )

        if result['shouldSplit']:
            for draft in result['drafts']:
                self.assertIn('mcId', draft)
                self.assertIn('mcTitle', draft)
                self.assertIn('text', draft)
                self.assertIsInstance(draft['mcId'], int)
                self.assertIsInstance(draft['mcTitle'], str)
                self.assertIsInstance(draft['text'], str)
                self.assertGreater(len(draft['text']), 0)


if __name__ == '__main__':
    # Запускаем тесты с выводом подробной информации
    unittest.main(verbosity=2)