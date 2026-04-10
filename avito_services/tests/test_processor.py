"""
Тесты для процессора объявлений.
"""

import unittest
import sys
import os

# Добавляем родительскую директорию в path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.processor import process_advertisement, evaluate_predictions


class TestProcessor(unittest.TestCase):
    """Тесты для основного процессора."""
    
    def test_single_service_no_split(self):
        """Тест: 0 一 услуга без сплита."""
        result = process_advertisement(
            item_id=1,
            source_mc_id=108,
            source_mc_title="Штукатурные работы",
            description="штукатурка под обои в офис. Делаю ремонт штукатурки, поможем с материалом."
        )
        
        self.assertEqual(result['itemId'], 1)
        self.assertIn(108, result['detectedMcIds'])
        self.assertFalse(result['shouldSplit'])
        self.assertEqual(len(result['drafts']), 0)
    
    def test_turnkey_with_separate_services(self):
        """Тест: ремонт под ключ с отдельными услугами."""
        result = process_advertisement(
            item_id=2,
            source_mc_id=101,
            source_mc_title="Ремонт квартир и домов под ключ",
            description="Косметический ремонт под ключ в доме. Отдельно демонтаж сантехники делаем отдельно цементная штукатурка отдельно декоративные конструкции из гкл делаем отдельно потолок в коридоре."
        )
        
        self.assertTrue(result['shouldSplit'])
        self.assertGreater(len(result['drafts']), 0)
    
    def test_complex_only_no_split(self):
        """Тест: комплексные работы без сплита."""
        result = process_advertisement(
            item_id=5,
            source_mc_id=101,
            source_mc_title="Ремонт квартир и домов под ключ",
            description="Комплекс работ по ремонту во вторичке. гкл, гкл и другие этапы выполняем как часть ремонта. Ищу заказы именно на комплекс, по отдельным видам работ не выезжаю."
        )
        
        self.assertFalse(result['shouldSplit'])
        self.assertEqual(len(result['drafts']), 0)
    
    def test_multi_service_split(self):
        """Тест: несколько услуг со сплитом."""
        result = process_advertisement(
            item_id=22,
            source_mc_id=105,
            source_mc_title="Укладка плитки",
            description="Укладка плитки елочкой. Можем отдельно ремонт комнаты под ключ."
        )
        
        # Должен быть сплит на ремонт под ключ
        self.assertTrue(result['shouldSplit'])
        self.assertIn(101, [d['mcId'] for d in result['drafts']])
    
    def test_evaluate_predictions_perfect(self):
        """Тест: оценка идеальных предсказаний."""
        predictions = [
            {
                'itemId': 1,
                'detectedMcIds': [101, 102],
                'shouldSplit': True,
                'drafts': [{'mcId': 102, 'mcTitle': 'Сантехника', 'text': 'test'}]
            }
        ]
        ground_truth = [
            {
                'itemId': 1,
                'targetDetectedMcIds': [101, 102],
                'targetSplitMcIds': [102],
                'shouldSplit': True
            }
        ]
        
        metrics = evaluate_predictions(predictions, ground_truth)
        
        self.assertEqual(metrics['precision_micro'], 1.0)
        self.assertEqual(metrics['recall_micro'], 1.0)
        self.assertEqual(metrics['f1_score_micro'], 1.0)
        self.assertEqual(metrics['accuracy_should_split'], 1.0)
    
    def test_evaluate_predictions_wrong(self):
        """Тест: оценка неправильных предсказаний."""
        predictions = [
            {
                'itemId': 1,
                'detectedMcIds': [101, 103],
                'shouldSplit': True,
                'drafts': [{'mcId': 103, 'mcTitle': 'Электрика', 'text': 'test'}]
            }
        ]
        ground_truth = [
            {
                'itemId': 1,
                'targetDetectedMcIds': [101, 102],
                'targetSplitMcIds': [102],
                'shouldSplit': True
            }
        ]
        
        metrics = evaluate_predictions(predictions, ground_truth)

        self.assertEqual(metrics['precision_micro'], 0.0)
        self.assertEqual(metrics['recall_micro'], 0.0)


class TestDetector(unittest.TestCase):
    """Тесты для детектора микрокатегорий."""
    
    def test_detect_plumbing(self):
        """Тест: обнаружение сантехники."""
        from src.detector import detect_microcategories
        
        text = "Выполняем сантехнические работы: разводка труб, установка унитаза"
        detected = detect_microcategories(text)
        
        self.assertIn(102, detected)
    
    def test_detect_electrician(self):
        """Тест: обнаружение электрики."""
        from src.detector import detect_microcategories
        
        text = "Электромонтажные работы, замена проводки, установка розеток"
        detected = detect_microcategories(text)
        
        self.assertIn(103, detected)
    
    def test_detect_turnkey(self):
        """Тест: обнаружение ремонта под ключ."""
        from src.detector import detect_microcategories
        
        text = "Ремонт квартир под ключ, комплексный ремонт"
        detected = detect_microcategories(text)
        
        self.assertIn(101, detected)


if __name__ == '__main__':
    unittest.main()
