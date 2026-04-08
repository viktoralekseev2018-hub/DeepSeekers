"""
Пакет для автоматического выделения самостоятельных услуг и генерации черновиков объявлений.
"""

from .microcatalog import (
    Microcategory,
    MICROCATALOG,
    MC_DICT,
    get_microcatalog,
    get_mc_by_id,
    get_all_mc_ids
)

from .detector import (
    normalize_text,
    detect_microcategories
)

from .splitter import (
    is_service_offered_separately,
    should_split_announcement
)

from .generator import (
    generate_draft_text,
    create_draft,
    create_drafts
)

from .processor import (
    process_advertisement,
    evaluate_predictions
)

__all__ = [
    # Microcatalog
    'Microcategory',
    'MICROCATALOG',
    'MC_DICT',
    'get_microcatalog',
    'get_mc_by_id',
    'get_all_mc_ids',
    
    # Detector
    'normalize_text',
    'detect_microcategories',
    
    # Splitter
    'is_service_offered_separately',
    'should_split_announcement',
    
    # Generator
    'generate_draft_text',
    'create_draft',
    'create_drafts',
    
    # Processor
    'process_advertisement',
    'evaluate_predictions',
]
