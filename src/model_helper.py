from dataclasses import dataclass
import pandas as pd
from typing import List, Sequence

from src.preprocessing import BestDataPreparator, BestData
from src.model import Model


@dataclass
class ModelApplier:
    model: Model
    data: List[pd.DataFrame]
    _preprocessor = BestDataPreparator

    def apply(self) -> Sequence:
        data = BestData(self.data)
        prep_data = BestDataPreparator().prepare(data)
        result = []
        for chunk in prep_data:
            self.model.apply(chunk)
        return result
