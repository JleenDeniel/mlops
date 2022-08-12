from dataclasses import dataclass
import pandas as pd
from typing import List, Sequence

from src.preprocessing import BestDataPreparator, BestData
from src.model import Model


@dataclass
class ModelApplier:
    _model: Model
    _data: pd.DataFrame
    _preprocessor = BestDataPreparator

    def apply(self) -> Sequence:
        data = BestData([self._data])
        prep_data = BestDataPreparator().prepare(data)
        result = self._model.apply(prep_data[0])
        return result
