from abc import ABCMeta, abstractmethod
from typing import List, Sequence
import pandas as pd


class Model(metaclass=ABCMeta):
    @abstractmethod
    def apply(self, data: pd.DataFrame) -> Sequence:
        pass

    @abstractmethod
    def info(self):
        pass


class ConstModel(Model):
    def __init__(self, value: int) -> None:
        self.value = value
        super().__init__()

    def __repr__(self):
        return 'Model id: {}, model_info, const model, value: {}'.format(str(self.model_id), self.value)

    def apply(self, data: pd.DataFrame):
        return [self.value, ] * len(data)

    def info(self):
        return repr(self)


class AIModel(Model):
    def apply(self, data: pd.DataFrame) -> Sequence:
        self._check_data(data)
        return self._run_model(data)

    def _check_data(self, data: pd.DataFrame) -> None:
        if not all(item in data.columns for item in ['feat1', 'feat2', 'feat3']):
            raise ValueError("Can't apply AIModel: Not enough feats")

    def _run_model(self, data: pd.DataFrame) -> List:
        result = []
        for _, row in data.iterrows():
            if row['feat1'] > 10:
                if row['feat2'] > 5:
                    result.append(1)
                elif row['feat3'] < 3:
                    result.append(0)
                else:
                    result.append(2)
            elif row['feat1'] < 3:
                if row['feat2'] > 5:
                    result.append(0)
                elif row['feat3'] < 3:
                    result.append(2)
                else:
                    result.append(1)
            else:
                result.append(0)
        return result

    def info(self):
        return "SOTA AI model"
