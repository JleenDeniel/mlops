import pandas as pd
from src.model import ConstModel, AIModel
from src.preprocessing import BestData, BestDataPreparator

from src.model_helper import ModelApplier


def sample1():
    sample_data = [
        pd.DataFrame([[1, 2], [3, 4], [0, 3]], columns=['feat1', 'feat2']),
        pd.DataFrame([[0, 0], [1, 1], [2, 2]], columns=['feat1', 'feat2'])
    ]
    data = BestData(sample_data)
    data_prepared = BestDataPreparator().prepare(data)
    for chunk in data_prepared:
        print(AIModel().apply(chunk))
        print(ConstModel(1).apply(chunk))


def sample2(data: pd.DataFrame):
    """
    DS should write somthing like this - and it should work
    """
    ai_model = AIModel()
    ai_model_applier = ModelApplier(ai_model, data)
    return ai_model_applier.apply()


if __name__ == "__main__":
    data = pd.DataFrame([[1, None], [3, 4], [0, 3]], columns=['feat1', 'feat2'])
    print(sample2(data))

