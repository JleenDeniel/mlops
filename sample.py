import pandas as pd
from src.model import ConstModel, AIModel
from src.preprocessing import BestData, BestDataPreparator


def sample1():
    sample_data = [
        pd.DataFrame([[1, 2], [3, 4], [0, 3]], columns=['feat1', 'feat2']),
        pd.DataFrame([[0, 0], [1, 1], [2, 2]], columns=['feat1', 'feat2'])
    ]
    data = BestData(sample_data)
    # fix BestData to run this:
    data = BestDataPreparator().prepare(data)

    # hotfix
    sample_data[0]['feat3'] = 0
    sample_data[1]['feat3'] = 0
    # hotfix end
    for chunk in data:
        print(AIModel().apply(chunk))
        print(ConstModel(1).apply(chunk))


def sample2(data: pd.DataFrame):
    """
    DS should write somthing like this - and it should work
    """
    model = ConstModel(1)
    return model.apply(data)


if __name__ == "__main__":
    sample1()
