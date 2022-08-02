import pandas as pd
from typing import List


class BestData:
    """Data-object, contains list of pandas.DataFrame 
    self.data (List[pd.DataFrame]) - read-only 
    """

    def __init__(self, data: List[pd.DataFrame]):
        self.data = data


class BestDataPreparator:
    def prepare(self, data: BestData) -> BestData:
        """Calculate features and print all chunks to console

        Args:
            data (BestData): raw data

        Returns:
            BestData: Data with calculated features
        """
        result = []
        for chunk in data:
            result.append(self._prepare_chunk(chunk))
        best_result = BestData(result)
        self._show_data(best_result)
        return best_result

    def _show_data(self, data: BestData) -> None:
        print(f"Prepared data contains {len(data)} DFs")
        for i in range(len(data)):
            print(data[i])

    def _prepare_chunk(self, data: pd.DataFrame):
        temp = data.copy()
        temp['feat1'] = temp['feat1'] * 2
        temp['feat2'] = temp['feat2'].fillna(5)
        temp['feat3'] = temp['feat1'] + temp['feat2']
        return temp
