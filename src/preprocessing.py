import pandas as pd
from typing import List
from dataclasses import dataclass, field


@dataclass
class BestData:
    """Data-object, contains list of pandas.DataFrame 
    self.data (List[pd.DataFrame]) - read-only 
    """
    _data: List[pd.DataFrame]

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        self._current_index = 0
        return self

    def __next__(self) -> pd.DataFrame:
        if self._current_index >= len(self._data):
            raise StopIteration
        result = self._data[self._current_index]
        self._current_index += 1
        return result

    def __getitem__(self, item_index):
        return self._data[item_index]

    def __repr__(self):
        result = ''
        for i in range(len(self._data)):
            result.join([self._data[i].to_string(), '\n'])
        return result


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
