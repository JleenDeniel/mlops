import pandas as pd
from collections.abc import Sequence
import pytest

from src.model import ConstModel, AIModel


def test_const_model_answers():
    """
    Тест на правильные ответы модели
    Не уверен в том, что так делать правильно, потому что в теории ответы модели могут поменяться,
    но в теории этот тест можно потом изменить на проверку не конкретных ответов, а проверять "правильность"
    распределения
    """
    const = 0
    test_object = ConstModel(const)
    test_pd_df = pd.DataFrame([[0, 1, 2], [0, 0, 0]])
    assert isinstance(test_object.apply(test_pd_df), Sequence)
    assert test_object.apply(test_pd_df) == [0, 0]


def test_returning_obj_ai_model():
    """
    Тест на что модель эплаится и вовзращает обьект нужного типа
    """
    test_pd_df1 = pd.DataFrame([], columns=['feat1', 'feat2', 'feat3'])
    model = AIModel()
    assert isinstance(model.apply(test_pd_df1), Sequence)


def test_apply_ai_model():
    """
    Тест на то что вылетает ValueError
    """
    model = AIModel()
    with pytest.raises(ValueError):
        model.apply(pd.DataFrame([]))

