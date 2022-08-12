import click
import pandas as pd
import json

from src.model import ConstModel, AIModel
from src.model_helper import ModelApplier


@click.command()
@click.argument('i', type=int)
@click.option('--data', '-d', type=click.Path())
def apply_model(i, data):
    if i == 0:
        model = ConstModel(5)
    else:
        model = AIModel()
    file = open(data)
    file_data = json.load(file)['data']
    for i, sample in enumerate(file_data):
        df = pd.DataFrame(sample)
        model_applier = ModelApplier(model, df)
        click.echo("Sample {0}, result: \n {1}".format(i, model_applier.apply()))


if __name__ == '__main__':
    apply_model()
