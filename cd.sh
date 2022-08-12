#!/bin/bash

unzip -o build.zip
source venv/bin/activate
pip install requirements.txt
python main_cli.py 1 --data sample_data.json
