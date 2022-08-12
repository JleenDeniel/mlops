#!/bin/bash

pytest

( zip -o build.zip main_cli.py requirements.txt -x "**/__pycache__/*" -x "**/.pytest_cache/*" -r src tests )