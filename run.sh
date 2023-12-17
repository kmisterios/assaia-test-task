#!/bin/bash

# create and activate venv
python -m venv env
source env/bin/activate
# install dependancies
pip install -r requirements.txt
# run code
python main.py --video_path $1 --polygon_path $2 --output_path $3 --annotation_path ${4:-""}