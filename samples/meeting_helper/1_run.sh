#!/bin/bash
curl localhost:8000 &

source ../../hub_venv/bin/activate
python meeting_helper.py