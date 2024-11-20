#!/bin/bash
curl localhost:8000 &

source hub_venv/bin/activate
python run_a2a-v2.py
