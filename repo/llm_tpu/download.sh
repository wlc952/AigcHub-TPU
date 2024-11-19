#!/bin/bash

mkdir -p bmodels

cd bmodels

# 下载模型
# python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/phi3-4b_int4_1dev.bmodel

python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen2.5-3b_int4_seq512_1dev.bmodel

# python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen1.5-1.8b_int8_1dev_seq1280.bmodel

cd ..

echo "所有模型下载完成。"
