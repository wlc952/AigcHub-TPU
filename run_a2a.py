import os
import subprocess
import gradio as gr
import requests
import time
from pydub import AudioSegment
from pydub.silence import split_on_silence
from concurrent.futures import ThreadPoolExecutor
import random


def preprocess_audio(file_path):
    audio = AudioSegment.from_file(file_path)
    chunks = split_on_silence(audio, min_silence_len=500, silence_thresh=-40)
    processed_audio = AudioSegment.empty()

    for chunk in chunks:
        processed_audio += chunk

    processed_audio.export("/data/tmpdir/processed_audio.wav", format="wav")
    return "/data/tmpdir/processed_audio.wav"

# 初始化历史记录
history = [{"role": "system", "content": "You are a helpful assistant."}]
conversation_count = 0

def a2t(file_path):
    url = "http://localhost:8000/sherpa/v1/audio/transcriptions"
    data = {'response_format': 'json', 'file_tmp_path': file_path}
    response = requests.post(url, data=data)
    return response.json()['text']

def llm(messages):
    url = "http://localhost:8000/llm_tpu/v1/chat/completions"
    headers = {'Content-Type': 'application/json'}
    data = {
        "model": "qwen2.5-3b_int4_seq512_1dev.bmodel",
        "messages": messages,
        "stream": False
    }
    # response = requests.post(url, headers=headers, json=data, stream=True)
    # for line in response.iter_lines():
    #     if line:
    #         decoded_line = line.decode('utf-8')
    #         yield decoded_line
    #         t2a(decoded_line)
    st = time.time()
    response = requests.post(url, headers=headers, json=data)
    print(f"LLM Response: {response.text}， Time: {time.time() - st}")
    mes = response.json()
    t2a(mes)
    return mes

def t2a(txt):
    url = "http://localhost:8000/emotivoice/v1/audio/speech"
    headers = {'Content-Type': 'application/json'}
    data = {"input": txt}
    response = requests.post(url, headers=headers, json=data)
    print(f"Audio Response: {response.text}")
    os.system(f'aplay {response.text}')
    return

def process_audio(file_path):
    st = time.time()
    global history, conversation_count
    # Step 0: Preprocess Audio
    file_path = preprocess_audio(file_path)
    # Step 1: Audio to Text
    text = a2t(file_path)
    print(f"Transcribed Text: {text}， Time: {time.time() - st}")
    wait_wav = ["/data/tmpdir/waiting1.wav", "/data/tmpdir/waiting2.wav", "/data/tmpdir/waiting3.wav"]

    selected_wav = random.choice(wait_wav)

    subprocess.Popen(['aplay', selected_wav])

    # Step 2: Update history with user input
    history.append({"role": "user", "content": text})

    # Step 3: Text to LLM Response (Streamed)
    response_text = ""
    for partial_response in llm(history):
        response_text += partial_response

    # Step 4: Update history with assistant response
    history.append({"role": "assistant", "content": response_text})

    # Step 5: Increment conversation count and reset history if needed
    conversation_count += 1
    if conversation_count >= 5:
        history = [{"role": "system", "content": "You are a helpful assistant."}]
        conversation_count = 0

    return response_text

iface = gr.Interface(fn=process_audio, inputs=gr.Audio(type="filepath"), outputs="text")

iface.launch(server_name="0.0.0.0", server_port=5000, inbrowser=True)