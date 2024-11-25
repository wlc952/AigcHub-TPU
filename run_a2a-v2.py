import codecs
import os
import subprocess
import gradio as gr
import requests
import time
from pydub import AudioSegment
from pydub.silence import split_on_silence
import random
import queue
import threading
import re
import socket

session = requests.Session()

def get_host_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # doesn't even have to be reachable
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

host_ip = get_host_ip()

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

# 语音播放队列
audio_queue = queue.Queue()

def audio_player():
    while True:
        audio_file = audio_queue.get()
        if (audio_file is None):
            break
        subprocess.run(['aplay', audio_file])
        audio_queue.task_done()

# 启动语音播放线程
threading.Thread(target=audio_player, daemon=True).start()

def a2t(file_path):
    url = "http://localhost:8000/sherpa/v1/audio/transcriptions"
    data = {'response_format': 'json', 'file_tmp_path': file_path}
    response = session.post(url, data=data)
    return response.json()['text']

def llm(messages):
    url = "http://localhost:8000/llm_tpu/v1/chat/completions"
    headers = {'Content-Type': 'application/json'}
    data = {
        "model": "qwen2.5-3b_int4_seq512_1dev.bmodel",
        "messages": messages,
        "stream": True
    }
    st = time.time()
    buffer = ""
    response = session.post(url, headers=headers, json=data, stream=True)
    # 检查响应状态
    desired_chars = 10  # 例如10个字符
    chunk_size = 30  # 例如10个字符 * 3字节 = 30字节

    for chunk in response.iter_content(chunk_size=chunk_size):
        if chunk:
            decoded_chunk = decoder.decode(chunk)
            buffer += decoded_chunk
            match = re.search(r'([.,!?，。！？])', buffer)
            if match:
                end_index = match.end()
                to_send = buffer[:end_index]
                threading.Thread(target=t2a, args=(to_send,)).start()
                yield to_send
                print(f"LLM Response: {to_send}, Time: {time.time() - st:.2f}s")
                buffer = buffer[end_index:]
    
    # 处理剩余的缓冲区内容
    remaining = decoder.decode(b'', final=True)
    if remaining:
        t2a(remaining)
        print(f"LLM Response: {remaining}, Time: {time.time() - st:.2f}s")
    # 处理剩余的缓冲区内容
    if buffer:
        t2a(buffer)
        print(f"LLM Response: {buffer}, Time: {time.time() - st:.2f}s")

def t2a(txt):
    url = "http://localhost:8000/emotivoice/v1/audio/speech"
    headers = {'Content-Type': 'application/json'}
    data = {"input": txt}
    response = session.post(url, headers=headers, json=data)
    audio_queue.put(response.text)
    return

def process_audio(file_path):
    st = time.time()
    global history, conversation_count

    wait_wav = ["/data/tmpdir/waiting1.wav", "/data/tmpdir/waiting2.wav", "/data/tmpdir/waiting3.wav"]
    selected_wav = random.choice(wait_wav)
    subprocess.Popen(['aplay', selected_wav])

    # Step 0: Preprocess Audio
    file_path = preprocess_audio(file_path)
    # Step 1: Audio to Text
    text = a2t(file_path)
    print(f"Transcribed Text: {text}， Time: {time.time() - st}")
    

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
print(f"\nRunning on: http://{host_ip}:5000")
iface.launch(server_name="0.0.0.0", server_port=5000, inbrowser=True)
