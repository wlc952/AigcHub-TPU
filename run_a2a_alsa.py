import re
import requests
import os
import logging
import subprocess
import time
import pexpect
import signal
import sys
import threading  # 新增

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

session = requests.Session()

llm_running = False  # 新增全局变量

def extract_text(s):
    print(s)
    match = re.findall(r'\d{1,2}:\s*(.+)', s)
    if match:
        cleaned_text = re.sub(r'\x1B\[[0-?]*[ -/]*[@-~]|\d+:\s*', '', match[-1])
        return cleaned_text
    else:
        return ''


def send_to_llm(text):
    global llm_running
    def task():
        global llm_running
        llm_running = True
        url = "http://localhost:8000/llm_tpu/v1/chat/completions"
        headers = {'Content-Type': 'application/json'}
        data = {
            "model": "qwen2.5-3b_int4_seq512_1dev.bmodel",
            "messages": [
                {"role": "system", "content": "你是可以精简回答问题的助理。我的提问中可能存在语句错误，你会自动纠正。"},
                {"role": "user", "content": text}
            ],
            "stream": False
        }
        response = session.post(url, headers=headers, json=data)
        print("llm:",response.content.decode())
        text_to_audio(response.content.decode())
        llm_running = False  # 线程结束时重置标志
    thread = threading.Thread(target=task)
    thread.start()

def text_to_audio(text):
    global llm_running
    llm_running = True  # 播放音频前设置标志
    url = "http://localhost:8000/emotivoice/v1/audio/speech"
    headers = {'Content-Type': 'application/json'}
    data = {"input": text}
    resp = session.post(url, headers=headers, json=data)
    subprocess.run(['aplay', '/data/tmpdir/speech.wav'])
    time.sleep(3)  # 添加短暂延迟，确保音频播放完成
    llm_running = False  # 播放音频后重置标志


def main():
    command = "repo/sherpa/build/bin/sherpa-onnx-alsa --tokens=repo/sherpa/sherpa_models/tokens.txt --zipformer2-ctc-model=repo/sherpa/sherpa_models/zipformer2_ctc_F32.bmodel plughw:1,0"

    child = pexpect.spawn(command, timeout=None, encoding='utf-8')

    def signal_handler(sig, frame):
        logging.info("接收到中断信号，正在退出...")
        child.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    try:
        while True:
            line = child.readline().strip() # 读取一行，但如果此时llm在运行，则舍弃，不进一步处理
            if not line or llm_running:  # 如果读取为空或者llm正在运行，则跳过
                continue
            match = extract_text(line)
            if len(match)>=2: # 如果字符小于2 则不发送
                send_to_llm(match)  # 使用线程调用
            

    except pexpect.EOF:
        logging.info("子进程已终止。")
    except Exception as e:
        logging.error(f"运行时出错: {e}")

if __name__ == "__main__":
    main()

