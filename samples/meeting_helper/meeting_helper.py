import gradio as gr
import os
import hashlib
import shutil
from pathlib import Path
import re
import requests
from pydub import AudioSegment
from pydub.silence import split_on_silence
import json

session = requests.Session()
if not os.path.exists("audios"):
    os.makedirs("audios")

def llm_chat(ip, input_str, chatbot, prompt, show_input=True):
    messages = [{"role": "system", "content": prompt}, {"role": "user", "content": input_str}]
    response_text = ''
    url = f"http://{ip}/v1/chat/completions"
    headers = {'Content-Type': 'application/json'}

    json_data = {
        "messages": messages,
        "stream": True,
    }
    if show_input:
        chatbot.append({"role": "user", "content": input_str})
    chatbot.append({"role": "assistant", "content": ""})
    try:
        response = session.post(url, headers=headers, json=json_data, stream=True)
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8').strip()
                if decoded_line.startswith('data:'):
                    decoded_line = decoded_line[len('data:'):].strip()
                if decoded_line == '':
                    continue  # 跳过心跳包或空行
                data_json = json.loads(decoded_line)
                delta = data_json['choices'][0]['delta']
                content = delta.get('content', '')
                if content == "<think>":
                    content = 'Thinking...\n'
                if content == "</think>":
                    content = '\nEnd of thinking...\n'

                response_text += content
                chatbot[-1] = {"role": "assistant", "content": response_text}  # ✅ 更新而非追加
                
                yield chatbot.copy()  # ✅ 返回列表的副本避免引用问题
            
    except Exception as e:
        chatbot[-1] = {"role": "assistant", "content": f"Error: {str(e)}"}
        yield chatbot

def reset():
    return []

def clear():
    return ""

def sherpa(ip, filepath, if_process=True):
    url = f"http://{ip}/sherpa/v1/audio/transcriptions"
    data = {'response_format': 'text'}
    if if_process: # 是否预处理音频
        filepath = preprocess_audio(filepath)
    files = {'file': open(filepath, 'rb')}
    response = session.post(url, files=files, data=data)
    if response.status_code == 200:
        return response.text
    else:
        return f"请求失败，状态码：{response.status_code}"
    
def preprocess_audio(file_path):
    audio = AudioSegment.from_file(file_path)
    chunks = split_on_silence(audio, min_silence_len=500, silence_thresh=-40)
    processed_audio = AudioSegment.empty()

    for chunk in chunks:
        processed_audio += chunk

    processed_audio.export("./audios/processed_audio.wav", format="wav")
    return "./audios/processed_audio.wav"

def text_processing(ip, content, chatbot, prompt, show_input=False):
    """文档内容清洗和标准化处理"""
    # 基础清洗流程
    processed = re.sub(r'\s+', ' ', content)  # 合并连续空白
    processed = re.sub(r'[^\w\s\u4e00-\u9fff]', '', processed)  # 去除非文字符号
    processed = processed.strip()  # 去除首尾空格
    # segments = [processed[i:i+200] for i in range(0, len(processed), 200)]
    # input_str = "\n".join([f"段落 {i+1}: {seg}" for i, seg in enumerate(segments)])
    input_str = processed
    # 调用llm_caht
    yield from llm_chat(ip, input_str, chatbot, prompt, show_input)

def validate_directory(directory):
    """验证目录是否有效且可读写"""
    directory = directory.strip()
    if not directory:
        return False, "目录路径不能为空"
    if not os.path.isdir(directory):
        return False, "路径不存在或不是目录"
    if not os.access(directory, os.R_OK | os.W_OK):
        return False, "目录不可读写"
    return True, ""

def get_folder_structure(directory):
    """获取目录结构"""
    return "\n".join([f"{root.replace(directory, '')}/{file}" 
                    for root, dirs, files in os.walk(directory) 
                    for file in files])

def delete_duplicate_files(directory):
    """移动重复文件到回收站"""
    valid, msg = validate_directory(directory)
    if not valid:
        return msg
    
    trash_path = os.path.join(directory, "回收站")
    os.makedirs(trash_path, exist_ok=True)
    
    hash_dict = {}
    duplicates = []
    for root, _, files in os.walk(directory):
        if root == trash_path:  # 跳过回收站
            continue
        for file in files:
            file_path = os.path.join(root, file)
            try:
                with open(file_path, "rb") as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()
                if file_hash in hash_dict:
                    duplicates.append(file_path)
                else:
                    hash_dict[file_hash] = file_path
            except Exception as e:
                return f"处理文件失败: {str(e)}"
    
    for dup in duplicates:
        try:
            dest = os.path.join(trash_path, os.path.basename(dup))
            if os.path.exists(dest):
                base, ext = os.path.splitext(os.path.basename(dup))
                counter = 1
                while os.path.exists(dest):
                    dest = os.path.join(trash_path, f"{base}_({counter}){ext}")
                    counter += 1
            shutil.move(dup, dest)
        except Exception as e:
            return f"移动文件失败: {str(e)}"
    
    return f"操作完成，移动{len(duplicates)}个文件到回收站\n\n当前目录结构:\n{get_folder_structure(directory)}"

def empty_trash(directory):
    """清空回收站"""
    valid, msg = validate_directory(directory)
    if not valid:
        return msg
    
    trash_path = os.path.join(directory, "回收站")
    if not os.path.exists(trash_path):
        return "回收站不存在"
    
    try:
        shutil.rmtree(trash_path)
        return f"回收站已清空\n\n当前目录结构:\n{get_folder_structure(directory)}"
    except Exception as e:
        return f"清空失败: {str(e)}"

def categorize_files(directory):
    """文件分类"""
    valid, msg = validate_directory(directory)
    if not valid:
        return msg
    
    try:
        for root, _, files in os.walk(directory):
            for file in files:
                src_path = os.path.join(root, file)
                if os.path.isfile(src_path):
                    ext = Path(file).suffix[1:] or "无后缀"
                    dest_dir = os.path.join(directory, ext)
                    os.makedirs(dest_dir, exist_ok=True)
                    shutil.move(src_path, os.path.join(dest_dir, file))
        return f"文件分类完成\n\n当前目录结构:\n{get_folder_structure(directory)}"
    except Exception as e:
        return f"分类失败: {str(e)}"

with gr.Blocks() as demo:
    gr.Markdown("🔖 会议助手")
    with gr.Tab("文档总结"):
        gr.Markdown("### 📄 文档智能提炼")
        with gr.Row():
            with gr.Column(scale=2):
                ip_0 = gr.Textbox(lines=1, value="localhost:8000/qwen", placeholder="请输入api url", show_label=False)
                input_0 = gr.Textbox(label="输入文档内容", lines=21, placeholder="请直接粘贴文档内容或输入文本...")
                with gr.Row():
                    submitBtn_0 = gr.Button("Submit", variant="primary")
                    emptyBtn_0 = gr.Button(value="Clear")
            with gr.Column(scale=3):
                prompt_0 = gr.Textbox(lines=1, value="语言简练地总结文字内容", label="PROMPT", visible=False)
                chatbot_0 = gr.Chatbot(label="文档总结", height=600, type='messages')
        input_0.submit(text_processing, inputs=[ip_0, input_0, chatbot_0, prompt_0], outputs=chatbot_0)
        submitBtn_0.click(reset, outputs=chatbot_0).then(text_processing, inputs=[ip_0, input_0, chatbot_0, prompt_0], outputs=chatbot_0)
        emptyBtn_0.click(reset, outputs=chatbot_0).then(clear, outputs=input_0)

    with gr.Tab("语音转录"):
        gr.Markdown("### 📝 会议纪要生成")
        with gr.Row():
            with gr.Column(scale=1):
                ip_1 = gr.Textbox(lines=1, value="localhost:8000", placeholder="请输入api url...", show_label=False, visible=False)
                ip_2 = gr.Textbox(lines=1, value="localhost:8000/qwen", placeholder="请输入api url...", show_label=False)
                prompt_1 = gr.Textbox(lines=1, value="语言简练地总结文字内容", label="PROMPT")
                audio_file = gr.Audio(type="filepath", label="上传音频文件", show_label=False)
                with gr.Row():
                    if_process = gr.Checkbox(label="是否预处理音频", value=True)
                    emptyBtn_1 = gr.Button(value="清空")
                    submitBtn_1 = gr.Button("提交")
            with gr.Column(scale=2):
                output_1 = gr.Textbox(label="语音转文本", visible=True)
                chatbot_1 = gr.Chatbot(label="会议纪要", height=500, type='messages', show_label=False)
        submitBtn_1.click(reset, outputs=chatbot_1).then(fn=sherpa, inputs=[ip_1, audio_file, if_process], outputs=output_1).then(text_processing, inputs=[ip_2, output_1, chatbot_1, prompt_1], outputs=chatbot_1)
        emptyBtn_1.click(reset, outputs=chatbot_1)

    with gr.Tab("Chat bot"):
        gr.Markdown("### 🤖 聊天机器人")
        with gr.Row():
            with gr.Column():
                ip = gr.Textbox(lines=1, value="localhost:8000/qwen", placeholder="请输入api url...", show_label=False)
                prompt = gr.Textbox(lines=1, value="You are a helpful assistant.", label="PROMPT")
                chatbot = gr.Chatbot(label="Chat with LLM", height=400, type='messages')
                with gr.Row():
                    with gr.Column():
                        input_str = gr.Textbox(show_label=False, placeholder="Chat with LLM")
                        with gr.Row():
                            submitBtn = gr.Button("Submit", variant="primary")
                            emptyBtn = gr.Button(value="Clear")
        input_str.submit(llm_chat, inputs=[ip, input_str, chatbot, prompt], outputs=chatbot).then(clear, outputs=input_str)
        submitBtn.click(llm_chat, inputs=[ip, input_str, chatbot, prompt], outputs=chatbot).then(clear, outputs=input_str)
        emptyBtn.click(reset, outputs=chatbot)
        
    with gr.Tab("重复文件管理"):
        gr.Markdown("### 🗑️ 重复文件处理")
        dir_input = gr.Textbox(label="输入目录路径", placeholder="请输入绝对路径...")
        with gr.Row():
            del_btn = gr.Button("删除重复文件")
            empty_btn = gr.Button("清空回收站")
        output = gr.Textbox(label="操作结果", lines=15)
        
        del_btn.click(delete_duplicate_files, inputs=dir_input, outputs=output)
        empty_btn.click(empty_trash, inputs=dir_input, outputs=output)
    
    with gr.Tab("文档分类"):
        gr.Markdown("### 🗂️ 文件自动分类")
        dir_input2 = gr.Textbox(label="输入目录路径", placeholder="请输入绝对路径...")
        cat_btn = gr.Button("执行分类")
        output2 = gr.Textbox(label="操作结果", lines=15)
        
        cat_btn.click(categorize_files, inputs=dir_input2, outputs=output2)

demo.launch()
