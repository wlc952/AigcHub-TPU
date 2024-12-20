from fastapi import Form
from fastapi.responses import JSONResponse, StreamingResponse
from api.base_api import BaseAPIRouter, change_dir, init_helper
from typing import Optional
import argparse
import os
import re
import json
from pydantic import BaseModel, Field
from difflib import get_close_matches
import base64
from io import BytesIO
from PIL import Image

app_name = "llm_tpu"

def match_model(model_name, patterns):
    model_name = re.sub(r'\W', '', model_name.lower().replace('_', ''))
    normalized_patterns = [re.sub(r'\W', '', pattern.lower().replace('_', '')) for pattern in patterns]
    for x in range(len(normalized_patterns)):
        if normalized_patterns[x] in model_name:
            return x
    return None

class AppInitializationRouter(BaseAPIRouter):
    dir = f"repo/{app_name}"
    @init_helper(dir)
    async def init_app(self):
        import importlib

        self.models = {}
        self.models_list = os.listdir("llm_bmodels")
        tokenizer_dict = {}

        for root, dirs, files in os.walk('./llm_models'):
            if 'token_config' in dirs:
                full_path = os.path.join(root, 'token_config')
                # 分割路径
                parts = root.split(os.sep)
                # 假设模型目录总是位于 './models' 目录下的第一级子目录
                if len(parts) > 2 and parts[1] == 'models':
                    model_name = parts[2]
                else:
                    model_name = os.path.basename(root)
                tokenizer_dict[model_name] = full_path

        args = argparse.Namespace(
            devid='0',
            temperature=1.0,
            top_p=1.0,
            repeat_penalty=1.0,
            repeat_last_n=32,
            max_new_tokens=512,
            generation_mode="greedy",
            prompt_mode="prompted",
            enable_history=False,
            lib_path=''
        )

        mm = list(tokenizer_dict.keys())
        nn = list(tokenizer_dict.values())

        for model_name in self.models_list:
            id = match_model(model_name, mm)
            if id is None:
                print(f"Model {model_name} does not match any available model.")
                continue
            tokenizer_path = nn[id]
            args.model_path = f"llm_bmodels/{model_name}"
            args.tokenizer_path = tokenizer_path

            module_name = f"llm_models.{mm[id]}.python_demo.pipeline"
            module = importlib.import_module(module_name)

            model_class = getattr(module, mm[id])

            self.models[f"{model_name}"] = model_class(args)

        return {"message": f"应用 {self.app_name} 已成功初始化。"}
    
    async def destroy_app(self):
        del self.models, self.models_list

router = AppInitializationRouter(app_name=app_name)

class ChatRequest(BaseModel):
    model: str = Field("minicpm3-4b_int4_seq512_1dev.bmodel", description="bmodel file name")
    messages: list = Field([{"role":"system","content":"You are a helpful assistant."},{"role":"user","content":"hello"}], description="Chat history")
    stream: bool = Field(False, description="Stream response")

@router.post("/v1/chat/completions")
@change_dir(router.dir)
async def chat_completions(request: ChatRequest):
    best_match = get_close_matches(request.model, router.models_list, n=1, cutoff=0.0)
    if best_match:
        m_name = best_match[0]
        seq_len = int(m_name.split('seq')[1].split('_')[0].split('.')[0]) if 'seq' in m_name else 512
        slm = router.models[best_match[0]]
    else:
        slm = router.models['minicpm3-4b_int4_seq512_1dev.bmodel']
    
    if isinstance(request.messages[-1]['content'], list):
        content = request.messages[-1]['content']
        for x in content:
            if x['type'] == 'text':
                slm.input_str = x['text']
            elif x['type'] == 'image_url':
                image_data = x['image_url']['url']
                if image_data.startswith("data:"):
                    base64_data = image_data.split(",")[1]  # 去掉前缀
                    image_bytes = base64.b64decode(base64_data)  # 解码
                    image = Image.open(BytesIO(image_bytes))  # 读取图片
                    image.save("/data/tmpdir/image.png", format='PNG')
                    slm.image_str = "/data/tmpdir/image.png"
                else:
                    png = image_data.split('/')[-1]
                    os.system(f"wget {image_data} -O /data/tmpdir/{png}")
                    slm.image_str = f"/data/tmpdir/{png}"
            elif x['type'] == 'image_path':
                slm.image_str = x['image_path']['path'] if isinstance(x['image_path'], dict) else x['image_path']
            else:
                slm.image_str = ''
    else:
        if len(request.messages[-1]['content']) > seq_len:
            request.messages[-1]['content'] = request.messages[-1]['content'][:seq_len]
        slm.input_str = request.messages[-1]['content']

    if "minicpmv" in request.model.lower():
        try:
            image_path = slm.image_str
        except:
            slm.image_str = ''

        if slm.image_str:
            if not os.path.exists(slm.image_str):
                print("Can't find image: {}".format(slm.image_str))
        
        if request.messages[0]['role'] == 'system':
            prompt = request.messages[0]['content']
        else:
            prompt = "You are a helpful assistant."
        slm.system_prompt = f'<|im_start|>system\n{prompt}\n<|im_end|>\n<|im_start|>user\n'
    
        slm.encode()
        token = slm.model.forward_first(slm.input_ids, slm.pixel_values, slm.image_offset)
        EOS = [slm.ID_EOS, slm.ID_IM_END]
    else:
        slm.clear()
        tokens = slm.tokenizer.apply_chat_template(request.messages, tokenize=True, add_generation_prompt=True)
        token =  slm.model.forward_first(tokens)
        EOS = slm.EOS if isinstance(slm.EOS, list) else [slm.EOS]

    if request.stream:
        def generate_responses(token):
            output_tokens = []
            while True:
                output_tokens.append(token)
                if token in EOS or slm.model.token_length >= slm.model.SEQLEN:
                    break
                word = slm.tokenizer.decode(output_tokens, skip_special_tokens=True)
                if "�" not in word:
                    if len(output_tokens) == 1:
                        pre_word = word
                        word = slm.tokenizer.decode([token, token], skip_special_tokens=True)[len(pre_word):]
                    data = {"choices": [{"delta": {"role": "assistant", "content": word}}]}
                    yield f"data:{json.dumps(data)}\n\n"
                    output_tokens = []
                token = slm.model.forward_next()
        return StreamingResponse(generate_responses(token), media_type="text/event-stream")
    else:
        output_tokens = [token]
        while True:
            token = slm.model.forward_next()
            if token in EOS or slm.model.token_length >= slm.model.SEQLEN:
                break
            output_tokens += [token]
        slm.answer_cur = slm.tokenizer.decode(output_tokens)
        return JSONResponse({"choices": [{"message": {"role": "assistant", "content": slm.answer_cur}}]})
    
### 常规测试
# curl --no-buffer -X 'POST' \
#   'http://localhost:8000/llm_tpu/v1/chat/completions' \
#   -H 'Content-Type: application/json' \
#   -d '{
#   "model": "qwen1.5-1.8b_int8_1dev_seq1280.bmodel",
#   "messages": [
#     {
#       "role": "system",
#       "content": "You are a helpful assistant."
#     },
#     {
#       "role": "user",
#       "content": "hello"
#     }
#   ],
#   "stream": true
# }'


### 图片测试
# curl -X 'POST' \
#   'http://localhost:8000/llm_tpu/v1/chat/completions' \
#   -H 'accept: application/json' \
#   -H 'Content-Type: application/json' \
#   -d '{
#   "model": "minicpmv2",
#   "messages": [
#     {
#       "role": "system",
#       "content": "You are a helpful assistant."
#     },
#     {
#       "role": "user",
#       "content": [{"type":"text","text":"what is it?"},{"type":"image_url","image_url":{"url":"https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"}}]
#     }
#   ],
#   "stream": true
# }'
