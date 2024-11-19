from fastapi.responses import JSONResponse, StreamingResponse
from api.base_api import BaseAPIRouter, change_dir, init_helper
import argparse
import os
from pydantic import BaseModel, Field

app_name = "llm_tpu"

class AppInitializationRouter(BaseAPIRouter):
    dir = f"repo/{app_name}"
    @init_helper(dir)
    async def init_app(self):
        args = argparse.Namespace(
            model_path = 'bmodels/qwen2.5-3b_int4_seq512_1dev.bmodel',
            tokenizer_path = 'llm_models/Qwen2_5/support/token_config',
            devid='0',
            temperature=1.0,
            top_p=1.0,
            repeat_penalty=1.0,
            repeat_last_n=32,
            max_new_tokens=1024,
            generation_mode="greedy",
            prompt_mode="prompted",
            enable_history=False,
            lib_path=''
        )

        from repo.llm_tpu.llm_models.Qwen2_5.python_demo.pipeline import Qwen2_5
        self.llm_model = Qwen2_5(args)
        return {"message": f"应用 {self.app_name} 已成功初始化。"}
    
    async def destroy_app(self):
        del self.llm_model

router = AppInitializationRouter(app_name=app_name)

class ChatRequest(BaseModel):
    model: str = Field("qwen2.5-3b_int4_seq512_1dev.bmodel", description="bmodel file name")
    messages: list = Field([{"role":"system","content":"You are a helpful assistant."},{"role":"user","content":"hello"}], description="Chat history")
    stream: bool = Field(False, description="Stream response")

@router.post("/v1/chat/completions")
@change_dir(router.dir)
async def chat_completions(request: ChatRequest):

    slm = router.llm_model
    
    tokens = slm.tokenizer.apply_chat_template(request.messages, tokenize=True, add_generation_prompt=True)

    token = slm.model.forward_first(tokens)
    output_tokens = [token]

    if not isinstance(slm.EOS, list):
        slm.EOS = [slm.EOS]

    if request.stream:
        def generate_responses():
            # yield '{"delta": {"role": "assistant", "content": "'
            while True:
                token = slm.model.forward_next()
                if token in slm.EOS or slm.model.token_length >= slm.model.SEQLEN:
                    break
                output_tokens.append(token)
                response_text = slm.tokenizer.decode([token])
                yield response_text
            # yield '"}}'
        return StreamingResponse(generate_responses(), media_type="text/plain")
    
    else:
        while True:
            token = slm.model.forward_next()
            if token in slm.EOS or slm.model.token_length >= slm.model.SEQLEN:
                break
            output_tokens += [token]
        slm.answer_cur = slm.tokenizer.decode(output_tokens)
        # return JSONResponse({"delta": {"role": "assistant", "content": slm.answer_cur}})
        return slm.answer_cur
    

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
