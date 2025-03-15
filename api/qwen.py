from fastapi.responses import JSONResponse, StreamingResponse
from api.base_api import BaseAPIRouter, change_dir, init_helper
import argparse
import os
from pydantic import BaseModel, Field
import json

app_name = "qwen"

abs_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

class AppInitializationRouter(BaseAPIRouter):
    dir = f"{abs_dir}/repo/{app_name}"
    @init_helper(dir)
    async def init_app(self):
        args = argparse.Namespace(
            model_path     = f'{abs_dir}/repo/{app_name}/qwen2.5-3b_int4_seq4096_1dev_opt.bmodel',
            tokenizer_path = f'{abs_dir}/repo/{app_name}/support/token_config',
            devid          = '0',
            temperature    = 1.0,
            top_p          = 1.0,
            repeat_penalty = 1.0,
            repeat_last_n  = 32,
            generation_mode= "greedy",
            enable_history = False,
        )

        from repo.qwen.python_demo_opt.pipeline import Qwen2
        self.llm_model = Qwen2(args)
        return {"message": f"应用 {self.app_name} 已成功初始化。"}
    
    async def destroy_app(self):
        del self.llm_model

router = AppInitializationRouter(app_name=app_name)

class ChatRequest(BaseModel):
    model: str = Field("qwen2.5-3b_int4_seq4096_1dev_opt.bmodel", description="model name")
    messages: list = Field([{"role":"user","content":"hello"}], description="Chat history")
    stream: bool = Field(False, description="Stream response")

@router.post("/v1/chat/completions")
@change_dir(router.dir)
async def chat_completions(request: ChatRequest):
    slm = router.llm_model

    # slm.history = [{"role": "system", "content": "You are a helpful assistant."}]
    slm.history = []
    slm.history += request.messages

    text = slm.tokenizer.apply_chat_template(slm.history, tokenize=False, add_generation_prompt=True)
    ids = slm.tokenizer(text).input_ids
    
    if not isinstance(slm.EOS, list):
        EOS = [slm.EOS]
    else:
        EOS = slm.EOS

    if request.stream:
        def generate_responses():
            token = slm.model.forward_first(ids)
            output_tokens = []
            while token not in EOS and slm.model.token_length < slm.model.SEQLEN:
                output_tokens.append(token)
                word = slm.tokenizer.decode(output_tokens, skip_special_tokens=True)
                if "�" in word:
                    token = slm.model.forward_next()
                    continue
                data = {"choices": [{"delta": {"role": "assistant", "content": word}}]}
                yield f"data:{json.dumps(data, ensure_ascii=False)}\n\n"
                output_tokens = []
                token = slm.model.forward_next()
        return StreamingResponse(generate_responses(), media_type="text/event-stream")
    else:
        token = slm.model.forward_first(ids)
        output_tokens = [token]
        while True:
            token = slm.model.forward_next()
            if token in EOS or slm.model.token_length >= slm.model.SEQLEN:
                break
            output_tokens += [token]
        slm.answer_cur = slm.tokenizer.decode(output_tokens)
        return JSONResponse({"choices": [{"message": {"role": "assistant", "content": slm.answer_cur}}]})
