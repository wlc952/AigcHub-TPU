from fastapi.responses import JSONResponse, StreamingResponse
from api.base_api import BaseAPIRouter, change_dir, init_helper
import argparse
import os
from pydantic import BaseModel, Field
import json

app_name = "deepseek"

abs_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

class AppInitializationRouter(BaseAPIRouter):
    dir = f"{abs_dir}/repo/{app_name}"
    @init_helper(dir)
    async def init_app(self):
        args = argparse.Namespace(
            devid='0',
            dir_path = f'{abs_dir}/repo/{app_name}/deepseek-r1-distill-qwen-1.5b-2048',
            generation_mode='greedy',
            test_input=None,
            test_media=None,
            model_type=None,
            enable_history=False,
            max_new_tokens=2048,
            model_path='',
            repeat_last_n=32,
            repeat_penalty=1.2,
            temperature=1.0,
            top_p=1.0,
        )

        from repo.deepseek.deepseek_r1_distill_qwen.pipeline import Model
        self.llm_model = Model(args)
        return {"message": f"应用 {self.app_name} 已成功初始化。"}
    
    async def destroy_app(self):
        del self.llm_model

router = AppInitializationRouter(app_name=app_name)

class ChatRequest(BaseModel):
    model: str = Field("deepseek-r1-distill-qwen-1.5b-2048", description="model name")
    messages: list = Field([{"role":"user","content":"hello"}], description="Chat history")
    stream: bool = Field(False, description="Stream response")

@router.post("/v1/chat/completions")
@change_dir(router.dir)
async def chat_completions(request: ChatRequest):
    slm = router.llm_model

    # slm.history = [{"role": "system", "content": "You are a helpful assistant."}]
    slm.history = []
    slm.history += request.messages

    text = slm.apply_chat_template(slm.history)
    ids = slm.tokenizer(text).input_ids
    
    if not isinstance(slm.EOS, list):
        EOS = [slm.EOS]
    else:
        EOS = slm.EOS

    if request.stream:
        def generate_responses():
            token = slm.model.forward_first(ids)
            output_tokens = []
            while token not in EOS and slm.model.total_length < slm.model.SEQLEN:
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
            if token in EOS or slm.model.total_length >= slm.model.SEQLEN:
                break
            output_tokens += [token]
        slm.answer_cur = slm.tokenizer.decode(output_tokens)
        return JSONResponse({"choices": [{"message": {"role": "assistant", "content": slm.answer_cur}}]})


