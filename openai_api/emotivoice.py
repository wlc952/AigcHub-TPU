from pydantic import BaseModel, Field
import base64
from api.base_api import BaseAPIRouter, change_dir, init_helper
import os, io
from typing import Optional
from fastapi import Response
import soundfile as sf
from pydub import AudioSegment
import uuid

app_name = "emotivoice"

def convert(src_wav, tgt_wav, tone_color_converter, get_se, save_path="./temp/output.wav", encode_message=""):
    try:
        # extract the tone color features of the source speaker and target speaker
        source_se, _ = get_se(src_wav, tone_color_converter, target_dir='processed', vad=True)
        target_se, _  = get_se(tgt_wav, tone_color_converter, target_dir='processed', vad=True)
    except Exception as e:
        return {"error": f"Failed to extract speaker embedding: {e}"}
    tone_color_converter.convert(
        audio_src_path=src_wav, 
        src_se=source_se, 
        tgt_se=target_se, 
        output_path=save_path,
        message=encode_message)
    return save_path

class AppInitializationRouter(BaseAPIRouter):
    dir = f"repo/{app_name}"
    @init_helper(dir)
    async def init_app(self):
        from repo.emotivoice.demo_page import get_models
        models, tone_color_converter, g2p, lexicon = get_models()
        self.models = {
            "models": models, 
            "tone_color_converter": tone_color_converter,
            "g2p": g2p, 
            "lexicon": lexicon
        }
        return {"message": f"应用 {self.app_name} 已成功初始化。"}
    
    async def destroy_app(self):
        del models, tone_color_converter, g2p, lexicon

router = AppInitializationRouter(app_name=app_name)


### 音色转换；兼容openai api，audio/translation
class ConversionRequest(BaseModel):
    ## 有意义的兼容参数
    file: str = Field(..., description="要转换语音的路径")
    prompt: str = Field(..., description="参考音色的路径") 
    ## 无意义的兼容参数
    response_format: Optional[str] = Field('', description="（形式参数无意义）")
    model: Optional[str] = Field("", description="（形式参数无意义）")
    temperature: Optional[float] = Field(0.0, description="（形式参数无意义）")

@router.post("/v1/audio/translation")
@change_dir(router.dir)
async def voice_changer(request: ConversionRequest):
    from repo.emotivoice.tone_color_conversion import get_se
    save_path = convert(src_wav=request.file, tgt_wav=request.prompt, tone_color_converter=router.models['tone_color_converter'], get_se=get_se, encode_message='Airbox')
    if isinstance(save_path, dict):
        return {"text": save_path['error'], 'info': 'error message'}

    with open(save_path, 'rb') as file:
        audio_data = file.read()
    audio_base64 = base64.b64encode(audio_data).decode()
    return {"text": audio_base64, 'info': 'text is the base64 encoded audio'}


### 文本转语音 以及 音色克隆；兼容openai api，audio/speech
class TTSRequest(BaseModel):
    ## 有意义的兼容参数
    input: str = Field(..., description="要转换为语音的文本")
    voice: Optional[str] = Field('8051', description="说话人ID")
    response_format: Optional[str] = Field('wav', description="音频格式")

    ## 专有参数
    emotion: Optional[str] = Field('', description="情感提示（可选）")
    audio_path: Optional[str] = Field('', description="要参考的目标音色（可选）")

    ## 无意义的兼容参数
    model: Optional[str] = Field("", description="（形式参数无意义）")
    speed: Optional[float] = Field(1.0, description="（形式参数无意义）")

@router.post("/v1/audio/speech")
@change_dir(router.dir)
async def text_to_speech(request: TTSRequest):    
    from repo.emotivoice.demo_page import tts
    from repo.emotivoice.tone_color_conversion import get_se

    _name = f'./temp/{str(uuid.uuid4())}.wav'
    src_wav = tts(request.input, request.emotion, request.voice, _name,
                  router.models['models'], router.models['g2p'], router.models['lexicon'])
    save_path = _name
    if request.audio_path and os.path.exists(request.audio_path):
        save_path = convert(src_wav=src_wav, tgt_wav=request.prompt, tone_color_converter=router.models['tone_color_converter'], get_se=get_se, encode_message='Airbox')
        if isinstance(save_path, dict):
            return {"text": save_path['error'], 'info': 'error message'}
    np_audio, sr = sf.read(save_path)
    wav_buffer = io.BytesIO()
    sf.write(file=wav_buffer, data=np_audio, samplerate=sr, format='WAV')
    buffer = wav_buffer
    response_format = request.response_format
    if response_format != 'wav':
        wav_audio = AudioSegment.from_wav(wav_buffer)
        wav_audio.frame_rate=sr
        buffer = io.BytesIO()
        wav_audio.export(buffer, format=response_format)

    return Response(content=buffer.getvalue(), media_type=f"audio/{response_format}")

# 测试用指令
# curl http://0.0.0.0:8000/emotivoice/v1/audio/speech -H "Content-Type: application/json" \
# -d '{
#   "input": "大家好啊",
#   "voice": "8051",
#   "response_format": "wav",
#   "model": "emotivoice",
#   "speed": 1,
#   "emotion": "",
#   "audio_path": ""
# }' --output speech.wav