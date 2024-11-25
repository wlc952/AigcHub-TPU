本项目提供端到端的语音对话助手项目，asr + llm + tts级联实现，在Airbox上的语音响应时延小于2s。

## 使用方式一
盒子连接电脑，然后启动服务
```bash
bash AigcHub-TPU/run_api.sh
```
电脑本地创建新文件夹，下载AigcHub-TPU/run_a2a_windows.py，自行pip安装所需的库。

```bash
python run_a2a_windows.py
```
服务将在`http://localhost:5000/`

## 使用方式二
盒子连接电脑，然后启动服务
```bash
bash AigcHub-TPU/run_api.sh
```

盒子端开启网页前端
```bash
bash AigcHub-TPU/run.sh
```
服务将在`http://盒子ip:5000/`

开启http网页的麦克风权限，参考https://blog.csdn.net/zwj1030711290/article/details/125425877

## 探索方式三
盒子连接电脑，然后启动服务
```bash
bash AigcHub-TPU/run_api.sh
```
盒子端开启实时语音聊天
```bash
source hub_venv/bin/activate
python  run_a2a_alsa.py
```
