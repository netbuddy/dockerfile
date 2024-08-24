FROM modelscope:ubuntu22.04-py310-torch2.3.0-1.16.0

# RUN mkdir -p /root/.pip && \
#     echo "[global]" > /root/.pip/pip.conf && \
#     echo "index-url = https://pypi.tuna.tsinghua.edu.cn/simple" >> /root/.pip/pip.conf

# RUN git clone https://github.com/FunAudioLLM/SenseVoice
RUN git clone https://mirror.ghproxy.com/https://github.com/FunAudioLLM/SenseVoice

WORKDIR /SenseVoice

RUN pip install -r requirements.txt

COPY main.py               \
     audio_processor.py    \
     /SenseVoice/

ENTRYPOINT ["python", "main.py"]