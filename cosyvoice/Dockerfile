FROM modelscope:ubuntu20.04-py38-torch2.0.1-tf1.15.5-1.8.1

# RUN mkdir -p /root/.pip && \
#     echo "[global]" > /root/.pip/pip.conf && \
#     echo "index-url = https://pypi.tuna.tsinghua.edu.cn/simple" >> /root/.pip/pip.conf

# RUN git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git
RUN git clone --recursive https://mirror.ghproxy.com/https://github.com/FunAudioLLM/CosyVoice.git

WORKDIR /CosyVoice

RUN git submodule update --init --recursive

RUN pip install pynini==2.1.5

RUN pip uninstall -y tensorflow

RUN pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com

RUN apt update && apt-get install -y unzip sox libsox-dev

COPY main.py                                    \
     ttsfrd-0.3.6-cp38-cp38-linux_x86_64.whl    \
     /CosyVoice/

RUN pip install ./ttsfrd-0.3.6-cp38-cp38-linux_x86_64.whl

ENV PYTHONPATH third_party/Matcha-TTS

ENTRYPOINT ["python", "main.py"]