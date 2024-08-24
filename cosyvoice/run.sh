#!/bin/bash
docker build -t cosyvoice-cpu .
docker run -it --name cosyvoice -v ${HOME}/wp/CosyVoice/pretrained_models:/CosyVoice/pretrained_models -p 8002:8002 cosyvoice-cpu
