from flask import Flask, request, send_file
from flask_cors import CORS
from cosyvoice.cli.cosyvoice import CosyVoice
from cosyvoice.utils.file_utils import load_wav
import torchaudio

app = Flask(__name__)
CORS(app)  # 为跨域请求添加 CORS 支持

class CosyVoiceService:
    def __init__(self, model_path):
        self.cosyvoice = CosyVoice(model_path)

    def list_available_spks(self):
        return self.cosyvoice.list_avaliable_spks()

    def synthesize_audio(self, text, speaker):
        output = self.cosyvoice.inference_sft(text, speaker)
        return output['tts_speech']

    def save_audio(self, audio, filename):
        torchaudio.save(filename, audio, 22050)
        return filename


# 初始化服务
service = CosyVoiceService('pretrained_models/CosyVoice-300M-SFT')

@app.route('/synthesize', methods=['POST'])
def synthesize():
    data = request.json
    text = data.get('text', '')
    speaker = data.get('speaker', '中文女')

    if not text:
        return {"error": "Missing required field 'text'"}, 400

    try:
        audio = service.synthesize_audio(text, speaker)
        filename = 'temp_audio.wav'
        file_path = service.save_audio(audio, filename)

        # 返回生成的音频文件
        return send_file(file_path, as_attachment=True, download_name='synthesized.wav')
    except Exception as e:
        return {"error": str(e)}, 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8002, debug=False)
