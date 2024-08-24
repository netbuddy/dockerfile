from flask import Flask, request, jsonify
import audio_processor as ap

app = Flask(__name__)

@app.route('/recognize', methods=['POST'])
def recognize():
#    if 'audio_data' not in request.files:
#        return jsonify({"error": "No audio data received"}), 400

    audio_data = request.data
    # Save the received audio data to a file
    with open("output.wav", "wb") as audio_file:
        audio_file.write(audio_data)
    
    print("开始识别")
    text = processor.get_text_from_audio("output.wav")
    print("识别出的正文为")
    print(text)
    return text

if __name__ == '__main__':
    print("开始初始化语音识别大模型")
    model_dir = "iic/SenseVoiceSmall"
    processor = ap.AudioProcessor(model_dir, language="zh")
    print("语音识别大模型初始化完成")
    app.run(host='0.0.0.0', port=8001, debug=False)
