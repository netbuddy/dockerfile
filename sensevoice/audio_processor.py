from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

class AudioProcessor:
    def __init__(self, model_dir, language="zh"):
        self.model_dir = model_dir
        self.language = language
        self.model = self._initialize_model()

    def _initialize_model(self):
        model = AutoModel(
            model=self.model_dir,
            trust_remote_code=True,
            remote_code="./model.py",
            vad_model="fsmn-vad",
            vad_kwargs={"max_single_segment_time": 30000},
            device="cuda:0",
            disable_update=True,
            check_latest = False
        )
        return model

    def process_audio(self, audio_path):
        res = self.model.generate(
            input=audio_path,
            cache={},
            language=self.language,
            use_itn=True,
            batch_size_s=60,
            merge_vad=True,
            merge_length_s=15,
        )
        return res

    def get_text_from_audio(self, audio_path):
        result = self.process_audio(audio_path)
        text = rich_transcription_postprocess(result[0]["text"])
        return text

# Example usage
if __name__ == "__main__":
    model_dir = "/home/public/models/iic/SenseVoiceSmall"
    processor = AudioProcessor(model_dir, language="zh")

    audio_path = "output.wav"
    text = processor.get_text_from_audio(audio_path)
    print(text)
