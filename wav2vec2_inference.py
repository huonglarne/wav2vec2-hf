
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2PhonemeCTCTokenizer
from datasets import load_dataset
import torch

import soundfile as sf
import librosa

speech_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
speech_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")
speech_model = speech_model.cuda()

phoneme_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft")
phoneme_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft")
phoneme_model = phoneme_model.cuda()
    
def get_transcript(processor, model, input):
    input_values = processor(input, return_tensors="pt", padding="longest").input_values  # Batch size 1
    input_values = input_values.cuda()
    logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)

    return transcription
    
# load dummy dataset and read soundfiles
# ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")
# input = ds[0]["audio"]["array"]

input, _ = librosa.load("data/audio/Read83.m4a", sr = 16000)

speech = get_transcript(speech_processor, speech_model, input)
phoneme = get_transcript(phoneme_processor, phoneme_model, input)

print(speech)

print(phoneme)