from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2PhonemeCTCTokenizer
from datasets import load_dataset
import torch
 
# load model and processor
# processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")

# model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")
# model = model.cuda()

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft")
# model = model.cuda()
    
    
# load dummy dataset and read soundfiles
ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")

# tokenize
input_values = processor(ds[0]["audio"]["array"], return_tensors="pt", padding="longest").input_values  # Batch size 1
# input_values = input_values.cuda()

# retrieve logits
logits = model(input_values).logits

# # take argmax and decode
# predicted_ids = torch.argmax(logits, dim=-1)
# transcription = processor.batch_decode(predicted_ids)

# print(transcription)

# # ['b ɪ k ʌ z j uː ɚ s l iː p ɪ ŋ ɪ n s t ɛ d ʌ v k 
# # ɑː ŋ k ɚ ɹ ɪ ŋ ð ə l ʌ v l i ɹ oʊ z p ɹ ɪ n s ɛ s h ɐ z b ɪ k ʌ m ɐ f ɪ d əl w ɪ ð aʊ ɾ ɐ b oʊ w ɑː l p ɔːɹ ɹ ʃ æ ɡ i s ɪ t s ð ɛ ɐ k uː ɪ ŋ d ʌ v']
