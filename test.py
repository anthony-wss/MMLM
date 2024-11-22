# from transformers import MoshiForConditionalGeneration
# import torch
#
# model = MoshiForConditionalGeneration.from_pretrained("kmhf/hf-moshiko")
# inputs = model.get_unconditional_inputs()
#
# print(inputs)
# # text_labels = inputs["input_ids"]
# text_labels = torch.tensor([[32000, 32000]])
# audio_labels = torch.stack([torch.tensor([2048] * text_labels.shape[1]) for _ in range(16)]).reshape([1, 8, -1])
# print(text_labels.shape)
# print(audio_labels.shape)
# outputs = model(**inputs, text_labels=text_labels, audio_labels=audio_labels)
# logits = outputs.logits
# logits.shape  # (bsz, seq_len, text_vocab_size)
# print(logits.shape)
# print(outputs.loss)
#  
# exit()


from mmlm.tts_model import LlamaTTSForCausalLM
from transformers import AutoTokenizer, PhiForCausalLM
import torch



model = LlamaTTSForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

prompt = "This is an example script ."
# prompt = "I"
inputs = tokenizer([prompt], return_tensors="pt")
text_labels = inputs["input_ids"]
audio_labels = torch.stack([torch.tensor([1024] * text_labels.shape[1]) for _ in range(8)]).reshape([1, 8, -1])
print(text_labels.shape)
print(audio_labels.shape)

outputs = model(**inputs, text_labels=text_labels, audio_labels=audio_labels)
print(outputs.loss)

