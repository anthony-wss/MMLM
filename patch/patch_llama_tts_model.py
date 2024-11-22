import nlp2
import torch
import copy
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoProcessor

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-11B-Vision")
pretrained = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-11B-Vision")

processor.tokenizer.push_to_hub("voidful/Llama-3.2-11B-ASR")
pretrained.push_to_hub("voidful/Llama-3.2-11B-ASR")



"""

Patch the LLaMA with 8 more prediction head for predicting speech units.

"""
# import nlp2
import torch
import copy
from transformers import AutoTokenizer, AutoModelForCausalLM
from mmlm.tts_model import LlamaTTSForCausalLM, TTSUtility

# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
# pretrained = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
pretrained = LlamaTTSForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
processor = AutoProcessor.from_pretrained("openai/whisper-large-v3-turbo")

whisper_dict = processor.tokenizer.get_vocab()
llama_dict = tokenizer.get_vocab()

intersection = {i: llama_dict[i]
                for i in set(whisper_dict.keys()).intersection(set(llama_dict.keys()))}
symmetric_difference = {i: whisper_dict[i]
                for i in set(intersection.keys()).symmetric_difference(set(whisper_dict.keys()))}

# asset len(intersection)+len(symmetric_difference) should be equal to len(whisper_dict)
assert len(intersection)+len(symmetric_difference) == len(whisper_dict)

# backup embedding layer
state_dict_weight_clone = pretrained.get_input_embeddings().state_dict()['weight'].clone()
# create new inputting embedding layer with size equal to len(whisper_dict)
pretrained.resize_token_embeddings(len(whisper_dict))
input_embedding = pretrained.get_input_embeddings()
state_dict_weight = input_embedding.state_dict()['weight']
for i in intersection:
    state_dict_weight[whisper_dict[i]] = state_dict_weight_clone[llama_dict[i]]
pretrained.set_input_embeddings(input_embedding)

# check if the new embedding layer is correctly set
assert torch.equal(pretrained.get_input_embeddings().state_dict()['weight'], state_dict_weight)

print(tokenizer.tokenize(
    "<|begin_of_text|>我是<PAD><PAD><EPAD>曾<PAD><PAD><EPAD>沛<EPAD>慈<PAD><EPAD>歡迎<PAD><PAD><PAD><EPAD>收<PAD><PAD><EPAD>聽<|end_of_text|>"))

add_tokens = (["<PAD>", "<EPAD>"])

origin_vocab_size = tokenizer.vocab_size
print("===ADD TOKEN===")
num_added_toks = tokenizer.add_tokens(add_tokens)
print('We have added', num_added_toks, 'tokens')
print(origin_vocab_size, num_added_toks, len(tokenizer))
print(tokenizer.tokenize(
    "<|begin_of_text|>我是<PAD><PAD><EPAD>曾<PAD><PAD><EPAD>沛<EPAD>慈<PAD><EPAD>歡迎<PAD><PAD><PAD><EPAD>收<PAD><PAD><EPAD>聽<|end_of_text|>"))

print("===============")

pretrained.resize_token_embeddings(len(tokenizer))
input_embedding = pretrained.get_input_embeddings()
state_dict_weight = input_embedding.state_dict()['weight']
print(state_dict_weight.shape)
random_indices = torch.randperm(origin_vocab_size)[:num_added_toks]
new_tokens_weights = state_dict_weight[random_indices]
state_dict_weight[origin_vocab_size:origin_vocab_size + num_added_toks] = copy.copy(new_tokens_weights)
pretrained.set_input_embeddings(input_embedding)
print("===============")

tokenizer.save_pretrained("llama-3.2-1b-tts")
pretrained.save_pretrained("llama-3.2-1b-tts")


# tokenizer.push_to_hub("anthony-wss/llama-3.2-1b-tts")
# pretrained.push_to_hub("anthony-wss/llama-3.2-1b-tts")
