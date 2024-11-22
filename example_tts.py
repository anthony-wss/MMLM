"""
    Implementing https://arxiv.org/pdf/2402.08846v1

    - LLM: meta-llama/Llama-3.2-3B-Instruct

    - Speech encoder: ntu-spml/distilhubert

    - adapter: single linear layer
"""

from mmlm.tts_model import LlamaTTSForCausalLM, TTSUtility
from transformers import AutoTokenizer, PhiForCausalLM
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

tts_model = LlamaTTSForCausalLM.from_pretrained("voidful/Llama-3.2-11B-TTS")
tokenizer = AutoTokenizer.from_pretrained("voidful/Llama-3.2-11B-TTS")

# Freeze tts_model's LLM
for param in tts_model.model.parameters():
    param.requires_grad = False

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

from datasets import load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, AutoModel


# lm_model = AutoModelForCausalLM.from_pretrained('voidful/Llama-3.2-3B-Instruct')
# lm_tokenizer = AutoTokenizer.from_pretrained('voidful/Llama-3.2-3B-Instruct')
# audio_model = AutoModel.from_pretrained('ntu-spml/distilhubert')

# mmlm = MMLM(
#     lm_model=lm_model,
#     lm_tokenizer=lm_tokenizer,
#     audio_model=audio_model,
# )
# mmlu = MMLMUtility(mmlm)

# dataset = load_dataset("anthony-wss/moshi_tts_dataset_dummy")
dataset = load_from_disk("./moshi_tts_dataset_dummy")

# TODO: move this into TTSUtility
# The bug is that the one in TTSUtility will not be executed
import torch
def tokenize_functioN(examples):
    """ The map function for dataset `voidful/cv_13_tw_speech_tokenizer_asr` """
    model_inputs = tokenizer(examples['input'], add_special_tokens=False)
    model_inputs["audio_labels"] = torch.tensor(examples["label"]).squeeze()
    return model_inputs

tts_util = TTSUtility(tokenizer)
tokenized_datasets = dataset.map(tokenize_functioN, batched=False)

# mmlm.tokenizer.pad_token = mmlm.tokenizer.eos_token
training_args = TrainingArguments(
    output_dir='./results_asr',
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    logging_steps=1,
    num_train_epochs=10,
    weight_decay=0.01,
    logging_dir='./logs',
    bf16=True
)

# Initialize the Trainer
trainer = Trainer(
    model=tts_model,
    args=training_args,
    # train_dataset=tokenized_datasets["train"],
    # eval_dataset=tokenized_datasets["train"],
    train_dataset=tokenized_datasets,
    eval_dataset=tokenized_datasets,
    tokenizer=tokenizer,
    data_collator=tts_util.MMLMDataCollator(tokenizer)
)

trainer.train()
