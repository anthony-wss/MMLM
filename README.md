# MMLM (Multi-Modality Learning Model)

## Introduction

## Usage

data format(input/output):  
`instruction <audio> audio file path </audio> instruction <image> image file path </image> instruction`

## Documentation

### `mmlm/`

- `model.py`: Define the MMLM module, including the weighted sum of audio/visual input features.



### `patch/`

- `patch_llama_model.py`: 

    1. Add audio and vision token to the tokenizer of Llama-3.2-3B-Instruct. View this file for details about the added tokens.

    2. Push the llama with new tokenizer to huggingface.


