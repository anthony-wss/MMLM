import librosa
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

import torch
from torch.nn.utils.rnn import pad_sequence


class MMLMUtility():
    def __init__(self, mmlm_model):
        self.mmlm_model = mmlm_model

    def tokenize_function(self, examples):
        model_inputs = self.mmlm_model.tokenizer(examples['input'] + examples['label'])
        labels = self.mmlm_model.tokenizer(examples['label'] + self.mmlm_model.tokenizer.eos_token)
        padding_size = len(model_inputs['input_ids']) - len(labels["input_ids"])
        model_inputs["label_ids"] = [-100] * padding_size + labels["input_ids"]
        return model_inputs

    class MMLMDataCollator:
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer

        def __call__(self, features):
            return {
                'input_ids': pad_sequence([torch.tensor(i['input_ids']) for i in features], batch_first=True,
                                          padding_value=self.tokenizer.eos_token_id),
                'labels': pad_sequence([torch.tensor(i['label_ids']) for i in features], batch_first=True,
                                       padding_value=-100),
            }


def load_audio_to_tensor(audio_input, sr=None):
    def resample_if_needed(audio, orig_sr, target_sr):
        """Resample the audio only if the original and target sampling rates differ."""
        return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr) if orig_sr != target_sr else audio

    # Load from file path with optional resampling
    if isinstance(audio_input, str):
        audio_array, orig_sr = librosa.load(audio_input, sr=None)
        if sr is not None:
            audio_array = resample_if_needed(audio_array, orig_sr, sr)
        audio_array = torch.tensor(audio_array).float()

    # Handle NumPy array input
    elif isinstance(audio_input, np.ndarray):
        if sr is not None:
            audio_input = resample_if_needed(audio_input, orig_sr=librosa.get_samplerate(audio_input), target_sr=sr)
        audio_array = torch.from_numpy(audio_input).float()

    # Handle list of arrays or lists
    elif isinstance(audio_input, list):
        audio_tensors = [
            torch.tensor(
                resample_if_needed(arr, librosa.get_samplerate(arr), sr)).float() if sr is not None else torch.tensor(
                arr).float()
            for arr in audio_input
        ]
        if all(t.dim() == 1 for t in audio_tensors):
            audio_array = pad_sequence(audio_tensors, batch_first=True)
        else:
            raise ValueError("All elements in the list must be 1D tensors or numpy arrays.")

    # Handle Torch Tensor directly
    elif isinstance(audio_input, torch.Tensor):
        audio_array = audio_input
        if sr is not None and audio_array.dim() == 1:
            audio_array = torch.tensor(
                resample_if_needed(audio_array.numpy(), orig_sr=librosa.get_samplerate(audio_array), target_sr=sr)
            ).float()
    else:
        raise ValueError("Unsupported audio input type")

    # Dimension adjustment
    if audio_array.dim() == 1:
        audio_array = audio_array.unsqueeze(0).unsqueeze(0)
    elif audio_array.dim() == 2:
        audio_array = audio_array.unsqueeze(1)
    elif audio_array.dim() == 3:
        pass  # No adjustment needed for 3D
    else:
        raise ValueError("Audio input has unsupported dimensions after processing")

    return audio_array
