from transformers.models.llama import LlamaForCausalLM
from transformers.models.llama.modeling_llama import KwargsForCausalLM
from transformers.models.moshi.modeling_moshi import MoshiDepthDecoder
from transformers.models.moshi.configuration_moshi import MoshiDepthConfig
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.processing_utils import Unpack

from torch import nn
import torch
from typing import List, Optional, Tuple, Union
from transformers.modeling_outputs import CausalLMOutputWithPast
import torch.nn.functional as F
import torch
from torch.nn.utils.rnn import pad_sequence



class LlamaTTSForCausalLM(LlamaForCausalLM):
    # TODO: check if this helps memory usage
    # _tied_weights_keys = ["lm_head.weight"]

    # Copied from transformers.models.phi.modeling_phi.PhiForCausalLM.__init__
    def __init__(self, config):
        super().__init__(config)
        # self.speech_token_heads = [nn.Linear(config.hidden_size, SPEECH_TOKEN_DICT_SIZE, bias=True) for _ in range(NUM_RVQ_LAYERS)]
        depth_config = MoshiDepthConfig(vocab_size=62108, input_size=3072)
        self.depth_decoder = MoshiDepthDecoder(depth_config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        text_labels: Optional[torch.LongTensor] = None,
        audio_labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            num_logits_to_keep (`int`, *optional*):
                Calculate logits for the last `num_logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
            logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])

        loss = None
        if text_labels is not None:
            loss = self.loss_function(logits=logits, labels=text_labels, vocab_size=self.config.vocab_size, **kwargs)

        # Copy from transformers.models.moshi.modeling_moshi.py > forward
        kwargs_depth_decoder = {
            argument[len("depth_decoder_") :]: value
            for argument, value in kwargs.items()
            if argument.startswith("depth_decoder_")
        }
        if text_labels is not None and audio_labels is not None:
            print(text_labels.shape)
            print(audio_labels.shape)
            # (batch_size, sequence_length) -> (batch_size * sequence_length, 1)
            text_labels = text_labels.view(-1, 1)

            # (batch_size, num_codebooks, sequence_length) -> (batch_size * sequence_length, num_codebooks)
            audio_labels = audio_labels.transpose(1, 2).reshape(-1, audio_labels.shape[1])

            
            depth_input_ids = torch.cat([text_labels, audio_labels], dim=1)
            # keep the last codebook out of input_ids
            depth_input_ids = depth_input_ids[:, :-1]

            decoder_last_hidden_state = outputs.last_hidden_state
            decoder_last_hidden_state = decoder_last_hidden_state.view(-1, 1, decoder_last_hidden_state.shape[-1])
            depth_decoder_outputs = self.depth_decoder(
                last_hidden_state=decoder_last_hidden_state,
                input_ids=depth_input_ids,
                attention_mask=attention_mask,
                labels=audio_labels,
                **kwargs_depth_decoder,
            )

            if loss is None:
                loss = depth_decoder_outputs.loss
            else:
                loss += depth_decoder_outputs.loss
         

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
    # def forward(
    #     self,
    #     input_ids: torch.LongTensor = None,
    #     attention_mask: Optional[torch.Tensor] = None,
    #     position_ids: Optional[torch.LongTensor] = None,
    #     past_key_values: Optional[List[torch.FloatTensor]] = None,
    #     inputs_embeds: Optional[torch.FloatTensor] = None,
    #     text_labels: Optional[torch.LongTensor] = None,
    #     audio_labels: Optional[torch.LongTensor] = None,
    #     use_cache: Optional[bool] = None,
    #     output_attentions: Optional[bool] = None,
    #     output_hidden_states: Optional[bool] = None,
    #     return_dict: Optional[bool] = None,
    #     cache_position: Optional[torch.LongTensor] = None,
    #     num_logits_to_keep: int = 0,
    #     **kwargs,
    # ) -> Union[Tuple, CausalLMOutputWithPast]:
    #     r"""
    #     Args:
    #         labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
    #             Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
    #             config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
    #             (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
    #
    #         num_logits_to_keep (`int`, *optional*):
    #             Calculate logits for the last `num_logits_to_keep` tokens. If `0`, calculate logits for all
    #             `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
    #             token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
    #
    #     Returns:
    #
    #     Example:
    #
    #     ```python
    #     >>> from transformers import AutoTokenizer, PhiForCausalLM
    #
    #     >>> model = PhiForCausalLM.from_pretrained("microsoft/phi-1")
    #     >>> tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1")
    #
    #     >>> prompt = "This is an example script ."
    #     >>> inputs = tokenizer(prompt, return_tensors="pt")
    #
    #     >>> # Generate
    #     >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
    #     >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    #     'This is an example script .\n\n\n\nfrom typing import List\n\ndef find_most_common_letter(words: List[str'
    #     ```"""
    #
    #     output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    #     output_hidden_states = (
    #         output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    #     )
    #     return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    #
    #     # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    #     outputs = self.model(
    #         input_ids=input_ids,
    #         attention_mask=attention_mask,
    #         position_ids=position_ids,
    #         past_key_values=past_key_values,
    #         inputs_embeds=inputs_embeds,
    #         use_cache=use_cache,
    #         output_attentions=output_attentions,
    #         output_hidden_states=output_hidden_states,
    #         return_dict=return_dict,
    #         cache_position=cache_position,
    #     )
    #
    #     hidden_states = outputs[0]
    #     # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
    #     logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])
    #
    #     loss = None
    #     if text_labels is not None:
    #         loss = self.loss_function(logits, text_labels, self.vocab_size, **kwargs)
    #
    #     # Copy from transformers.models.moshi.modeling_moshi.py > forward
    #     kwargs_depth_decoder = {
    #         argument[len("depth_decoder_") :]: value
    #         for argument, value in kwargs.items()
    #         if argument.startswith("depth_decoder_")
    #     }
    #     if text_labels is not None and audio_labels is not None:
    #         # (batch_size, sequence_length) -> (batch_size * sequence_length, 1)
    #         text_labels = text_labels.view(-1, 1)
    #
    #         # (batch_size, num_codebooks, sequence_length) -> (batch_size * sequence_length, num_codebooks)
    #         audio_labels = audio_labels.transpose(1, 2).reshape(-1, audio_labels.shape[1])
    #
    #         depth_input_ids = torch.cat([text_labels, audio_labels], dim=1)
    #         # keep the last codebook out of input_ids
    #         depth_input_ids = depth_input_ids[:, :-1]
    #
    #         decoder_last_hidden_state = outputs.last_hidden_state
    #         decoder_last_hidden_state = decoder_last_hidden_state.view(-1, 1, decoder_last_hidden_state.shape[-1])
    #         depth_decoder_outputs = self.depth_decoder(
    #             last_hidden_state=decoder_last_hidden_state,
    #             input_ids=depth_input_ids,
    #             attention_mask=attention_mask,
    #             labels=audio_labels,
    #             **kwargs_depth_decoder,
    #         )
    #         loss += depth_decoder_outputs.loss
    #
    #     if not return_dict:
    #         output = (logits,) + outputs[1:]
    #         return (loss,) + output if loss is not None else output
    #
    #     return CausalLMOutputWithPast(
    #         loss=loss,
    #         logits=logits,
    #         past_key_values=outputs.past_key_values,
    #         hidden_states=outputs.hidden_states,
    #         attentions=outputs.attentions,
    #     )


class TTSUtility():
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def tokenize_function(self, examples):
        """ The map function for dataset `voidful/cv_13_tw_speech_tokenizer_asr` """
        model_inputs = self.tokenizer(examples['input'], add_special_tokensS=False)
        model_inputs["audio_labels"] = torch.tensor(examples["label"]).squeeze()
        return model_inputs

    class MMLMDataCollator:
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer

        def __call__(self, features):
            """ Pad the batched inputs """

            # TODO: Support batch training
            audio_labels = torch.tensor(features[0]["label"]).unsqueeze(0)
            input_ids = torch.tensor(features[0]["input_ids"]).unsqueeze(0)
            print(input_ids.shape)
            return {
                "input_ids": input_ids,
                "audio_labels": audio_labels,
                "text_labels": input_ids
            }
            # batch_text_labels = [torch.tensor(i['input_ids']).squeeze() for i in features]
            # for i in range(len(batch_text_labels)):
            #     batch_text_labels[i] = batch_text_labels[i][:features[i]['input_ids'].shape[1]]
            # 
            #
            # batch_audio_labels = [torch.tensor(i['audio_labels']).squeeze() for i in features] 
            # max_len = max(t.shape[1] for t in batch_audio_labels)
            # batch_audio_labels = torch.stack([
            #     torch.nn.functional.pad(t, (0, max_len - t.shape[1]), value=-100) for t in batch_audio_labels
            # ])
            #
            # return {
            #     'input_ids': batch_text_labels,
            #     'text_labels': batch_text_labels,
            #     'audio_labels': batch_audio_labels
            # }

