from typing import Optional, Union, Tuple

from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoModel, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast
from embedder import export_embedder
import torch
import torch.nn.functional as F


class MMLM(nn.Module):
    def __init__(
        self,
        lm_config,
        lm_model=None,
        lm_tokenizer=None,
        audio_config=1,
        audio_model=None,
        audio_adapter_config=None,
        visual_config=1,
        visual_model=None,
        visual_adapter_config=None,
    ):
        """
            MMLM model class. The pad token is set to [EOS].

            Parameters:

            - audio_config(int OR str): Use an int to represent the number of layers of the discrete feature.
                Use a str to represent the hf repo name for a continuous encoder.

            - audio_adapter_config(str): The name of audio adapter in model weights dict.

            - lm_config(str): hf repo name of the LLM.

            Attributes:

            - continue_audio_feature_type_ids(List[int]): Range of audio feature type ids. The right end number is not included.
                E.g. [39456, 39556)
        """
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Language Model Setup
        self.lm_model = (
            lm_model.to(self.device)
            if lm_model is not None
            else AutoModelForCausalLM.from_pretrained(lm_config).to(self.device)
        )
        self.tokenizer = (
            lm_tokenizer
            if lm_tokenizer
            else AutoTokenizer.from_pretrained(lm_config)
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Modality Configurations
        self.audio_config = audio_config
        self.visual_config = visual_config

        # Audio Feature Processing
        if isinstance(audio_config, int):
            self._setup_discrete_feature_weights(audio_config, "audio")
        else:
            self._setup_continuous_feature_processing(
                audio_config, audio_model, audio_adapter_config, "audio"
            )

        # Visual Feature Processing
        if isinstance(visual_config, int):
            self._setup_discrete_feature_weights(visual_config, "visual")
        else:
            self._setup_continuous_feature_processing(
                visual_config, visual_model, visual_adapter_config, "visual"
            )

        base_audio_tag_id = self.tokenizer.encode(f"CAUDIO_TAG_{0}")[0]
        self.continue_audio_feature_type_ids = [base_audio_tag_id, base_audio_tag_id + 100]

        base_continue_visual_tag_id = self.tokenizer.encode(f"CVISUAL_TAG_{0}")[0]
        self.continue_visual_feature_type_ids = [base_continue_visual_tag_id, base_continue_visual_tag_id + 100]

        base_discrete_audio_tag_id = self.tokenizer.encode(f"a_tok_{0}")[0]
        self.discrete_audio_feature_type_ids = [base_discrete_audio_tag_id, base_discrete_audio_tag_id + 1024 * 10]

        base_discrete_visual_tag_id = self.tokenizer.encode(f"v_tok_{0}")[0]
        self.discrete_visual_feature_type_ids = [base_discrete_visual_tag_id, base_discrete_visual_tag_id + 1024 * 10]

    def _setup_discrete_feature_weights(self, config, modality):
        """ 
            Add trainable parameters for the weights of each discrete feature layer.
            Larger weights for early RVQ layer.
        """
        learnable_weight_init = torch.arange(config, 0, step=-1).float().view(config, 1, 1)
        setattr(self, f"{modality}_learnable_weight", nn.Parameter(learnable_weight_init))

    def _setup_continuous_feature_processing(self, config, model, adapter_config, modality):
        """
            Add adapter for continuous feature.
            
            Parameters:

            - config(str): The huggingface repo name of the feature encoder.

            - adapter_config(str): The corresponding adapter name in "mmlm/embedder.py"
        """
        model = model.to(self.device) if model is not None else AutoModel.from_pretrained(config)
        setattr(self, f"{modality}_model", model)
        setattr(
            self,
            f"{modality}_adapter",
            export_embedder[adapter_config](
                input_size=model.config.hidden_size, output_size=self.lm_model.config.hidden_size
            ).to(self.device),
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        audio_features=None,
        vision_features=None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
            Parameters:

            - labels(List[List[torch.LongTensor]]): Shape [batch_num, max_seq_len]
        """

        output_attentions = output_attentions if output_attentions is not None else self.lm_model.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.lm_model.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.lm_model.config.use_return_dict

        # The inputs need to be tokenized.
        if inputs_embeds is None:
            embeder = self.lm_model.get_input_embeddings()
            inputs_embeds = []
            for batch_num, batch_input in enumerate(input_ids):
                audio_discrete_token = []
                visual_discrete_token = []
                text_ids = []
                input_embeds = []
                for i in batch_input:

                    # case 1: `i` is discrete audio token
                    if self.discrete_audio_feature_type_ids[0] < i < self.discrete_audio_feature_type_ids[1]:
                        audio_discrete_token.append(i)

                        # Clear text_ids by appending it to `input_embeds`.
                        if text_ids:
                            input_embeds.append(embeder(torch.LongTensor(text_ids).to(self.device)))
                        text_ids = []

                    # case 2: `i` is discrete visual token
                    elif self.discrete_visual_feature_type_ids[0] < i < self.discrete_visual_feature_type_ids[1]:
                        visual_discrete_token.append(i)

                        # Clear text_ids by appending it to `input_embeds`.
                        if text_ids:
                            input_embeds.append(embeder(torch.LongTensor(text_ids).to(self.device)))
                        text_ids = []

                    # case 3: `i` is text token
                    else:
                        text_ids.append(i)
                        if len(audio_discrete_token) > 0:

                            # Truncate audio_discrete_token to a multiple of self.audio_config
                            audio_discrete_token = audio_discrete_token[
                                :len(audio_discrete_token) // self.audio_config * self.audio_config
                            ]

                            # (N, ) -> (self.audio_config, L)
                            discrete_audio_input_id = torch.tensor(audio_discrete_token, dtype=torch.long).view(
                                self.audio_config, -1
                            )

                            discrete_audio_input_ids = []
                            # Embed the discrete audio input layer by layer
                            for i in range(self.audio_config):
                                input_scale = embeder(discrete_audio_input_id[i, :].to(self.device))
                                discrete_audio_input_ids.append(input_scale)

                            # Weighted sum. Larger weight for early layer.
                            weighted_discrete_inputs_embeds = torch.mul(
                                torch.stack(discrete_audio_input_ids, dim=0).to(self.device),
                                F.softmax(self.audio_learnable_weight, dim=0).to(self.device)
                            )
                            weighted_discrete_inputs_embeds = torch.sum(weighted_discrete_inputs_embeds, dim=0)

                            if discrete_audio_input_ids:
                                input_embeds.append(weighted_discrete_inputs_embeds)
                            audio_discrete_token = []

                        # Simiar to audio token, but visual token is not hierarchical
                        elif len(visual_discrete_token) > 0:
                            discrete_visual_input_id = torch.tensor(visual_discrete_token).view(self.visual_config, -1)
                            discrete_visual_input_ids = []
                            for i in range(self.visual_config):
                                input_scale = embeder(discrete_visual_input_id[i, :].to(self.device))
                                discrete_visual_input_ids.append(input_scale)
                            weighted_discrete_inputs_embeds = torch.mul(
                                torch.stack(discrete_visual_input_ids, dim=0).to(self.device),
                                F.softmax(self.visual_learnable_weight, dim=0).to(self.device)
                            )
                            weighted_discrete_inputs_embeds = torch.sum(weighted_discrete_inputs_embeds, dim=0)
                            if discrete_visual_input_ids:
                                input_embeds.append(weighted_discrete_inputs_embeds)
                            visual_discrete_token = []
                if text_ids:
                    input_embeds.append(embeder(torch.LongTensor(text_ids).to(self.device)))
                inputs_embeds.append(torch.cat(input_embeds))
            max_length = max(tensor.size(0) for tensor in inputs_embeds)
            inputs_embeds = torch.stack([
                F.pad(tensor, (0, 0, max_length - tensor.size(0), 0), "constant", 0)
                for tensor in inputs_embeds
            ])
            outputs = self.lm_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                use_cache=False,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # Single batch of input_embeds is given. No tokenization needed.
        elif input_ids.shape[-1] == 1:
            outputs = self.lm_model(
                input_ids=input_ids,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # What is this case for?
        elif self.audio_config or self.visual_config:
            for batch_num, batch_input in enumerate(input_ids):
                vision_features_id = 0
                audio_features_id = 0
                for pos, ids in enumerate(batch_input):
                    if self.continue_audio_feature_type_ids[0] < ids < self.continue_audio_feature_type_ids[1]:
                        audio_feature = self.audio_adapter(audio_features[batch_num][audio_features_id]).to(self.device)
                        audio_features_id += 1
                        inputs_embeds = torch.cat(
                            (inputs_embeds[:, :pos, :], audio_feature, inputs_embeds[:, pos + 1:, :]), dim=1
                        ).to(self.device)
                    if self.continue_visual_feature_type_ids[0] < ids < self.continue_visual_feature_type_ids[1]:
                        vision_features = self.visual_adapter(vision_features[batch_num][vision_features_id])
                        vision_features_id += 1
                        inputs_embeds = torch.cat(
                            (inputs_embeds[:, :pos, :], vision_features, inputs_embeds[:, pos + 1:, :]), dim=1
                        )
            outputs = self.lm_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                use_cache=False,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # Shape of `logits`: [batch_size, num_token, num_classes]
        logits = outputs[0]

        loss = None
        if labels is not None:
            shift_logits = logits.reshape(-1, logits.size(-1)) # Shape: [batch_size * num_token, num_classes]
            
            labels = labels[:, -logits.shape[1]:]
            shift_labels = labels.reshape(-1) # Shape: [batch_size * num_token]

            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
        )

    def generate(self, input_ids, audio_feature=None, max_length=50):
        """ Greedy decoding. """
        self.eval()
        generated = input_ids.to(self.device)
        begin_gen_pos = input_ids.shape[1]
        with torch.no_grad():
            for _ in range(max_length):
                outputs = self.forward(input_ids=generated, audio_features=audio_feature)
                next_token_logits = outputs.logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                generated = torch.cat((generated, next_token), dim=-1)
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
        return generated[:, begin_gen_pos:-1]
