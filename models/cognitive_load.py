import torch
import torch.nn as nn

class CognitiveLoadFeatureExtractor(nn.Module):
    def __init__(self, pretrained_model_path):
        super(CognitiveLoadFeatureExtractor, self).__init__()
        checkpoint = torch.load(pretrained_model_path)
        
        
class MAF_acoustic(nn.Module):
    def __init__(self,
                dim_model,
                dropout_rate):
        super(MAF_acoustic, self).__init__()
        self.dropout_rate = dropout_rate

        self.acoustic_context_transform = nn.Linear(ACOUSTIC_MAX_LEN, SOURCE_MAX_LEN, bias = False)
        # self.visual_context_transform = nn.Linear(VISUAL_MAX_LEN, SOURCE_MAX_LEN, bias = False)

        self.acoustic_context_attention = ContextAwareAttention(dim_model=dim_model,
                                                                dim_context=ACOUSTIC_DIM,
                                                                dropout_rate=dropout_rate)

        # self.visual_context_attention = ContextAwareAttention(dim_model=dim_model,
        #                                                     dim_context=VISUAL_DIM,
        #                                                     dropout_rate=dropout_rate)

        self.acoustic_gate = nn.Linear(2*dim_model, dim_model)
        # self.visual_gate = nn.Linear(2*dim_model, dim_model)
        self.dropout_layer = nn.Dropout(dropout_rate)
        self.final_layer_norm = nn.LayerNorm(dim_model)

    def forward(self,
                text_input,
                acoustic_context):

        acoustic_context = acoustic_context.permute(0,2,1)
        acoustic_context = self.acoustic_context_transform(acoustic_context.float())
        acoustic_context = acoustic_context.permute(0,2,1)

        audio_out = self.acoustic_context_attention(q=text_input,
                                                    k=text_input,
                                                    v=text_input,
                                                    context=acoustic_context)
        weight_a = F.sigmoid(self.acoustic_gate(torch.cat([text_input, audio_out], dim=-1)))
        # weight_v = F.sigmoid(self.visual_gate(torch.cat([text_input, video_out], dim=-1)))

        # output = self.final_layer_norm(text_input + weight_a * audio_out + weight_v * video_out)

        output = self.final_layer_norm(text_input + weight_a * audio_out)

        return output
    
    
class MAF_visual(nn.Module):
    def __init__(self,
                dim_model,
                dropout_rate):
        super(MAF_visual, self).__init__()
        self.dropout_rate = dropout_rate

        # self.acoustic_context_transform = nn.Linear(ACOUSTIC_MAX_LEN, SOURCE_MAX_LEN, bias = False)
        self.visual_context_transform = nn.Linear(VISUAL_MAX_LEN, SOURCE_MAX_LEN, bias = False)
        self.visual_context_attention = ContextAwareAttention(dim_model=dim_model,
                                                            dim_context=VISUAL_DIM,
                                                            dropout_rate=dropout_rate)

        # self.acoustic_gate = nn.Linear(2*dim_model, dim_model)
        self.visual_gate = nn.Linear(2*dim_model, dim_model)
        self.dropout_layer = nn.Dropout(dropout_rate)
        self.final_layer_norm = nn.LayerNorm(dim_model)

    def forward(self,
                text_input,
                visual_context):
        visual_context = visual_context.permute(0,2,1)
        visual_context = self.visual_context_transform(visual_context.float())
        visual_context = visual_context.permute(0,2,1)

        video_out = self.visual_context_attention(q=text_input,
                                                    k=text_input,
                                                    v=text_input,
                                                    context=visual_context)

        weight_v = F.sigmoid(self.visual_gate(torch.cat([text_input, video_out], dim=-1)))

        # output = self.final_layer_norm(text_input + weight_a * audio_out + weight_v * video_out)

        output = self.final_layer_norm(text_input  + weight_v * video_out)

        return output

class MultiModalBartEncoder(BartPretrainedModel):

    def __init__(self, config: BartConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)

        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model
        self.padding_idx = config.pad_token_id
        self.max_source_position = config.max_position_embeddings
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, embed_dim, self.padding_idx)

        self.embed_positions = BartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            embed_dim
        )
        # change the forward



        self.layers = nn.ModuleList([BartEncoderLayer(config) for _ in range(config.encoder_layers)])

        self.layernorm_embedding = nn.LayerNorm(embed_dim)

        self.init_weights()
        self.gradient_checkpointing = False

        self.fusion_at_layer4 = [4]
        self.fusion_at_layer5 = [5]

        self.MAF_layer4 = MAF_acoustic(dim_model=embed_dim,
                             dropout_rate=0.2)

        self.MAF_layer5 = MAF_visual(dim_model=embed_dim,
                             dropout_rate=0.2)



    def forward(self,
            input_ids = None,
            attention_mask = None,
            acoustic_input = None,
            visual_input = None,
            head_mask = None,
            inputs_embeds = None,
            output_attentions = None,
            output_hidden_states  = None,
            return_dict = None):

            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )

            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            if input_ids is not None and inputs_embeds is not None:
                raise ValueError("You can't specify both input_ids and inputs_embeds at the same time")
            elif input_ids is not None:
                input_shape = input_ids.size()
                input_ids = input_ids.view(-1, input_shape[-1])

            if inputs_embeds is None:
                # input_ids가 텐서인지 확인
                if isinstance(input_ids, int):
                    input_ids = torch.tensor([[input_ids]])  # 2D 텐서로 변환
                elif len(input_ids.shape) == 1:  # 1D 텐서인 경우
                    input_ids = input_ids.unsqueeze(0)  # 배치 차원 추가

                print(f"Processed input_ids shape: {input_ids.shape}")

                inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

            # 시퀀스 길이 확인
            input_shape = input_ids.size() if inputs_embeds is None else inputs_embeds.size()[:-1]

            # embed_positions에 시퀀스 길이 전달
            seq_len = input_shape[1]  # 시퀀스 길이
            embed_pos = self.embed_positions(torch.arange(seq_len, device=input_ids.device).unsqueeze(0))


            hidden_states = inputs_embeds + embed_pos
            hidden_states = self.layernorm_embedding(hidden_states)
            hidden_states = F.dropout(hidden_states, p = self.dropout, training=self.training)

            if attention_mask is not None:
                attention_mask =  _expand_mask(attention_mask, inputs_embeds.dtype)

            encoder_states = () if output_hidden_states else None
            all_attentions = () if output_attentions else None

            if head_mask is not None:
                assert head_mask.size()[0] == (
                    len(self.layers)
                ), f"The head mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."

            for idx, encoder_layer in enumerate(self.layers):

                if idx in self.fusion_at_layer4:

                    hidden_states = self.MAF_layer4(text_input = hidden_states,
                                                   acoustic_context = acoustic_input
                                                   )
                if idx in self.fusion_at_layer5:
                    hidden_states = self.MAF_layer5(text_input = hidden_states,
                                                   visual_context = visual_input)

                if output_hidden_states:
                    encoder_states = encoder_states + (hidden_states,)

                dropout_probability = random.uniform(0,1)

                if self.training and (dropout_probability < self.layerdrop):
                    layer_outputs = (None, None)

                else:
                    if self.gradient_checkpointing and self.training:

                        def create_custom_forward(module):
                            def custom_forward(*inputs):
                                return module(*inputs, output_attentions)

                            return custom_forward

                        layer_outputs = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(encoder_layer),
                            hidden_states,
                            attention_mask,
                            (head_mask[idx] if head_mask is not None else None),
                        )

                    else:
                        layer_outputs = encoder_layer(
                            hidden_states,
                            attention_mask,
                            layer_head_mask = (head_mask[idx] if head_mask is not None else None),
                            output_attentions = output_attentions
                        )

                    hidden_states = layer_outputs[0]

                if output_attentions:
                    all_attentions  = all_attentions + (layer_outputs[1],)

            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)

            if not return_dict:
                return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)

            return BaseModelOutput(
                last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
            )


# %%
class MultimodalBartModel(BartPretrainedModel):
    def __init__(self, config: BartConfig):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.encoder = MultiModalBartEncoder(config, self.shared)
        self.decoder = BartDecoder(config, self.shared)

        self.init_weights()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    def get_encoder(self):
        return self.encoder


    def get_decoder(self):
        return self.decoder

    def forward(
        self,
        input_ids = None,
        attention_mask = None,
        # context_input_ids = None,
        # context_attention_mask = None,
        acoustic_input = None,
        visual_input = None,
        decoder_input_ids = None,
        decoder_attention_mask = None,
        head_mask = None,
        decoder_head_mask = None,
        cross_attn_head_mask = None,
        encoder_outputs = None,
        past_key_values = None,
        inputs_embeds = None,
        decoder_inputs_embeds = None,
        use_cache = None,
        output_attentions = None,
        output_hidden_states = None,
        return_dict = None
    ):

        if decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id

            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # print("attention mask shape 2 : ", attention_mask.shape)

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids = input_ids,
                attention_mask = attention_mask,
                # context_input_ids = context_input_ids,
                # context_attention_mask = context_attention_mask,
                acoustic_input = acoustic_input,
                visual_input = visual_input,
                head_mask = head_mask,
                inputs_embeds = inputs_embeds,
                output_attentions = output_attentions,
                output_hidden_states = output_hidden_states,
                return_dict = return_dict
            )

        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions = encoder_outputs[2] if len(encoder_outputs) > 2 else None
            )

        decoder_outputs = self.decoder(
            input_ids = decoder_input_ids,
            attention_mask = decoder_attention_mask,
            encoder_hidden_states = encoder_outputs[0],
            encoder_attention_mask = attention_mask,
            head_mask = decoder_head_mask,
            cross_attn_head_mask = cross_attn_head_mask,
            past_key_values = past_key_values,
            inputs_embeds = decoder_inputs_embeds,
            use_cache = use_cache,
            output_attentions = output_attentions,
            output_hidden_states = output_hidden_states,
            return_dict = return_dict
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions

        )

# %%
class MultimodalBartClassification(nn.Module):
    def __init__(
        self,
        input_dim: int,
        inned_dim: int,
        num_classes: int,
        pooler_dropout: float
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inned_dim)
        self.dropout = nn.Dropout(p = pooler_dropout)
        self.out_proj = nn.Linear(inned_dim, num_classes)

    def forward(self, hidden_states):
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states




class MultimodalBartForSequenceClassification(BartPretrainedModel):
    _keys_to_ignore_on_load_missing = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]

    def __init__(self, config: BartConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.model = MultimodalBartModel(config)
        self.classification_head = MultimodalBartClassification(
            config.d_model,
            config.d_model,
            2,
            config.classifier_dropout
        )
        self.model._init_weights(self.classification_head.dense)
        self.model._init_weights(self.classification_head.out_proj)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.tensor] = None,
        # context_input_ids : torch.LongTensor = None,
        # context_attention_mask : Optional[torch.tensor] = None,
        acoustic_input = None,
        visual_input = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None
    ) -> Union[Tuple, Seq2SeqSequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if labels is not None:
            use_cache = False

        if input_ids is None and inputs_embeds is not None:
            raise NotImplementedError(
                f"Passing input embeddings is currently not supported for {self.__class__.__name__}"
            )

        # print("attention mask shape 1 : ", attention_mask.shape )
        outputs = self.model(
            input_ids,
            attention_mask = attention_mask,
            # context_input_ids = context_input_ids,
            # context_attention_mask = context_attention_mask,
            acoustic_input = acoustic_input,
            visual_input = visual_input,
            decoder_input_ids = decoder_input_ids,
            decoder_attention_mask = decoder_attention_mask,
            head_mask = head_mask,
            decoder_head_mask = decoder_head_mask,
            cross_attn_head_mask = cross_attn_head_mask,
            encoder_outputs = encoder_outputs,
            inputs_embeds = inputs_embeds,
            decoder_inputs_embeds = decoder_inputs_embeds,
            use_cache = use_cache,
            output_attentions = output_attentions,
            output_hidden_states = output_hidden_states,
            return_dict = return_dict

        )

        hidden_states = outputs[0]

        eos_mask = input_ids.eq(self.config.eos_token_id).to(hidden_states.device)

        if len(torch.unique_consecutive(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have same number of <eos> tokens")

        sentence_representation = hidden_states[eos_mask, :].view(hidden_states.size(0), -1, hidden_states.size(-1))[
            :, -1, :
        ]

        logits = self.classification_head(sentence_representation)

        loss = None

        loss_fct = CrossEntropyLoss()
        # print("logits shape : ", logits.shape)
        loss = loss_fct(logits.view(-1, 2), labels.view(-1))
        # print("Loss : ", loss)
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output


        return Seq2SeqSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions = outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions
        )
