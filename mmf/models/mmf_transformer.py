# Copyright (c) Facebook, Inc. and its affiliates.

from copy import deepcopy
from typing import Any, Dict, Type

import torch
from mmf.common.registry import registry
from mmf.common.typings import DictConfig
from mmf.models.transformers.base import (
    BaseTransformer,
    BaseTransformerConfigType,
    BaseTransformerInput,
)
from mmf.modules.encoders import MultiModalEncoderBase
from torch import Tensor, nn
from transformers.modeling_bert import BertPooler, BertPredictionHeadTransform


class ImageEncoder(MultiModalEncoderBase):
    """Extends the MultiModalEncoderBase class which builds the encoder based on
    the config parameters. We can set the type of image encoder(resnet50, resnet152,
    resnext50 etc) and other parameters like num of features, type of pooling etc.
    """

    def __init__(self, config: DictConfig):
        super().__init__(config)

    def build(self):
        self.encoder = self._build_modal_encoder(self.config.image_encoder)

    def forward(self, x: Tensor) -> Tensor:
        return self.encoder(x)


class MMFTransformerEmbeddings(nn.Module):
    """Embedding class that can take any number of image or text modalities, each can
    have their input id, position id and segment id. We generate embeddings of
    dimension config.hidden_size for each and then first add the three embeddings
    for each modality to have a modality specific embedding. We then concat the
    modality specific embeddings to have a joint embedding.
    """

    def __init__(
        self,
        model_config: Dict[str, Any],
        transformer_config: Dict[str, Any],
        transformer: Type[nn.Module],
    ):
        super().__init__()
        self.model_config = model_config
        self.transformer_config = transformer_config
        self._build_layers()
        self._init_weights(transformer)

    def _build_layers(self):
        for modality in self.model_config.modalities:
            if "text" in modality:
                setattr(
                    self,
                    modality + "_embedding",
                    nn.Embedding(
                        self.transformer_config.vocab_size,
                        self.transformer_config.hidden_size,
                        padding_idx=self.transformer_config.pad_token_id,
                    ),
                )
            elif "image" in modality:
                setattr(
                    self,
                    modality + "_embedding",
                    nn.Sequential(
                        nn.Linear(
                            self.model_config.visual_embedding_dim,
                            self.transformer_config.hidden_size,
                        ),
                        torch.nn.LayerNorm(
                            self.transformer_config.hidden_size, eps=1e-12
                        ),
                    ),
                )
            # Set the position embeddings
            setattr(
                self,
                modality + "_pos_embedding",
                nn.Embedding(
                    self.transformer_config.max_position_embeddings,
                    self.transformer_config.hidden_size,
                ),
            )
            # Layer norm
            setattr(
                self,
                modality + "_layer_norm",
                torch.nn.LayerNorm(self.transformer_config.hidden_size, eps=1e-12),
            )
            setattr(
                self,
                modality + "_dropout",
                nn.Dropout(self.transformer_config.hidden_dropout_prob),
            )

        self.token_type_embeddings = nn.Embedding(
            len(self.model_config.modalities), self.transformer_config.hidden_size
        )

    def _init_weights(self, transformer: Type[nn.Module]):
        for modality in self.model_config.modalities:
            if "text" in modality:
                setattr(
                    self,
                    modality + "_embedding",
                    transformer.embeddings.word_embeddings,
                )
            pos_embedding_layer = getattr(self, modality + "_pos_embedding")
            pos_embedding_layer.weight = nn.Parameter(
                deepcopy(transformer.embeddings.position_embeddings.weight.data),
                requires_grad=True,
            )
            setattr(self, modality + "_layer_norm", transformer.embeddings.LayerNorm)

        # Token Type Embeddings
        if hasattr(transformer.embeddings, "token_type_embeddings"):
            token_vocab_size = self.transformer_config.type_vocab_size
            self.token_type_embeddings.weight.data[:token_vocab_size].copy_(
                transformer.embeddings.token_type_embeddings.weight
            )
            for idx in range(token_vocab_size, len(self.model_config.modalities)):
                self.token_type_embeddings.weight.data[idx].copy_(
                    transformer.embeddings.token_type_embeddings.weight.data.mean(dim=0)
                )

    def forward(
        self,
        input_ids: Dict[str, Tensor],
        position_ids: Dict[str, Tensor],
        segment_ids: Dict[str, Tensor],
    ) -> Tensor:
        list_embeddings = []
        for idx, modality in enumerate(self.model_config.modalities):
            device = input_ids[modality].device
            total_embedding = getattr(self, modality + "_embedding")(
                input_ids[modality]
            )
            if modality not in position_ids:
                position_ids[modality] = (
                    torch.arange(
                        0, input_ids[modality].size(1), dtype=torch.long, device=device
                    )
                    .unsqueeze(0)
                    .expand(input_ids[modality].size()[:2])
                )
            total_embedding += getattr(self, modality + "_pos_embedding")(
                position_ids[modality]
            )

            if modality not in segment_ids:
                segment_ids[modality] = torch.zeros(
                    input_ids[modality].size()[:2], dtype=torch.long, device=device
                ).fill_(idx)
            total_embedding += self.token_type_embeddings(segment_ids[modality])

            layer_norm_layer = getattr(self, modality + "_layer_norm")
            dropout_layer = getattr(self, modality + "_dropout")
            list_embeddings.append(dropout_layer(layer_norm_layer(total_embedding)))

        return torch.cat(list_embeddings, dim=1)


@registry.register_model("mmf_transformer")
class MMFTransformer(BaseTransformer):
    def __init__(self, config: BaseTransformerConfigType):
        super().__init__(config)

    @classmethod
    def config_path(cls) -> str:
        return "configs/models/mmf_transformer/defaults.yaml"

    def build_encoders(self):
        self.image_encoder = ImageEncoder(self.config)
        if getattr(self.config, "freeze_encoder", False):
            for param in self.image_encoder.parameters():
                param.requires_grad = False

    def build_embeddings(self):
        """Initialize the embedding class we will use for multiple
        modalities (here just text and image). For the text embeeddings we will use the
        pretrained weights from the trasnformer model rather than training from scratch.
        """
        self.embeddings = MMFTransformerEmbeddings(
            self.config, self.transformer_config, self.transformer
        )

    def build_heads(self):
        """Initialize the classifier head. It takes the output of the
        transformer encoder and passes it through a pooler (we use the pooler from BERT
        model), then dropout, BertPredictionHeadTransform (which is a liner layer,
        followed by activation and layer norm) and lastly a linear layer projecting the
        hidden output to classification labels.
        """
        self.classifier = nn.Sequential(
            BertPooler(self.transformer_config),
            nn.Dropout(self.transformer_config.hidden_dropout_prob),
            BertPredictionHeadTransform(self.transformer_config),
            nn.Linear(self.transformer_config.hidden_size, self.config.num_labels),
        )

    def preprocess_sample(self, sample_list: Dict[str, Any]) -> BaseTransformerInput:
        """Preprocess the sample list elements and form a BaseTransformerInput
        type object. This object standardizes how we represent multiple modalities.
        Check the definition of this dataclass in BaseTransformer.
        """

        # Input IDs (or text tokens/image features)
        input_ids: Dict[str, Tensor] = {}
        for idx, modality in enumerate(self.config.modalities):
            if "text" in modality:
                if sample_list.input_ids.dim() > 2:
                    input_ids[modality] = sample_list.input_ids[:, idx]
                else:
                    input_ids[modality] = sample_list.input_ids

        if "image" in sample_list:
            input_ids["image"] = self.image_encoder(sample_list.image)
        elif "image_feature_0" in sample_list:
            input_ids["image"] = sample_list.image_feature_0

        # Position IDs
        position_ids: Dict[str, Tensor] = {}

        # Segment IDs
        segment_ids: Dict[str, Tensor] = {}
        for idx, modality in enumerate(self.config.modalities):
            if "text" in modality:
                if sample_list.segment_ids.dim() > 2:
                    segment_ids[modality] = sample_list.segment_ids[:, idx]
                else:
                    segment_ids[modality] = sample_list.segment_ids

        # Masks
        masks: Dict[str, Tensor] = {}
        for idx, modality in enumerate(self.config.modalities):
            if "text" in modality:
                if sample_list.input_mask.dim() > 2:
                    masks[modality] = sample_list.input_mask[:, idx]
                else:
                    masks[modality] = sample_list.input_mask
        if "image_mask" in sample_list:
            masks["image"] = sample_list.image_mask
        else:
            masks["image"] = torch.ones(
                input_ids["image"].size()[:-1],
                dtype=torch.long,
                device=input_ids["image"].device,
            )

        return BaseTransformerInput(input_ids, position_ids, segment_ids, masks)

    def forward(self, sample_list: Dict[str, Any]) -> Dict[str, Tensor]:
        # Sample preprocess
        output = self.preprocess_sample(sample_list)

        # Transformer Input Embeddings
        embedding_output = self.embeddings(
            input_ids=output.input_ids,
            position_ids=output.position_ids,
            segment_ids=output.segment_ids,
        )

        # Transformer Attention mask
        # concat the attention masks for all modalities
        masks = []
        for modality in self.config.modalities:
            masks.append(output.masks[modality])
        attention_mask = torch.cat(masks, dim=-1)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # Transformer Encoder
        encoded_layers = self.transformer.encoder(
            embedding_output,  # combined embedding
            extended_attention_mask,  # combined attention mask
            [None] * len(self.transformer.encoder.layer),  # head masks
        )

        # Transformer Heads
        head_output = self.classifier(encoded_layers[0])

        # Postprocess outputs
        return self.postprocess_output(head_output)

    def postprocess_output(self, output: Tensor) -> Dict[str, Tensor]:
        """Postprocess the output from the classifier head and reshape it.
        This will be used to calculate losses and metrics in mmf.
        """
        output_dict = {}
        output_dict["scores"] = output.contiguous().view(-1, self.config.num_labels)
        return output_dict
