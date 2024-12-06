from transformers import AutoModel, AutoTokenizer, AutoConfig, PreTrainedModel
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from dataclasses import dataclass
from .configuration import GECToRConfig
from typing import List, Union, Optional, Tuple
import os
import json
from .vocab import load_vocab_from_official

@dataclass
class GECToROutput:
    loss: torch.Tensor = None
    loss_d: torch.Tensor = None
    loss_labels: torch.Tensor = None
    logits_d: torch.Tensor = None
    logits_labels: torch.Tensor = None
    accuracy: torch.Tensor = None
    accuracy_d: torch.Tensor = None

@dataclass
class GECToRPredictionOutput:
    probability_labels: torch.Tensor = None
    probability_d: torch.Tensor = None
    pred_labels: List[List[str]] = None
    pred_label_ids: torch.Tensor = None
    max_error_probability: torch.Tensor = None

class GECToR(PreTrainedModel):
    config_class = GECToRConfig
    def __init__(
        self,
        config: GECToRConfig
    ):
        super().__init__(config)
        self.config = config
        print('self.config.num_labels = ' + str(self.config.num_labels))
        print('self.config.d_num_labels = ' + str(self.config.d_num_labels))
        # self.tokenizer = AutoTokenizer.from_pretrained(
        #     self.config.model_id
        # )
        if self.config.has_add_pooling_layer:
            self.bert = AutoModel.from_pretrained(
                self.config.model_id,
                # add_pooling_layer=False
            )
        else:
            self.bert = AutoModel.from_pretrained(
                self.config.model_id
            )
        # +1 is for $START token
        self.bert.resize_token_embeddings(self.bert.config.vocab_size + 1)
        self.label_proj_layer = nn.Linear(
            self.bert.config.hidden_size,
            self.config.num_labels - 1
        )  # -1 is for <PAD>
        self.d_proj_layer = nn.Linear(
            self.bert.config.hidden_size,
            self.config.d_num_labels - 1
        )
        self.dropout = nn.Dropout(self.config.p_dropout)
        self.loss_fn = CrossEntropyLoss(
            label_smoothing=self.config.label_smoothing
        )
        
        self.post_init()
        self.tune_bert(False)

    def init_weight(self) -> None:
        self._init_weights(self.label_proj_layer)
        self._init_weights(self.d_proj_layer)

    def _init_weights(self, module) -> None:
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(
                mean=0.0,
                std=self.config.initializer_range
            )
            if module.bias is not None:
                module.bias.data.zero_()
        return

    def tune_bert(self, tune=True):
        # If tune=False, only classifier layers will be tuned.
        for param in self.bert.parameters():
            param.requires_grad = tune
        return
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        # token_type_ids: Optional[torch.Tensor] = None,
        # position_ids: Optional[torch.Tensor] = None,
        # inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        d_labels: Optional[torch.Tensor] = None,
        # output_attentions: Optional[bool] = None,
        # output_hidden_states: Optional[bool] = None,
        # return_dict: Optional[bool] = None,
        word_masks: Optional[torch.Tensor] = None,
    ) -> GECToROutput:
        bert_logits = self.bert(
            input_ids,
            attention_mask=attention_mask,
            # token_type_ids=token_type_ids,
            # position_ids=position_ids,
            # inputs_embeds=inputs_embeds,
            # output_attentions=output_attentions,
            # output_hidden_states=output_hidden_states,
            # return_dict=return_dict,
        ).last_hidden_state
        # print(bert_logits)
        logits_d = self.d_proj_layer(bert_logits)
        # print(logits_d)
        logits_labels = self.label_proj_layer(self.dropout(bert_logits))
        loss_d, loss_labels, loss = None, None, None
        accuracy, accuracy_d = None, None
        if d_labels is not None and labels is not None:
            pad_id = self.config.label2id[self.config.label_pad_token]
            # -100 is the default ignore_idx of CrossEntropyLoss
            labels[labels == pad_id] = -100
            d_labels[labels == -100] = -100
            loss_d = self.loss_fn(
                logits_d.view(-1, self.config.d_num_labels - 1),  # -1 for <PAD>
                d_labels.view(-1)
            )
            loss_labels = self.loss_fn(
                logits_labels.view(-1, self.config.num_labels - 1),
                labels.view(-1)
            )
            loss = loss_d + loss_labels

            pred_labels = torch.argmax(logits_labels, dim=-1)
            accuracy = torch.sum(
                (labels == pred_labels) * word_masks
            ) / torch.sum(word_masks)
            pred_d = torch.argmax(logits_d, dim=-1)
            accuracy_d = torch.sum(
                (d_labels == pred_d) * word_masks
            ) / torch.sum(word_masks)

        # print('loss='+str(loss))
        # print('loss_d='+str(loss_d))
        # print('loss_labels='+str(loss_labels))
        # print('logits_d='+str(logits_d))
        # print('logits_labels='+str(logits_labels))
        # print('accuracy='+str(accuracy))
        # print('accuracy_d='+str(accuracy_d))
        return GECToROutput(
            loss=loss,
            loss_d=loss_d,
            loss_labels=loss_labels,
            logits_d=logits_d,
            logits_labels=logits_labels,
            accuracy=accuracy,
            accuracy_d=accuracy_d
        )

    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        word_masks: torch.Tensor,
        keep_confidence: float=0,
        min_error_prob: float=0
    ): 
        with torch.no_grad():
            outputs = self.forward(
                input_ids,
                attention_mask
            )
            probability_labels = F.softmax(outputs.logits_labels, dim=-1)
            probability_d = F.softmax(outputs.logits_d, dim=-1)

            # Get actual labels considering inference parameters.
            keep_index = self.config.label2id[self.config.keep_label]
            probability_labels[:, :, keep_index] += keep_confidence
            incor_idx = self.config.d_label2id[self.config.incorrect_label]
            probability_d = probability_d[:, :, incor_idx]
            max_error_probability = torch.max(probability_d * word_masks, dim=-1)[0]
            probability_labels[max_error_probability < min_error_prob, :, keep_index] \
                                                                        = float('inf')
            pred_label_ids = torch.argmax(probability_labels, dim=-1)

            def convert_ids_to_labels(ids, id2label):
                labels = []
                for id in ids.tolist():
                    labels.append(id2label[id])
                return labels

            pred_labels = []
            for ids in pred_label_ids:
                labels = convert_ids_to_labels(
                    ids,
                    self.config.id2label
                )
                pred_labels.append(labels)

        return GECToRPredictionOutput(
            probability_labels=probability_labels,
            probability_d=probability_d,
            pred_labels=pred_labels,
            pred_label_ids=pred_label_ids,
            max_error_probability=max_error_probability
        )
    
    @classmethod
    def from_official_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        special_tokens_fix: int,  # 0 or 1
        transformer_model: str,
        vocab_path: str,
        p_dropout: float=0.0,
        max_length=80,
        label_smoothing=0.0,
    ) -> "GECToR":
        '''Load official weights.
        
        Args:
            pretrained_model_name_or_path (str): Path to the official weights.
                E.g. "bert_0_gectorv2.th".
            special_tokens_fix (int): If 0, is assume that the embedding layer does not extended to add $START.
                E.g. 0 if the model name is "bert_0_gectorv2.th", or 1 if the model name is "roberta_1_gectorv2".
            transformer_model (str): A model id of the huggingface.
                E.g. "bert-base-cased".
            vocab_path (str): Path to the official vocaburaly directory.
                E.g. "data/output_vocabulary".
            p_drouput (float): The probability for the dropout layer. Only for training.
            max_length (int): If the number of subwords is longer than this, it will be truncated.
            label_smoothing (float): The epsilon for the label smoothing in CrossEntropyLoss. Only for training.

        Returns: 
            GECToR: The instance of GECToR.
        '''
        def has_add_pooling_layer(model_id):
            for m in ['xlnet', 'deberta']:
                if m in model_id:
                    return False
            return True
    
        label2id, d_label2id = load_vocab_from_official(vocab_path)
        # print('label2id :' + str(label2id))
        # The order of labels is
        #   Ours    : [OOV, KEEP, DELETE, ..., PADDING]
        #   Official: [KEEP, DELETE, ..., OOV, PADDING]
        # The following is align the ours to officila one.

        # for l in label2id.keys():S
        #     label2id[l] = label2id[l]
        # label2id['<OOV>'] = len(label2id) - 2
        # label2id['<PAD>'] = len(label2id) - 1

        config = GECToRConfig(
            model_id=transformer_model,
            label2id=label2id,
            id2label={v: k for k, v in label2id.items()},
            d_label2id=d_label2id,
            p_dropout=p_dropout,
            max_length=max_length,
            label_smoothing=label_smoothing,
            has_add_pooling_layer=has_add_pooling_layer(transformer_model)
        )
        
        model = GECToR(config=config)
        
        official_state_dict = torch.load(pretrained_model_name_or_path)['model']
        # Resolve the official parameter names.
        new_state_dict = dict()
        for k, v in official_state_dict.items():
            if 'pool' in k:
                continue
            if 'position_id' in k:
                continue
            k = k.replace('text_field_embedder.token_embedder_bert.bert_model.', 'bert.') \
                .replace('tag_labels_projection_layer._module', 'label_proj_layer') \
                .replace('tag_detect_projection_layer._module', 'd_proj_layer')
            # if not special_tokens_fix and 'word_embedding' in k:
            #     # The official model does not extend the input dimension 
            #     #   of the embedding layer when special_tokens_fix is 0 
            #     #   (i.e. does not add $START token).
            #     # In contrast, our model always extends it, 
            #     #   so we add a dummy row to the embedding layer to match the dimensions.
            #     v = torch.cat([v, torch.zeros_like(v[0:1])], dim=0)
            # if 'd_proj_layer' in k:
            #     # The shape of the official layer is (d_model, 4).
            #     # The output dimension 4 corresponds to $CORRECT, $INCORRECT, @@UNKNOWN@@, @@PADDING@@ in this order.
            #     # In contrast, the shape of our layer is (d_model, 2), which means using only $CORRECT and $INCORRECT.
            #     # Thus we use only first and second rows.
            #     v = v[:2]
            # if 'label_proj_layer' in k:
            #     # Similar to detection projection layer,
            #     #   we do not use @@PADDING@@ as a prediction labels, so we remove its row from the label projection layer.
            #     v = v[:-1]
            new_state_dict[k] = v
        model.load_state_dict(new_state_dict)
        return model
        