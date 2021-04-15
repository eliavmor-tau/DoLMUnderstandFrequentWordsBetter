from config.config import *
import torch.tensor as Tensor
from torch.nn import CrossEntropyLoss
from transformers import BertForMaskedLM, RobertaForMaskedLM, XLNetLMHeadModel, AlbertForMaskedLM
from allennlp.models.model import Model
from allennlp.data import Vocabulary
from torch.nn.functional import softmax

from typing import Dict, Optional, List, Any

from transformers import RobertaForMaskedLM, XLNetLMHeadModel, BertForMaskedLM, AlbertForMaskedLM, RobertaModel
import re, json, os
import numpy as np
import torch
from torch.nn import CrossEntropyLoss

from allennlp.common.params import Params
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.nn import RegularizerApplicator, util
from allennlp.training.metrics import CategoricalAccuracy
import torch
from torch.nn.modules.linear import Linear
from torch.nn.functional import binary_cross_entropy_with_logits
from allennlp.common.params import Params
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.nn import RegularizerApplicator, util
from allennlp.training.metrics import CategoricalAccuracy



class BertMultiChoiceMLM(BertForMaskedLM):
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        masked_lm_labels=None,
        position_ids=None,
        head_mask=None,
        all_indices_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)
        # Mask prediction
        if all_indices_mask is not None:
            prediction_scores *= all_indices_mask

        masked_lm_loss = None
        if labels is not None:
            # Ignore unmasked words prediction (-100)
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        return {"loss": masked_lm_loss, "logits": prediction_scores}


# class RobertaForMultiChoiceMaskedLM(RobertaForMaskedLM):
#     def forward(self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None, position_ids=None,
#                 head_mask=None, all_masked_index_ids = None, label = None):
#         outputs = self.roberta(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
#                             attention_mask=attention_mask, head_mask=head_mask)
#         sequence_output = outputs[0]
#         prediction_scores = self.lm_head(sequence_output)
#
#         outputs = (prediction_scores,) + outputs[2:]  # Add hidden states and attention if they are here
#
#         if masked_lm_labels is not None:
#             loss_fct = CrossEntropyLoss(ignore_index=-1)
#             # choosing prediction with all_masked_index_ids
#             masked_lm_loss = 0
#             for i, choices in enumerate(all_masked_index_ids):
#                 masked_lm_loss += \
#                     loss_fct(prediction_scores[i, choices[0][0][0], [c[0][1] for c in choices]].unsqueeze(0), label[i].unsqueeze(0))
#
#             # masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
#             outputs = (masked_lm_loss,) + outputs
#
#         return outputs


class RobertaMultiChoiceMLM(RobertaForMaskedLM):
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        masked_lm_labels=None,
        position_ids=None,
        head_mask=None,
        all_indices_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        batch_size, max_length = input_ids.size()
        outputs = self.roberta(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)
        sequence_output = outputs[0]
        logits = self.lm_head(sequence_output)
        # Mask prediction
        output = {"logits": logits, "loss": -1}

        if all_indices_mask is not None and labels is not None:
            # Ignore unmasked words prediction (-100)
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            logits = logits.transpose(1, 2)
            masked_lm_loss = loss_fct(logits, labels)
            output["loss"] = masked_lm_loss / float(batch_size)
        return output


class TransformerMaskedLanguageModel(Model):

    def __init__(self, vocab: Vocabulary, model_name: str="bert-base", multi_choice: bool=False):
        super().__init__(vocab)
        self._model = None
        self._loss = CrossEntropyLoss()
        self.is_multi_choice = multi_choice

        if model_name.startswith('bert'):
            if self.is_multi_choice:
                self._model = BertMultiChoiceMLM.from_pretrained(model_name)
            else:
                self._model = BertForMaskedLM.from_pretrained(model_name)
        elif 'roberta' in model_name:
            if self.is_multi_choice:
                self._model = RobertaMultiChoiceMLM.from_pretrained(model_name)
            else:
                self._model = RobertaForMaskedLM.from_pretrained(model_name)

        elif 'albert' in model_name:
            self._model = AlbertForMaskedLM.from_pretrained(model_name)
        elif 'xlnet' in model_name:
            self._model = XLNetLMHeadModel.from_pretrained(model_name)
        else:
            raise("Riquiered model is not supported.")

    def forward(self, input_ids: Tensor, token_type_ids: Tensor, all_indices_mask: Tensor = None, labels: Tensor = None,
                attention_mask: Tensor=None) -> Dict[str, Tensor]:
        if self.is_multi_choice and all_indices_mask is not None:
            model_predictions = self._model(input_ids=input_ids, token_type_ids=token_type_ids,
                                            all_indices_mask=all_indices_mask,
                                            labels=labels, attention_mask=attention_mask)
        else:
            model_predictions = self._model(input_ids=input_ids, token_type_ids=token_type_ids,
                                            labels=labels, attention_mask=attention_mask)

        output = dict()
        if "loss" in model_predictions:
            output["loss"] = model_predictions["loss"]
        output["logits"] = model_predictions["logits"]
        return output


class RobertaYesNoQA(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 pretrained_model: str = None,
                 requires_grad: bool = True,
                 layer_freeze_regexes: List[str] = None,
                 regularizer: Optional[RegularizerApplicator] = None) -> None:

        super().__init__(vocab, regularizer)

        self._pretrained_model = pretrained_model
        self._padding_value = 1  # The index of the RoBERTa padding token
        self._transformer_model = RobertaModel.from_pretrained(pretrained_model)
        self._dropout = torch.nn.Dropout(self._transformer_model.config.hidden_dropout_prob)

        for name, param in self._transformer_model.named_parameters():
            if layer_freeze_regexes and requires_grad:
                grad = not any([bool(re.search(r, name)) for r in layer_freeze_regexes])
            else:
                grad = requires_grad
            if grad:
                param.requires_grad = True
            else:
                param.requires_grad = False

        transformer_config = self._transformer_model.config
        transformer_config.num_labels = 1
        self._output_dim = self._transformer_model.config.hidden_size

        # unifing all model classification layer
        self._classifier = Linear(self._output_dim, 1)
        self._classifier.weight.data.normal_(mean=0.0, std=0.02)
        self._classifier.bias.data.zero_()

        self._accuracy = CategoricalAccuracy()
        self._loss = torch.nn.CrossEntropyLoss()

    def forward(self,
                question: Dict[str, torch.LongTensor],
                segment_ids: torch.LongTensor = None,
                label: torch.LongTensor = None,
                metadata: List[Dict[str, Any]] = None) -> torch.Tensor:

        input_ids = question['tokens']
        batch_size = input_ids.size(0)
        num_choices = input_ids.size(1)

        question_mask = (input_ids != self._padding_value).long()

        transformer_outputs, pooled_output = self._transformer_model(input_ids=util.combine_initial_dims(input_ids),
                                                                     attention_mask=util.combine_initial_dims(question_mask))
        cls_output = self._dropout(pooled_output)
        label_logits = self._classifier(cls_output)
        label_logits = label_logits.view(-1, num_choices)

        output_dict = {}
        output_dict['label_logits'] = label_logits
        output_dict['label_probs'] = torch.nn.functional.softmax(label_logits, dim=1)
        output_dict['answer_index'] = label_logits.argmax(1)

        if label is not None:
            loss = self._loss(label_logits, label)
            self._accuracy(label_logits, label)
            output_dict["loss"] = loss

        return output_dict
