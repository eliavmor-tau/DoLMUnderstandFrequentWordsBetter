from config.config import *
import torch.tensor as Tensor
from torch.nn import CrossEntropyLoss
from transformers import BertForMaskedLM, RobertaForMaskedLM, XLNetLMHeadModel, AlbertForMaskedLM
from allennlp.models.model import Model
from allennlp.data import Vocabulary


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
        outputs = self.roberta(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)
        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)
        # Mask prediction
        if all_indices_mask is not None:
            prediction_scores *= all_indices_mask

        masked_lm_loss = None
        if labels is not None:
            # Ignore unmasked words prediction (-100)
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        return {"loss": masked_lm_loss, "logits": prediction_scores}


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

    def forward(self, input_ids: Tensor, token_type_ids: Tensor, all_indices_mask: Tensor = None, labels: Tensor = None) \
            -> Dict[str, Tensor]:
        if self.is_multi_choice and all_indices_mask is not None:
            model_predictions = self._model(input_ids=input_ids, token_type_ids=token_type_ids,
                                            all_indices_mask=all_indices_mask,
                                            labels=labels)
        else:
            model_predictions = self._model(input_ids=input_ids, token_type_ids=token_type_ids,
                                            labels=labels)

        output = dict()
        if "loss" in model_predictions:
            output["loss"] = model_predictions["loss"]
        output["logits"] = model_predictions["logits"]
        return output
