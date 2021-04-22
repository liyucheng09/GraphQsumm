from transformers import RobertaModel, BertTokenizer, RobertaForTokenClassification,\
    BertForTokenClassification, BertModel, BertPreTrainedModel
from transformers.models.bert.modeling_bert import BertLMPredictionHead
from transformers.modeling_outputs import TokenClassifierOutput
from transformers.activations import gelu
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss


class BertSeq2seq(BertPreTrainedModel):

    def __init__(self, config):
        super(BertSeq2seq, self).__init__(config)
        
        self.bert = BertModel(config, add_pooling_layer=False)
        self.predictions = BertLMPredictionHead(config)

        self.init_weights()

    def get_output_embeddings(self):
        return self.predictions.decoder

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        seq_len = input_ids.shape[1]
        if attention_mask is not None and token_type_ids is not None:
            ones = torch.ones((1, seq_len, seq_len), dtype=torch.float32, device=self.device)
            a_mask = ones.tril()

            part_a_mask = (1-token_type_ids)-(1-attention_mask)
            ex_part_a_mask_2 = part_a_mask.unsqueeze(1).float()
            ex_part_a_mask_3 = part_a_mask.unsqueeze(2).float()
            ex_token_type_13 = token_type_ids.unsqueeze(2).float()
            a_mask = ex_part_a_mask_2*ex_part_a_mask_3 + ex_token_type_13 * a_mask
            attention_mask=a_mask

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.predictions(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            predictions = logits[:, :-1].contiguous()
            target_mask = token_type_ids[:, 1:].contiguous()==1
            active_logits = predictions.view(-1, self.config.vocab_size)
            active_labels = torch.where(
                    target_mask.view(-1), labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
            loss = loss_fct(active_logits, active_labels)

            # if attention_mask is not None:
            #     active_loss = attention_mask.view(-1) == 1
            #     active_logits = logits.view(-1, self.num_labels)
            #     active_labels = torch.where(
            #         active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
            #     )
            #     loss = loss_fct(active_logits, active_labels)
            # else:
            #     loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class RobertaSeq2Seq(RobertaForTokenClassification):

    def __init__(self, config):
        super(RobertaSeq2Seq, self).__init__(config)
        self.transform = nn.Linear(config.hidden_size, config.hidden_size)

    def get_output_embeddings(self):
        return self.classifier

    # def tie_weights(self):
    #     assert self.classifier.weight.shape==self.roberta.embeddings.word_embeddings.weight.shape, \
    #         "Word embeddings shape do not match the output linear layer."
    #     self.classifier.weight = self.roberta.embeddings.word_embeddings.weight

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )