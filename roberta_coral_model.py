import numpy as np
import torch
from coral_pytorch.dataset import levels_from_labelbatch
from coral_pytorch.layers import CoralLayer
from coral_pytorch.losses import coral_loss
from transformers import BertPreTrainedModel, RobertaConfig, RobertaModel
from transformers.models.roberta.modeling_roberta import RobertaClassificationHead


class RobertaCoralForSequenceClassification(BertPreTrainedModel):
    config_class = RobertaConfig
    base_model_prefix = "roberta"

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config)
        self.classifier = RobertaClassificationHead(config)
        self.classifier.out_proj = CoralLayer(size_in=768, num_classes=self.num_labels)

        self.init_weights()

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
    ):
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        outputs = (logits,)
        if labels is not None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            labels = np.argmax(labels.cpu(), axis=-1)
            levels = levels_from_labelbatch(labels, num_classes=self.num_labels).to(
                device
            )
            loss = coral_loss(logits, levels)
            outputs = (loss,) + outputs

        return outputs  # (loss), logits
