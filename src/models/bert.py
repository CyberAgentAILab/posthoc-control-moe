import torch
import torch.nn as nn

from transformers import AutoModel


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.transform_act_fn = nn.ReLU()
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class Model(nn.Module):
    def __init__(self, pretrained_path, bert_config, args, num_labels):
        super(Model, self).__init__()
        self.bert_config = bert_config
        self.bert = AutoModel.from_pretrained(pretrained_path)
        if args.expert_layer_start < bert_config.num_hidden_layers:
            self.bert.encoder.layer = self.bert.encoder.layer[: args.expert_layer_start]
        self.phead = BertPredictionHeadTransform(bert_config)
        self.classifier = nn.Linear(bert_config.hidden_size, num_labels)
        self.criterion = nn.CrossEntropyLoss()
        nn.init.constant_(self.classifier.bias, 0.0)

    def forward(
        self,
        input_ids,
        target,
        attention_mask=None,
        token_type_ids=None,
        prob_mode=None,
    ):
        loss_dict = {}
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True,
        )
        pooled_output = self.phead(outputs.last_hidden_state[:, 0])
        logits = self.classifier(pooled_output)
        loss = self.criterion(logits, target)
        loss_dict["Loss"] = loss.detach().clone()

        return logits, loss, loss_dict
