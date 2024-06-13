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


def router_topk_mask(x, k):
    """x: shape(batch_size, n), k: int"""
    if k > x.size(1):  # for the batch at the end
        k = x.size(1)
    topk_vals, _ = torch.topk(x, k, dim=1)
    mask = x.lt(topk_vals[:, [-1]]).float()
    return mask


def is_same_elements(row):
    return (row == row[0]).all().unsqueeze(0)


class Model(nn.Module):
    def __init__(self, pretrained_path, bert_config, args, num_labels):
        super(Model, self).__init__()
        self.bert_config = bert_config
        self.bert = AutoModel.from_pretrained(pretrained_path)

        self.experts = nn.ModuleList(
            [BertPredictionHeadTransform(bert_config) for _ in range(args.num_experts)]
        )
        self.router = nn.Sequential(
            BertPredictionHeadTransform(bert_config),
            nn.Linear(bert_config.hidden_size, args.num_experts, bias=False),
        )
        if args.expert_layer_start < bert_config.num_hidden_layers:
            self.bert.encoder.layer = self.bert.encoder.layer[: args.expert_layer_start]
        self.num_experts = args.num_experts
        self.num_topk_mask = args.num_topk_mask
        self.router_loss = args.router_loss
        self.router_tau = args.router_tau
        self.router_dist_min = args.router_dist_min
        self.router_dist_max = args.router_dist_max

        self.classifier = nn.ModuleList(
            [
                nn.Linear(bert_config.hidden_size, num_labels)
                for _ in range(args.num_experts)
            ]
        )
        for c in self.classifier:
            nn.init.constant_(c.bias, 0.0)
        self.criterion = nn.NLLLoss()

    def forward(
        self,
        input_ids,
        target,
        attention_mask=None,
        token_type_ids=None,
        prob_mode="expert",
    ):
        loss_dict = {}
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True,
        )

        router_dist = self.router(outputs.last_hidden_state[:, 0])  # (batch, expert)
        if self.training:
            router_dist = (router_dist * self.router_tau).softmax(dim=-1)
        else:
            router_dist = router_dist.softmax(dim=-1)
        if self.router_dist_min > 0 and self.training:
            router_dist = router_dist + self.router_dist_min
        if self.router_dist_max > 0 and self.training:
            router_dist = torch.clamp(router_dist, max=self.router_dist_max)
        loss_dict["RouterDist"] = router_dist.detach().clone()
        router_dist_sim = torch.matmul(router_dist, router_dist.t())  # (batch, batch)
        router_sim_mask = (
            torch.eye(router_dist.size(0)).logical_not().float().to(router_dist.device)
        )  # (batch, batch)
        router_dist_sim = router_dist_sim * router_sim_mask
        if self.num_topk_mask > 0:
            topk_mask = router_topk_mask(router_dist_sim, self.num_topk_mask)
            router_dist_sim = router_dist_sim * topk_mask
        router_loss = torch.norm(router_dist_sim, p="fro") / torch.norm(
            router_sim_mask, p="fro"
        )
        loss_dict["RouterLoss"] = router_loss.detach().clone()

        prob_expert = 0
        prob_uniform = 0
        prob_argmin = 0
        prob_all = []
        for i, (expert_i, classifier_i) in enumerate(
            zip(self.experts, self.classifier)
        ):
            output_i = expert_i(outputs.last_hidden_state[:, 0])
            output_i = classifier_i(output_i).softmax(dim=-1)  # (batch, class)
            prob_expert = prob_expert + output_i * router_dist[:, i].unsqueeze(-1)
            if not self.training:
                prob_uniform = prob_uniform + output_i / self.num_experts
                if i == 0:
                    prob_argmin = output_i
                else:
                    prob_argmin = torch.min(prob_argmin, output_i)
            prob_all.append(output_i.mean(dim=0).detach().clone())
        assert len(prob_all) == self.num_experts
        loss_dict["ExpertOutDist"] = torch.stack(prob_all, dim=0).unsqueeze(
            0
        )  # (1, expert, class)
        loss = self.criterion(torch.log(prob_expert + 1e-8), target)
        loss_dict["ClassifierLoss"] = loss.detach().clone()
        if self.training:
            if self.router_loss > 0:
                loss = loss + router_loss * self.router_loss

        if prob_mode == "expert":
            prob = prob_expert
        elif prob_mode == "uniform":
            prob = prob_uniform
        elif prob_mode == "argmin":
            prob = prob_argmin
            all_equal = torch.cat([is_same_elements(row) for row in prob], dim=0).sum()
            loss_dict["AllEqual"] = all_equal
        else:
            raise KeyError("{} is not supported".format(prob_mode))

        return prob, loss, loss_dict
