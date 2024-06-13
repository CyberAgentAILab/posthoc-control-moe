import numpy as np

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from utilis.matrix import accuracy_whole
from sklearn.metrics import f1_score

from accelerate.logging import get_logger


logger = get_logger(__name__)


def validate(
    val_loader,
    model,
    accelerator,
    epoch=0,
    test=True,
    args=None,
    datasetname=None,
    prob_mode="expert",
):
    if test and datasetname == "mnli":
        datasetname = "HANS"
    if test and datasetname == "qqp":
        datasetname = "paws"
    if test and datasetname == "fever":
        datasetname = "symm_v1"

    Acc = []
    Loss = {}
    EqPred = 0
    Targ = []
    Pred = []
    step = 0

    model.eval()
    with torch.no_grad():
        for i, (input_ids, attention_masks, segment_ids, target) in enumerate(
            val_loader
        ):
            step += 1

            output, loss, loss_dict = model(
                input_ids, target, attention_masks, segment_ids, prob_mode=prob_mode
            )
            output = accelerator.gather_for_metrics(output)
            target = accelerator.gather_for_metrics(target)
            loss_dict = accelerator.gather_for_metrics(loss_dict)

            if args.f1_eval:
                assert (
                    args.classes_num == 2
                ), "Currently, F1 supports binary classification only"
                Pred.extend(output.softmax(dim=1).argmax(dim=1).tolist())
                Targ.extend(target.tolist())

            assert len(output) == len(target)
            acc_list = accuracy_whole(
                output, target, topk=(1,), args=args, datasetname=datasetname
            )
            Acc.extend(acc_list)
            for k, v in loss_dict.items():
                if k == "AllEqual":
                    EqPred += v.sum().item()
                elif k == "RouterDist":
                    if k in Loss:
                        Loss[k].extend(v.tolist())
                    else:
                        Loss[k] = v.tolist()
                    assert len(Loss[k][0]) == args.num_experts
                elif k == "ExpertOutDist":
                    if k in Loss:
                        Loss[k] += v.mean(dim=0).cpu()
                    else:
                        Loss[k] = v.mean(dim=0).cpu()
                else:
                    if k in Loss:
                        Loss[k].append(v.mean().item())
                    else:
                        Loss[k] = [v.mean().item()]

        assert len(Acc) == len(val_loader.dataset)
        if args.f1_eval:
            epoch_f1 = f1_score(Targ, Pred) * 100.0
            epoch_score = sum(Acc) / len(Acc)
            logger.info(
                "Eval on {}: {}, F1: {:.3f}, Acc: {:.3f}".format(
                    datasetname,
                    ", ".join(
                        [
                            "{}: {:.3f}".format(k, sum(v) / len(v))
                            for k, v in Loss.items()
                            if k.endswith("Loss")
                        ]
                    ),
                    epoch_f1,
                    epoch_score,
                )
            )
        else:
            epoch_score = sum(Acc) / len(Acc)
            logger.info(
                "Eval on {}: {}, Acc: {:.3f}".format(
                    datasetname,
                    ", ".join(
                        [
                            "{}: {:.3f}".format(k, sum(v) / len(v))
                            for k, v in Loss.items()
                            if k.endswith("Loss")
                        ]
                    ),
                    epoch_score,
                )
            )
        if prob_mode == "expert":
            if "RouterDist" in Loss:
                matrix = np.array(Loss["RouterDist"])
                stats = [
                    "{:.3f}:{:.3f}".format(matrix[:, i].mean(), matrix[:, i].std())
                    for i in range(matrix.shape[1])
                ]
                logger.info("RouterStats (mean:std): {}".format(" ".join(stats)))
            if "ExpertOutDist" in Loss:
                matrix_exout = Loss["ExpertOutDist"] / step  # (expert, class)
                stats_exout = [
                    " ".join(["{:.3f}".format(p) for p in row]) for row in matrix_exout
                ]
                stats_exout = "\n".join(stats_exout)
                logger.info("ExpertOutStats:\n{}".format(stats_exout))
        if "AllEqual" in Loss:
            logger.info("AllEqual: {}".format(EqPred))

    return epoch_score
