import numpy as np
from utilis.matrix import accuracy_whole
from accelerate.logging import get_logger


logger = get_logger(__name__)


def train(train_loader, model, optimizer, epoch, args, lr_scheduler, accelerator):
    logger.info("Epoch: {}".format(epoch))

    Acc = []
    Loss = {}
    step = 0

    model.train()

    for i, (input_ids, attention_masks, segment_ids, target) in enumerate(train_loader):
        step += 1

        output, loss, loss_dict = model(
            input_ids, target, attention_masks, segment_ids, prob_mode="expert"
        )
        output = accelerator.gather_for_metrics(output)
        target = accelerator.gather_for_metrics(target)
        loss_dict = accelerator.gather_for_metrics(loss_dict)

        assert len(output) == len(target)
        acc_list = accuracy_whole(output, target, topk=(1,))
        Acc.extend(acc_list)
        for k, v in loss_dict.items():
            if k == "RouterDist":
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

        optimizer.zero_grad()
        accelerator.backward(loss)
        if args.max_grad_norm > 0 and accelerator.sync_gradients:
            accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        lr_scheduler.step()

    assert len(Acc) == len(train_loader.dataset)
    logger.info(
        "Train: {}, Acc: {:.3f}".format(
            ", ".join(
                [
                    "{}: {:.3f}".format(k, sum(v) / len(v))
                    for k, v in Loss.items()
                    if k.endswith("Loss")
                ]
            ),
            sum(Acc) / len(Acc),
        )
    )
    if "RouterDist" in Loss:
        matrix = np.array(Loss["RouterDist"])
        stats = [
            "{:.3f}:{:.3f}".format(matrix[:, i].mean(), matrix[:, i].std())
            for i in range(matrix.shape[1])
        ]
        logger.info("RouterStats (mean:std): {}".format(" ".join(stats)))
    if "ExpertOutDist" in Loss:
        matrix = Loss["ExpertOutDist"] / step  # (expert, class)
        stats = [" ".join(["{:.3f}".format(p) for p in row]) for row in matrix]
        stats = "\n".join(stats)
        logger.info("ExpertOutStats:\n{}".format(stats))
