import sys
import os
import random
import numpy as np
import logging

import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
from torch.optim import AdamW
import torch.utils.data

from utilis.load_data import load_mnli, load_hans, load_fever, load_qqp_paws
from utilis.dataset import PairDatasets
from utilis.dataset import Collate_function

from training.train import train
from training.validate import validate
from utilis.saving import save_checkpoint
from ops.config import parser

from transformers import AutoConfig, AutoTokenizer
from transformers import get_linear_schedule_with_warmup

from accelerate.logging import get_logger
from accelerate import Accelerator, DistributedDataParallelKwargs


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.DEBUG,
)
logger = get_logger(__name__)

ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def set_random_seed(seed):
    """Set random seed for reproducability."""
    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        cudnn.deterministic = True


def main():
    args = parser.parse_args()

    args.best_model_name = os.path.join(args.save_dir, args.best_model_name)

    if args.dataset == "qqp":
        args.classes_num = 2
    else:
        args.classes_num = 3

    set_random_seed(args.seed)

    ngpus_per_node = torch.cuda.device_count()
    args.gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logging.info(" ".join(sys.argv))
    logger.info(args)

    main_worker(ngpus_per_node, args)


def main_worker(ngpus_per_node, args):
    if args.model == "bert_mos":
        from models.bert_mos import Model
    elif args.model == "bert_mos_stats":
        from models.bert_mos_stats import Model
    else:
        raise KeyError("{} is not supported".format(args.model))

    logger.info("Use {} GPUs for training".format(ngpus_per_node))

    bert_config = AutoConfig.from_pretrained(args.pretrained_path)
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_path)
    model = Model(args.pretrained_path, bert_config, args, args.classes_num)

    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    else:
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.01,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)

    # optionally resume from a checkpoint
    if args.resume:
        args.resume = os.path.join(args.save_dir, args.resume)
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                checkpoint = torch.load(args.resume, map_location=args.gpu)
            model.load_state_dict(checkpoint["state_dict"])
            logger.info(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, args.start_epoch
                )
            )
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    if args.dataset == "mnli":
        train_examples = load_mnli(mode="train")
        val_examples = load_mnli(mode="match_dev")
        test_examples = load_hans(shuffle=args.shuffle_hans)

    elif args.dataset == "fever":
        train_examples = load_fever(mode="train")
        val_examples = load_fever(mode="dev")
        test_examples = load_fever(mode="symmv1_generated")

    elif args.dataset == "qqp":
        train_examples = load_qqp_paws(mode="qqp_train")
        val_examples = load_qqp_paws(mode="qqp_dev")
        test_examples = load_qqp_paws(mode="paws_devtest")

    if args.debug:
        train_examples = train_examples[:2000]
    train_dataset = PairDatasets(train_examples, tokenizer, args)
    val_dataset = PairDatasets(val_examples, tokenizer, args)
    test_dataset = PairDatasets(test_examples, tokenizer, args)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=Collate_function(),
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=Collate_function(),
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=Collate_function(),
    )

    if args.dataset == "fever":
        test_examples_v2 = load_fever(mode="symmv2_test")
        test_dataset_v2 = PairDatasets(test_examples_v2, tokenizer, args)
        test_loader_v2 = torch.utils.data.DataLoader(
            test_dataset_v2,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            collate_fn=Collate_function(),
        )
        partial_examples = load_fever(mode="dev", vocab=args.claim_vocab)
        partial_dataset = PairDatasets(partial_examples, tokenizer, args)
        partial_loader = torch.utils.data.DataLoader(
            partial_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            collate_fn=Collate_function(),
        )

    training_steps = (len(train_loader) - 1 / args.epochs + 1) * args.epochs
    warmup_steps = 0.1 * training_steps
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=training_steps,
    )

    if args.dataset == "fever":
        (
            model,
            optimizer,
            lr_scheduler,
            train_loader,
            val_loader,
            test_loader,
            test_loader_v2,
            partial_loader,
        ) = accelerator.prepare(
            model,
            optimizer,
            lr_scheduler,
            train_loader,
            val_loader,
            test_loader,
            test_loader_v2,
            partial_loader,
        )
    else:
        (
            model,
            optimizer,
            lr_scheduler,
            train_loader,
            val_loader,
            test_loader,
        ) = accelerator.prepare(
            model, optimizer, lr_scheduler, train_loader, val_loader, test_loader
        )

    if args.evaluate:
        assert args.resume is not None
        for tmp_mode in ["expert", "uniform", "argmin"]:
            logger.info("## {} prediction".format(tmp_mode))
            _acc = validate(
                val_loader,
                model,
                accelerator,
                -1,
                False,
                args,
                datasetname=args.dataset,
                prob_mode=tmp_mode,
            )
            _acc = validate(
                test_loader,
                model,
                accelerator,
                -1,
                True,
                args,
                datasetname=args.dataset,
                prob_mode=tmp_mode,
            )
            if args.dataset == "fever":
                _acc = validate(
                    test_loader_v2,
                    model,
                    accelerator,
                    -1,
                    True,
                    args,
                    datasetname="symm_v2",
                    prob_mode=tmp_mode,
                )
                _acc = validate(
                    partial_loader,
                    model,
                    accelerator,
                    -1,
                    False,
                    args,
                    datasetname="fever_dev_{}".format(args.claim_vocab),
                    prob_mode=tmp_mode,
                )
        return

    # begin to train
    best_val = 0
    best_epoch = 0
    is_best = False
    for epoch in range(args.start_epoch, args.epochs):
        train(train_loader, model, optimizer, epoch, args, lr_scheduler, accelerator)
        prob_mode_list = ["expert"]
        for tmp_mode in prob_mode_list:
            logger.info("## {} prediction".format(tmp_mode))
            val_acc = validate(
                val_loader,
                model,
                accelerator,
                epoch,
                False,
                args,
                datasetname=args.dataset,
                prob_mode=tmp_mode,
            )

            if tmp_mode == "expert" and val_acc > best_val:
                best_val = val_acc
                best_epoch = epoch
                is_best = True

        if args.save and is_best:
            logger.info("Saving...")
            save_model = accelerator.unwrap_model(model)  # not required for optimizer
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": save_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                is_best,
                args.best_model_name,
                accelerator,
            )

        is_best = False

    logger.info(
        "Best Epoch: {}, Best val Acc: {:.3f}".format(
            best_epoch,
            best_val,
        )
    )


if __name__ == "__main__":
    main()
    logger.info("Done")
