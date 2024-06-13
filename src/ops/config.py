import argparse
from os.path import join
from collections import namedtuple


SOURCE_DIR = "dataset"

TextPairExample = namedtuple(
    "TextPairExample", ["id", "premise", "hypothesis", "label"]
)
FeverPairExample = namedtuple("FeverPairExample", ["id", "claim", "evidence", "label"])
PairExample = namedtuple("PairExample", ["id", "s1", "s2", "label"])
HardExample = namedtuple(
    "HardExample", ["input_id", "attention_mask", "segment_id", "uncertainty", "label"]
)

FEVER_MAPS = {"REFUTES": 0, "SUPPORTS": 1, "NOT ENOUGH INFO": 2}
FEVER_LABELS = ["REFUTES", "SUPPORTS", "NOT ENOUGH INFO"]
NLI_LABELS = ["contradiction", "entailment", "neutral"]
QQP_LABELS = ["not duplicated", "duplicated"]  # ['not_match', 'match']

HANS_SOURCE = join(SOURCE_DIR, "hans")
MULTINLI_SOURCE = join(SOURCE_DIR, "multinli")
FEVER_SOURCE = join(SOURCE_DIR, "fever")
QQP_PAWS_SOURCE = join(SOURCE_DIR, "qqp_paws")


parser = argparse.ArgumentParser(description="Debiasing")

parser.add_argument(
    "--pretrained_path",
    default="bert-base-uncased",
    type=str,
    help="pretrained model path",
)
parser.add_argument("--data_dir", default="dataset/", type=str, help="dir of dataset")
parser.add_argument(
    "--dataset",
    default="mnli",
    type=str,
    choices=["mnli", "fever", "qqp"],
    help="debiasing task",
)
parser.add_argument("--seed", default=777, type=int, help="seed")
parser.add_argument("--gpu", type=int, default=1, help="local gpu id")

parser.add_argument(
    "--workers",
    default=2,
    type=int,
    metavar="N",
    help="number of data loading workers (default: 2)",
)
parser.add_argument("--max_seq_len", default=128, type=int, help="max sequence length")
parser.add_argument(
    "--lr",
    "--learning-rate",
    default=5e-5,
    type=float,
    metavar="LR",
    help="initial learning rate",
    dest="lr",
)
parser.add_argument("--batch_size", default=64, type=int)
parser.add_argument(
    "--optimizer", default="AdamW", type=str, help="optimizer for training "
)
parser.add_argument(
    "--epochs", default=10, type=int, metavar="N", help="number of total epochs to run"
)
parser.add_argument(
    "--start_epoch",
    default=0,
    type=int,
    metavar="N",
    help="manual epoch number (useful on restarts)",
)
parser.add_argument("--best_model_name", type=str, default="", help="")
parser.add_argument(
    "--resume",
    default=None,
    type=str,
    metavar="PATH",
    help="path to latest checkpoint (default: none)",
)

parser.add_argument("--save_dir", type=str, default="saved_models")
parser.add_argument("--save", action="store_true", default=False)
parser.add_argument("--debug", action="store_true", default=False)
parser.add_argument("--model", type=str, default="bert")
parser.add_argument("--evaluate", action="store_true", default=False)
parser.add_argument("--shuffle_hans", action="store_true", default=False)
parser.add_argument("--claim_vocab", type=str, default="did not")

parser.add_argument("--num_experts", default=10, type=int)
parser.add_argument("--expert_layer_start", default=10000, type=int)
parser.add_argument("--num_topk_mask", default=0, type=int)
parser.add_argument("--router_loss", default=0.0, type=float)
parser.add_argument("--router_tau", default=1.0, type=float)
parser.add_argument("--router_dist_min", default=0.0, type=float)
parser.add_argument("--router_dist_max", default=0.0, type=float)

parser.add_argument("--max_grad_norm", default=0.0, type=float)
parser.add_argument("--f1_eval", action="store_true", default=False)
