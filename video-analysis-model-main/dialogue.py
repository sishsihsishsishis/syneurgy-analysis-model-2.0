import os
import math
import torch
import numpy as np
import pandas as pd
from collections import OrderedDict
from transformers import AutoTokenizer
from models.dialogue.models.ContextAwareDAC import ContextAwareDAC

os.environ["TOKENIZERS_PARALLELISM"] = "false"


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

config = {
        # data
        "data_dir": os.path.join(os.getcwd(), "data"),
        "dataset": "switchboard",
        "text_field": "Text",
        # "label_field":"act_label_1",
        "label_field": "DamslActTag",
        "max_len": 256,
        "batch_size": 64,
        "num_workers": 4,
        # model
        "model_name": os.path.join(
            os.path.abspath(BASE_DIR), "./models/dialogue/dialogueAct/roberta-base"
        ),  # roberta-base
        "hidden_size": 768,
        "num_classes": 43,  # there are 43 classes in switchboard corpus
        # training
        "save_dir": "./",
        "project": "dialogue-act-classification",
        "run_name": "context-aware-attention-dac",
        "lr": 1e-5,
        "monitor": "val_accuracy",
        "min_delta": 0.001,
        "filepath": "./checkpoints/{epoch}-{val_accuracy:4f}",
        "precision": 32,
        "average": "micro",
        "epochs": 100,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        # "device":torch.device("cpu"),
        "restart": False,
        "restart_checkpoint": "./checkpoints/epoch=10-val_accuracy=0.720291.ckpt",
    }

def create_label2act():
    label_dict = dict()
    train_data = pd.read_csv(
        os.path.join(os.path.abspath(BASE_DIR), "./models/dialogue/switchboard_valid.csv")
    )
    acts = list(train_data["DamslActTag"])
    classes = sorted(set(acts))
    for cls in classes:
        if cls not in label_dict.keys():
            label_dict[cls] = len(label_dict.keys())
    label2act = {}
    for k, v in label_dict.items():
        label2act[v] = k
    return label2act


dialogueAct_checkpoint = torch.load(
    os.path.join(
        os.path.abspath(BASE_DIR),
        "./models/dialogue/dialogueAct/checkpoints/epoch=4-val_accuracy=0.766026.ckpt",
    ),
    map_location="cpu",
)["state_dict"]

params = OrderedDict()
for k, v in dialogueAct_checkpoint.items():
    params[k.replace("model.", "")] = v
dialogueAct_model = ContextAwareDAC(
    model_name=config["model_name"],
    hidden_size=config["hidden_size"],
    num_classes=config["num_classes"],
    device=config["device"],
)
dialogueAct_model.load_state_dict(params, strict=False)
dialogueAct_model.to(config["device"])
dialogueAct_model.eval()
dialogueAct_tokenizer = AutoTokenizer.from_pretrained(config["model_name"])


def label2name():
    with open(
        os.path.join(os.path.abspath(BASE_DIR), "./models/dialogue/SWBD.txt"),
        "r",
        encoding="utf8",
    ) as f:
        swbd = f.readlines()
    with open(
        os.path.join(os.path.abspath(BASE_DIR), "./models/dialogue/DAMSL.txt"),
        "r",
        encoding="utf8",
    ) as f:
        damsl = f.readlines()
    l2n = {}
    for s, d in zip(swbd, damsl):
        s = s.strip()
        d = d.strip()
        l2n[s] = d
    return l2n


def go_dialogue_act(sentences):
    label2act = create_label2act()
    l2n = label2name()

    pred_acts_all = []
    with torch.no_grad():
        slice_num = 8
        sentences_group = [
            sentences[i * slice_num : (i + 1) * slice_num]
            for i in range(math.ceil((len(sentences) / slice_num)))
        ]

        for sent in sentences_group:
            input_encoding = dialogueAct_tokenizer.batch_encode_plus(
                sent,
                return_tensors="pt",
                return_attention_mask=True,
                max_length=256,
                padding="max_length",
                truncation=True,
            )
            # print(sent)
            input_encoding["input_ids"] = input_encoding["input_ids"].to(config["device"])  # type: ignore
            input_encoding["attention_mask"] = input_encoding["attention_mask"].to(config["device"])  # type: ignore
            logits = dialogueAct_model(input_encoding)
            probs = logits.softmax(dim=1)
            pred = torch.argmax(probs, dim=1)
            pred_labels = pred.cpu().numpy().tolist()
            pred_acts = [l2n[label2act[p]] for p in pred_labels]
            pred_acts_all.extend(pred_acts)

    return pred_acts_all
