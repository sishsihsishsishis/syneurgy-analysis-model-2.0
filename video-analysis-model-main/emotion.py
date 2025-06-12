import os
import torch
import vocab
import random
import numpy as np
import logging
import traceback
from transformers import AutoTokenizer
from models.emotion.emotionFlow.model import CRFModel
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print(BASE_DIR)

CONFIG = {
    "bert_path": os.path.join(
        os.path.abspath(BASE_DIR), "models/emotion/dialogueAct/roberta-base"
    ),
    "epochs": 20,
    "lr": 1e-4,
    "ptmlr": 5e-6,
    "batch_size": 16,
    "max_len": 256,
    "max_value_list": 16,
    "bert_dim": 1024,
    "pad_value": 1,
    "shift": 1024,
    "dropout": 0.3,
    "p_unk": 0.1,
    "data_splits": 20,
    "num_classes": 7,
    "wp": 1,
    "wp_pretrain": 5,
    "data_path": os.path.join(
        os.path.abspath(BASE_DIR), "models/emotion/emotionFlow/MELD/data/MELD/"
    ),
    "accumulation_steps": 8,
    "rnn_layers": 2,
    "tf_rate": 0.8,
    "aux_loss_weight": 0.3,
    "ngpus": torch.cuda.device_count(),
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    "weights_path": os.path.join(
        os.path.abspath(BASE_DIR),
        "models/emotion/emotionFlow/models/roberta-base-meld.pkl",
    ),
    "vocab_dict_path": os.path.join(
        os.path.abspath(BASE_DIR),
        "models/emotion/emotionFlow/vocabs/speaker_vocab_temp.pkl",
    ),
}

def pad_to_len(list_data, max_len, pad_value):
    list_data = list_data[-max_len:]
    len_to_pad = max_len - len(list_data)
    pads = [pad_value] * len_to_pad
    list_data.extend(pads)
    return list_data

def get_vocabs(datas, speaker_vocab_dict_path):
    speaker_vocab = vocab.UnkVocab()
    for data in datas:
        sentence, speaker = data
        speaker_vocab.word2index(speaker.lower(), train=True)
    speakers = list(speaker_vocab.counts.keys())
    speaker_vocab = vocab.UnkVocab()
    for speaker in speakers:
        speaker_vocab.word2index(speaker, train=True)

    torch.save(speaker_vocab.to_dict(), speaker_vocab_dict_path)

def get_inference_data(datas, tokenizer, speaker_vocab_dict_path):
    speaker_vocab = vocab.UnkVocab.from_dict(torch.load(speaker_vocab_dict_path))
    ret_utterances = []
    ret_speaker_ids = []
    utterances = []
    full_contexts = []
    speaker_ids = []
    max_turns = 0

    for data in datas:
        sentence, speaker = data
        utterance = sentence.replace("’", "'").replace('"', "")
        utterance = speaker + " says:, " + utterance
        speaker = speaker.lower()
        
        # logging.info(f"Processing sentence: {sentence}")
        # logging.info(f"Speaker: {speaker}")

        speaker_id = speaker_vocab.word2index(speaker)
        token_ids = tokenizer(utterance, add_special_tokens=False)["input_ids"] + [
            CONFIG["SEP"]
        ]
        full_context = []
        if len(utterances) > 0:
            context = utterances[-3:]
            for pre_uttr in context:
                full_context += pre_uttr
        full_context += token_ids

        query = "Now " + speaker + " feels <mask>"
        query_ids = tokenizer(query, add_special_tokens=False)["input_ids"] + [
            CONFIG["SEP"]
        ]
        full_context += query_ids

        full_context = pad_to_len(full_context, CONFIG["max_len"], CONFIG["pad_value"])
        utterances.append(token_ids)
        full_contexts.append(full_context)
        speaker_ids.append(speaker_id)

    max_turns = max(max_turns, len(utterances))
    ret_utterances.append(full_contexts)
    ret_speaker_ids.append(speaker_ids)

    pad_utterance = (
        [CONFIG["SEP"]]
        + tokenizer("1", add_special_tokens=False)["input_ids"]
        + [CONFIG["SEP"]]
    )
    pad_utterance = pad_to_len(pad_utterance, CONFIG["max_len"], CONFIG["pad_value"])

    ret_mask = []
    ret_last_turns = []
    for dial_id, utterances in enumerate(ret_utterances):
        mask = [1] * len(utterances)
        while len(utterances) < max_turns:
            utterances.append(pad_utterance)
            ret_speaker_ids[dial_id].append(0)
            mask.append(0)
        ret_mask.append(mask)
        ret_utterances[dial_id] = utterances

        last_turns = [-1] * max_turns
        for turn_id in range(max_turns):
            curr_spk = ret_speaker_ids[dial_id][turn_id]
            if curr_spk == 0:
                break
            for idx in range(0, turn_id):
                if curr_spk == ret_speaker_ids[dial_id][idx]:
                    last_turns[turn_id] = idx
        ret_last_turns.append(last_turns)

    logging.info(f"Dataset created with {len(ret_utterances)} utterances")

    new_ret_utterances, new_ret_speaker_ids, new_ret_mask, new_ret_last_turns = (
        [],
        [],
        [],
        [],
    )
    for a, b, c, d in zip(
        ret_utterances[0], ret_speaker_ids[0], ret_mask[0], ret_last_turns[0]
    ):
        new_ret_utterances.append([a])
        new_ret_speaker_ids.append([b])
        new_ret_mask.append([c])
        new_ret_last_turns.append([d])

    dataset = TensorDataset(
        torch.LongTensor(new_ret_utterances),
        torch.LongTensor(new_ret_speaker_ids),
        torch.ByteTensor(new_ret_mask),
        torch.LongTensor(new_ret_last_turns),
    )
    logging.info(f"TensorDataset created with {len(dataset)} items")
    return dataset


def infer(model, data):
    try:
        pred_list = []
        model.eval()
        sampler = SequentialSampler(data)
        dataloader = DataLoader(
            data,
            batch_size=CONFIG["batch_size"],
            sampler=sampler,
            num_workers=0,
        )

        logging.info(f"Starting inference on {len(data)} items")
        for batch_id, batch_data in enumerate(dataloader):
            batch_data = [x.to(CONFIG["device"]) for x in batch_data]
            sentences = batch_data[0]
            speaker_ids = batch_data[1]
            mask = batch_data[2]
            last_turns = batch_data[3]
            outputs = model(sentences, mask, speaker_ids, last_turns)
            temp = np.array(outputs)
            for batch_idx in range(mask.shape[0]):
                for seq_idx in range(mask.shape[1]):
                    if mask[batch_idx][seq_idx]:
                        pred_list.append(outputs[batch_idx][seq_idx])
        logging.info(f"Inference completed with {len(pred_list)} predictions")
        return pred_list
    except Exception as e:
        logger.error(f"Error during run: {e}")
        logger.error(traceback.format_exc())
        return None

def go_emotion(emotion_datas):
    try: 
        # Generate inference data
        speaker_vocab_dict_path = CONFIG["vocab_dict_path"]
        id2sent = {
            0: "neutral",
            1: "surprise",
            2: "fear",
            3: "sadness",
            4: "joy",
            5: "disgust",
            6: "anger",
        }

        fix_emotion = {
            "neutral": "neutral",
            "surprise": "positive",
            "fear": "negative",
            "sadness": "negative",
            "joy": "positive",
            "disgust": "positive",
            "anger": "positive",
        }
        sentiment_cate_all = []
        get_vocabs(emotion_datas, speaker_vocab_dict_path)

        dataset = get_inference_data(
            emotion_datas, emotion_tokenizer, speaker_vocab_dict_path
        )
        if dataset is None:
            logging.error("Dataset creation failed.")
            return []

        pred_list = infer(emotion_model, dataset)
        if pred_list is None:
            logging.error("Inference failed.")
            return []

        for pred in pred_list:
            sentiment_cate_all.append(fix_emotion[id2sent[pred]])

        logging.info("Emotion analysis completed successfully")
        return sentiment_cate_all
    
    except Exception as e:
        logger.error(f"Error during emotion analysis: {e}")
        logger.error(traceback.format_exc())
        return []

# Set random seed
seed = 1024
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = True

# Load tokenizer
emotion_tokenizer = AutoTokenizer.from_pretrained(CONFIG["bert_path"])
_special_tokens_ids = emotion_tokenizer("")["input_ids"]
CLS = _special_tokens_ids[0]
SEP = _special_tokens_ids[1]
CONFIG["CLS"] = CLS
CONFIG["SEP"] = SEP


# Initialize and load the model
def load_model(config):
    try:
        model = CRFModel(config)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(torch.device(device))
        model.load_state_dict(torch.load(config["weights_path"]))
        logging.info('Model weights loaded successfully')
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        logger.error(traceback.format_exc())
        raise

emotion_model = load_model(CONFIG)
logging.info(f"Emotion model loaded on device: {CONFIG['device']}")
