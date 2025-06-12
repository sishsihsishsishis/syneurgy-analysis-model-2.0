# FIX THIS MODEL
import os

import torch
from pyannote.audio import Pipeline

os.environ["TOKENIZERS_PARALLELISM"] = "false"

if torch.cuda.is_available():
    device = torch.device(0)
else:
    device = "cpu"

ANNOTE_PIPELINE = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token="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",
)
ANNOTE_PIPELINE.to(torch.device(device))


def diarization(wav_file, speakers_num=None):
    if speakers_num is None or speakers_num < 2:
        diarization = ANNOTE_PIPELINE(wav_file, min_speakers=0, max_speakers=1)
    else:
        diarization = ANNOTE_PIPELINE(
            wav_file, min_speakers=2, max_speakers=speakers_num + 1
        )

    result = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        result.append(
            {
                "start": turn.start,
                "stop": turn.end,
                "speaker": speaker.replace("SPEAKER", "speaker"),
            }
        )

    return result


def concat_speaker_chunks(speaker_chunks):
    speaker_time_chunks = []
    speaker_id_chunks = []

    last_start = 0
    last_stop = 0
    last_speaker = None
    for speaker_chunk in speaker_chunks:
        start = speaker_chunk["start"]
        stop = speaker_chunk["stop"]
        speaker = speaker_chunk["speaker"]

        if last_speaker is None:
            last_start = start
            last_stop = stop
            last_speaker = speaker
        elif speaker == last_speaker:
            last_stop = stop
        else:
            speaker_time_chunks.append((last_start, last_stop))
            speaker_id_chunks.append(last_speaker)

            last_start = start
            last_stop = stop
            last_speaker = speaker

    speaker_time_chunks.append((last_start, last_stop))
    speaker_id_chunks.append(last_speaker)

    return speaker_time_chunks, speaker_id_chunks
