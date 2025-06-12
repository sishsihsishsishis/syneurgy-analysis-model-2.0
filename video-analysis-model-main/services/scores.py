import math
from typing import List
from services.nlp import softmax_weights_output

def sigmoid(x):
    return 1 / (1 + math.exp(-10 * (x - 0.5)))

def compute_ave_score(scores: List[float]) -> float:
    return sum(scores) / len(scores)

def process_sync_data(sync_data: List[List[float]]) -> float:
    valid_scores = [d[1] for d in sync_data if not math.isnan(d[1])]
    if not valid_scores:
        return 0.0
    return sum((d / 2 + 0.5) for d in valid_scores) / len(valid_scores)

def get_scores(rppg_sync_data: List[List[float]], v_sync_data: List[List[float]], a_sync_data: List[List[float]]) -> dict:
    body_score = process_sync_data(rppg_sync_data)
    body_score = sigmoid(body_score) * 100 if body_score else 0.0

    radar_chart_array = [0.0] * 5  # Placeholder for radar chart data
    radar_score = softmax_weights_output(radar_chart_array)

    v_score = process_sync_data(v_sync_data)
    a_score = process_sync_data(a_sync_data)

    behavior_scores = [v_score, a_score]
    behavior_score = sigmoid(compute_ave_score(behavior_scores)) * 100 if behavior_scores else 0.0

    total_scores = [body_score, behavior_score]
    total_score = sigmoid(compute_ave_score(total_scores) / 100) * 100 if total_scores else 0.0

    return {
        "total": round(total_score),
        "body": round(sigmoid(body_score / 100) * 100),
        "behavior": round(sigmoid(behavior_score / 100) * 100),
        "brain": 0,
    }
