import math
from decimal import Decimal
from collections import defaultdict, OrderedDict

# Helper function to calculate time slot
def calculate_time_slot(start_time, end_time, chunk_size=30):
    # Calculate the average time and map it to a 30-second time slot
    return int(((start_time + end_time) / 2) // chunk_size * chunk_size)

def word_count(nlp_data, chunk_size=30):
    # Create a dictionary to store the word count for each speaker and time slot
    word_count = {}

    for entry in nlp_data:
        start_time = float(entry['start'])
        end_time = float(entry['end'])
        speaker = entry['speaker']
        sentence = entry['sentence']

        # Use the helper function to calculate the time slot
        time_slot = calculate_time_slot(start_time, end_time, chunk_size)

        # Split the sentence into words
        words = sentence.split()

        # Get or create the dictionary for the current speaker
        if speaker not in word_count:
            word_count[speaker] = {}

        # Update the word count for the speaker in the given time slot
        if time_slot in word_count[speaker]:
            word_count[speaker][time_slot] += Decimal(len(words))
        else:
            word_count[speaker][time_slot] = Decimal(len(words))

    return word_count

def calculate_speaker_time(nlp_data):
    speaker_time = {}
    total_speaking_time = Decimal('0')

    for entry in nlp_data:
        start_time = float(entry['start'])
        end_time = float(entry['end'])
        speaker = entry['speaker']
        
        # Calculate the duration for the current entry
        duration = end_time - start_time

        # Add the duration to the speaker's total time
        if speaker in speaker_time:
            speaker_time[speaker] += Decimal(str(duration))
        else:
            speaker_time[speaker] = Decimal(str(duration))

        # Update the speaking time
        total_speaking_time += Decimal(str(duration))
    
    # Calculate participation for each speaker
    participation = {
        speaker: (time / total_speaking_time * 100).quantize(Decimal('0.00'))
        for speaker, time in speaker_time.items()
    }

    return speaker_time, total_speaking_time, participation

def calculate_speaker_wpm(nlp_data):
    """
    Calculate words per minute for all speakers in a conversation and maintain running averages.
    
    Args:
        nlp_data (list): List of dictionaries containing conversation metadata with keys:
            - speaker: speaker identifier
            - start: start time in seconds
            - end: end time in seconds
            - sentence: the spoken text
            
    Returns:
        dict: Dictionary with speaker IDs and their average WPM
    """
    # Dictionary to store speaker statistics
    speaker_stats = {}
    
    def update_speaker_stats(speaker_id, words, duration):
        """Helper function to update speaker statistics"""
        if speaker_id not in speaker_stats:
            speaker_stats[speaker_id] = {
                'total_words': words,
                'total_duration_minutes': duration,
                'utterance_count': 1,
                'current_avg_wpm': 0
            }
        else:
            speaker_stats[speaker_id]['total_words'] += words
            speaker_stats[speaker_id]['total_duration_minutes'] += duration
            speaker_stats[speaker_id]['utterance_count'] += 1
        
        # Calculate average WPM
        stats = speaker_stats[speaker_id]
        if stats['total_duration_minutes'] > 0:
            stats['current_avg_wpm'] = stats['total_words'] / stats['total_duration_minutes']
    
    # Process each utterance
    for utterance in nlp_data:
        # Extract data from utterance
        speaker = utterance['speaker']
        duration_minutes = (float(utterance['end']) - float(utterance['start'])) / 60
        # print("start:", float(utterance['start']), "end:", float(utterance['end']))
        utterance_words = len(utterance['sentence'].split())
        
        # Update statistics for this speaker
        update_speaker_stats(speaker, utterance_words, duration_minutes)
        
        # Print current WPM stats for all speakers
        # print("\nCurrent WPM Statistics:")
        # print("-" * 50)
        # for spk, stats in speaker_stats.items():
        #     print(f"Speaker {spk}: {stats['current_avg_wpm']:.2f} WPM")
    
    print("speaker_stats", speaker_stats)
    # Create final dictionary with just speaker IDs and their average WPM
    final_wpm_dict = {
        speaker: stats['current_avg_wpm']
        for speaker, stats in speaker_stats.items()
    }
    print("fnal", final_wpm_dict)
    return final_wpm_dict

def calculate_speaker_rate_in_chunks(nlp_data, chunk_size=30):
    # Create a dictionary to store the word rates for each speaker in each time slot
    print("nlp_data", nlp_data)
    word_rates = {}
    average_rates = {}

    for entry in nlp_data:
        start_time = float(entry['start'])
        end_time = float(entry['end'])
        speaker = entry['speaker']
        sentence = entry['sentence']
        
        # Split the sentence into words
        words = sentence.split()
        num_words = len(words)
        
        # Process the dialogue in 30-second chunks
        current_time = start_time
        while current_time < end_time:
            next_chunk_end = min(current_time + chunk_size, end_time)
            duration = next_chunk_end - current_time
            print(next_chunk_end, duration)
            if duration > 0:  # Avoid division by zero
                # Calculate the proportional number of words for the current chunk
                chunk_word_count = int(num_words * (duration / (end_time - start_time)))
                
                rate = chunk_word_count / duration  # Words per second for the chunk
            else:
                rate = 0

            # Use the helper function to calculate the time slot
            time_slot = calculate_time_slot(current_time, next_chunk_end, chunk_size)

            # Initialize or update the word rates for this speaker and time slot
            if speaker not in word_rates:
                word_rates[speaker] = {}
            if time_slot in word_rates[speaker]:
                word_rates[speaker][time_slot] += Decimal(str(rate))
            else:
                word_rates[speaker][time_slot] = Decimal(str(rate))

            # Move to the next chunk
            current_time += chunk_size

    # Calculate the average rate for each speaker based on word_rates
    for speaker, time_slots in word_rates.items():
        total_rate = sum(time_slots.values())
        num_slots = len(time_slots)
        if num_slots > 0:
            average_rate = total_rate / num_slots
        else:
            average_rate = 0
        average_rates[speaker] = average_rate

    return word_rates, average_rates

def get_pie_and_bar(data, users):
    
    # Initialize all necessary lists
    s_keys = []
    s_time = []
    s_rate = []
    e_keys = []
    e_time = []
    e_rate = []
    a_keys = []
    a_time = []
    a_rate = []
    bar_speakers = []
    pie_emotions = []
    bar_emotions = []
    total = []
    sentences_array = []

    # Initialize speakers and emotion time tracking
    speakers_time = [0.0] * len(users)
    speakers_time_sep_by_emotions = [[0.0] * 3 for _ in range(len(users))]
    emotions_time = [0.0] * 3
    acts_time = [0.0] * 7

    # Mappings for speakers, emotions, and actions
    speakers_ind = {user: i for i, user in enumerate(users)}
    emotions_ind = {'negative': 0, 'neutral': 1, 'positive': 2}
    acts_ind = {
        'Statement-non-opinion': 0, 'Statement-opinion': 1, 'Collaborative Completion': 2,
        'Abandoned or Turn-Exit': 3, 'Uninterpretable': 4, 'Yes-No-Question': 5, 'Others': 6
    }

    # Process each row in the data
    for row in data:
        speaker = row['speaker']
        start = float(row['start'])
        end = float(row['end'])
        sentence = row['sentence']
        emotion = row['emotion']
        act = row['dialogue']
        
        start, end = float(start), float(end)
        delta_time = end - start
        act = act if act in acts_ind else 'Others'
        speakers_time[speakers_ind[speaker]] += delta_time
        emotions_time[emotions_ind[emotion]] += delta_time
        acts_time[acts_ind[act]] += delta_time
        speakers_time_sep_by_emotions[speakers_ind[speaker]][emotions_ind[emotion]] += delta_time
        sentences_array.append(sentence)

    # Calculate total time
    total_time = sum(speakers_time)
    total.append(total_time)

    # Populate speaker data
    for speaker, index in speakers_ind.items():
        s_keys.append(speaker)
        s_time.append(round(Decimal(speakers_time[index]), 4))
        s_rate.append(round(Decimal(speakers_time[index] / total_time), 4))

    # Populate emotion data
    for emotion, index in emotions_ind.items():
        e_keys.append(emotion)
        e_time.append(round(Decimal(emotions_time[index]), 3))
        e_rate.append(round(Decimal(emotions_time[index] / total_time), 3))
        
    for i, emotion_key in enumerate(e_keys):
        pie_emotion = {
            "emotion": e_keys,
            "emotion_time": emotions_time[i],
            "emotion_time_rate": e_rate[i]
        }
        pie_emotions.append(pie_emotion)
            
    # Populate action data
    for act, index in acts_ind.items():
        a_keys.append(act)
        a_time.append(round(Decimal(acts_time[index]), 3))
        a_rate.append(round(Decimal(acts_time[index] / total_time), 3))

    # Populate bar_speakers with speaker time per emotion
    for i in range(len(speakers_time_sep_by_emotions)):
        bar_speakers.append([round(Decimal(val), 3) for val in speakers_time_sep_by_emotions[i]])

    # Return all variables directly
    return s_keys, s_time, s_rate, e_keys, e_time, e_rate, a_keys, a_time, a_rate, bar_speakers, total, sentences_array


def get_radar_components(speakers_time, total_time, acts_time, emotions_time, sentences_array, radar_chart_list, r_keys, users):
    
    print("speakers_time:", speakers_time)
    print("total_time:", total_time)
    print("acts_time:", acts_time)
    print("emotions_time:", emotions_time)
    print("sentences_array:", sentences_array)
    
    
    radar_chart_ind = OrderedDict({
        "Equal Participation": 0,
        "Enjoyment": 1,
        "Shared Goal Commitment": 2,
        "Absorption or Task Engagement": 3,
        "Trust and Psychological Safety": 4
    })
    
    radar_chart_array = [0.0] * 5

    acts_ind = OrderedDict({
        "Statement-non-opinion": 0,
        "Statement-opinion": 1,
        "Collaborative Completion": 2,
        "Abandoned or Turn-Exit": 3,
        "Uninterpretable": 4,
        "Yes-No-Question": 5,
        "Others": 6
    })

    num_users = len(users)
    max_entropy = -math.log(1.0 / num_users) / math.log(2)
    
    speakers_time_rate = [time / Decimal(total_time) for time in speakers_time]
    
    entropy = sum(-float(rate) * (math.log(float(rate)) / math.log(2)) for rate in speakers_time_rate)
    equal_participation = entropy / max_entropy
    
    if math.isnan(equal_participation) or math.isinf(equal_participation):
        equal_participation = 0.1
    if equal_participation > 0.618:
        equal_participation = 0.5 + 1 / (1 + math.exp(-15 * (equal_participation - 1)))

    radar_chart_array[0] = equal_participation

    # Trust and Psychological Safety
    opinion_time = acts_time[acts_ind["Statement-opinion"]]
    non_opinion_time = acts_time[acts_ind["Statement-non-opinion"]]
    op_rate = opinion_time / (opinion_time + non_opinion_time)
    t = [equal_participation, op_rate]
    trust_psychological_safety = softmax_weights_output(t)
    
    if math.isnan(trust_psychological_safety) or math.isinf(trust_psychological_safety):
        trust_psychological_safety = 0.1

    radar_chart_array[4] = trust_psychological_safety

    # Enjoyment (NLP Emotion)
    emotions_ind = OrderedDict({
        "Negative": 0,
        "Neutral": 1,
        "Positive": 2
    })

    positive_time = emotions_time[emotions_ind["Positive"]]
    negative_time = emotions_time[emotions_ind["Negative"]]
    nlp_enjoyment = positive_time / (positive_time + negative_time)
    
    if math.isnan(nlp_enjoyment):
        nlp_enjoyment = 0.1
    if nlp_enjoyment > 0.95:
        nlp_enjoyment = 0.95
    if nlp_enjoyment > 0.618:
        nlp_enjoyment = 0.5 + 1 / (1 + math.exp(-15 * (nlp_enjoyment - 1)))
    if nlp_enjoyment < 0.1:
        nlp_enjoyment = 0.1

    radar_chart_array[1] = nlp_enjoyment

    # Shared Goal Commitment
    i_we_num = [0, 0]
    
    for s in sentences_array:
        words = s.split()
        for word in words:
            if word.lower() in ["i", "i'", "we", "we'"]:
                if word.lower().startswith("i"):
                    i_we_num[0] += 1
                elif word.lower().startswith("we"):
                    i_we_num[1] += 1
    
    shared_goal_commitment = i_we_num[1] * 1.0 / (i_we_num[0] + i_we_num[1]) if (i_we_num[0] + i_we_num[1]) > 0 else 0
    shared_goal_commitment = min(shared_goal_commitment * 1.5, 8.5)
    
    radar_chart_array[2] = shared_goal_commitment

    # Absorption or Task Engagement
    act_time_abandoned_rate = acts_time[acts_ind["Abandoned or Turn-Exit"]] / Decimal(total_time)
    act_time_not_abandoned_rate = Decimal(1.0) - Decimal(act_time_abandoned_rate)
    pn_rate = (positive_time + negative_time) / Decimal(total_time)
    ap = [act_time_not_abandoned_rate, pn_rate]
    absorption_or_task_engagement = softmax_weights_output(ap)
    
    if math.isnan(absorption_or_task_engagement) or math.isinf(absorption_or_task_engagement):
        absorption_or_task_engagement = 0.1
    
    radar_chart_array[3] = absorption_or_task_engagement

    # Final Radar Chart List
    for d in radar_chart_array:
        radar_chart_list.append(round(d, 3) if d is not None and not math.isinf(d) and not math.isnan(d) else 0.1)

    r_keys.extend(radar_chart_ind.keys())

def softmax_weights_output(t):
    # Placeholder implementation for softmax weights output
    max_t = max(t)
    exp_t = [math.exp(Decimal(i) - Decimal(max_t)) for i in t]
    sum_exp_t = sum(exp_t)
    softmax = [i / sum_exp_t for i in exp_t]
    weighted_sum = sum(Decimal(w) * Decimal(t[i]) for i, w in enumerate(softmax))
    return weighted_sum
