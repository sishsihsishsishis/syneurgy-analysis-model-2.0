import logging
import math
from decimal import Decimal, InvalidOperation

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def safe_decimal(value):
    try:
        if math.isnan(value):  # check for NaN
            return Decimal('0')
        return Decimal(str(value))
    except (ValueError, InvalidOperation):
        return Decimal('0')

def get_positive_and_negative(listA, listV, user_list, meeting_id):
    try:
        total_rate = []
        pos_neg_rates = []
        total_pos_cnt_a = 0
        total_pos_cnt_v = 0
        total_cnt_a = 0
        total_cnt_v = 0
        start = 3

        for user in user_list:
                # Extract the last 2 digits of the user as an index
                i = int(user[-2:])
                total_count_a = 0
                positive_count_a = 0
                total_count_v = 0
                positive_count_v = 0
                
                # For listA
                for s in listA:
                    try:
                        value = s[start + i]
                        if value is not None:
                            total_count_a += 1
                            total_cnt_a += 1
                            if value > 0.0:
                                positive_count_a += 1
                                total_pos_cnt_a += 1
                    except (IndexError, TypeError) as e:
                        # Handle potential index errors or type issues in listA
                        logging.error(f"Error processing listA for user {user}: {e}")
                        
                # For listV
                for s in listV:
                    try:
                        value = s[start + i]
                        if value is not None:
                            total_count_v += 1
                            total_cnt_v += 1
                            if value > 0.0:
                                positive_count_v += 1
                                total_pos_cnt_v += 1
                    except (IndexError, TypeError) as e:
                        error = e
                        # Handle potential index errors or type issues in listV
                        # logging.error(f"Error processing listV for user {user}: {e}")
                        
                # Calculate positive and negative rates for listA
                positive_rate_a = positive_count_a / total_count_a if total_count_a != 0 else float('nan')
                negative_rate_a = float('nan') if math.isnan(positive_rate_a) else 1.0 - positive_rate_a

                # Calculate positive and negative rates for listV
                positive_rate_v = positive_count_v / total_count_v if total_count_v != 0 else float('nan')
                negative_rate_v = float('nan') if math.isnan(positive_rate_v) else 1.0 - positive_rate_v
                
                # Append results for this user
                pos_neg_rates.append(
                    {
                        'user': user,
                        'positive_rate_a': safe_decimal(positive_rate_a),
                        'negative_rate_a': safe_decimal(negative_rate_a),
                        'positive_rate_v': safe_decimal(positive_rate_v),
                        'negative_rate_v': safe_decimal(negative_rate_v)
                    }
                )
                # Update total_rate for both A and V
                if total_cnt_a > 0:
                    total_rate.append(total_pos_cnt_a / total_cnt_a)
                    total_rate.append(1 - (total_pos_cnt_a / total_cnt_a))
                if total_cnt_v > 0:
                    total_rate.append(total_pos_cnt_v / total_cnt_v)
                    total_rate.append(1 - (total_pos_cnt_v / total_pos_cnt_v))
                
        return pos_neg_rates    

    except (ValueError, IndexError) as e:
        logging.error(f"Error processing user {user}: {e}")
        
    return []
