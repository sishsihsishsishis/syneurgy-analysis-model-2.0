import math
import numpy as np
from typing import List

class Heatmap:
    def __init__(self, meeting_id, x, y, img):
        self.meeting_id = meeting_id
        self.x = x
        self.y = y
        self.img = img

def va_heatmap(meeting_id: int, v_data_string: List[List[str]], a_data_string: List[List[str]], num_pixel: int) -> List[Heatmap]:
    img = np.zeros((num_pixel, num_pixel), dtype=int)

    v_data = []
    for row in v_data_string:
        v_data_row = [float(s) if s.strip() else float('nan') for s in row]
        v_data.append(v_data_row)

    a_data = []
    for row in a_data_string:
        a_data_row = [float(s) if s.strip() else float('nan') for s in row]
        a_data.append(a_data_row)

    v_mean_index = 1
    a_mean_index = 1
    v_std_index = 2
    a_std_index = 2

    for row_id in range(len(v_data)):
        v_mean = v_data[row_id][v_mean_index]
        a_mean = a_data[row_id][a_mean_index]
        v_std = v_data[row_id][v_std_index]
        a_std = a_data[row_id][a_std_index]

        sq_v_std = math.nan if math.isnan(v_std) else v_std ** 2
        sq_a_std = math.nan if math.isnan(a_std) else a_std ** 2

        for i in range(num_pixel):
            for j in range(num_pixel):
                v, a = ind_to_va(i, j, num_pixel)
                c = (math.nan if math.isnan(v_mean) or math.isnan(sq_v_std) or
                     math.isnan(a_mean) or math.isnan(sq_a_std)
                     else ((v - v_mean) ** 2 / sq_v_std + (a - a_mean) ** 2 / sq_a_std))

                if not math.isnan(c) and c <= 1:
                    img[i][j] += 1

    json_array = []
    for i in range(num_pixel):
        for j in range(num_pixel):
            if img[i][j] > 0:
                heatmap = [i, j, img[i][j]]
                json_array.append(heatmap)

    return json_array

def ind_to_va(row_ind: int, col_ind: int, num_pixel: int) -> (float, float):
    v = 2.0 * col_ind / (num_pixel - 1) - 1.0  # v
    a = 1.0 - 2.0 * row_ind / (num_pixel - 1)  # a
    return v, a

# if __name__ == "__main__":
#     meetingID = 402
#     dataV = [
#         ["1000", "-0.28603768", "0.079539545", "-0.36557722", "-0.20649813"],
#         ["2000", "-0.31249177", "0.20895797", "-0.52144974", "-0.103533804"],
#     ]
#     dataA = [
#         ["1000", "0.29523683", "0.15291701", "0.44815385", "0.14231981"],
#         ["2000", "0.1841645", "0.12190655", "0.30607104", "0.062257946"],
#     ]
#     num_pixel = 100  

#     heatmap_result = va_heatmap(meetingID, dataV, dataA, num_pixel)

#     for heatmap in heatmap_result:
#         print(heatmap)
