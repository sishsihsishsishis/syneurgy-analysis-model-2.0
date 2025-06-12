import os
import cv2
import json
import math
import torch
import numpy as np
import pandas as pd
import logging
from scipy import linalg
from scipy import signal
from scipy import sparse
import face_recognition
import time
import traceback
from models.rppg.emonet import EmoNet

print(os.path.dirname(os.path.abspath(__file__)))
device = "cuda" if torch.cuda.is_available() else "cpu"

n_expression = 8
net = EmoNet(n_expression=n_expression).to(device)
state_dict_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    f"models/rppg/emonet_{n_expression}.pth",
)
state_dict = torch.load(str(state_dict_path), map_location=torch.device(device))
state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
net.load_state_dict(state_dict, strict=False)
net.eval()


def face_distance(face_encodings, face_to_compare):
    """
    Given a list of face encodings, compare them to a known face encoding and get a Euclidean distance
    for each comparison face. The distance tells you how similar the faces are.

    :param faces: List of face encodings to compare
    :param face_to_compare: A face encoding to compare against
    :return: A numpy nd-array with the distance for each face in the same order as the 'faces' array
    """
    if len(face_encodings) == 0:
        return np.empty((0))

    return np.linalg.norm(face_encodings - face_to_compare, axis=1)


def compare_faces(know_encoding, face_encoding, t):
    """
    Compare face encodings with a known encoding list.

    :param know_encoding: List of known encodings
    :param face_encoding: Encoding of the face to compare
    :param t: Threshold
    :return: -1 if not matched, otherwise the index of the most similar user
    """
    distance = face_distance(know_encoding, face_encoding)
    if any(distance <= t):
        return distance.argmin()
    else:
        return -1


def if_intersect(box1, box2):
    """
    Check if two rectangles intersect.
    (top, right, bottom, left)
    """
    if box1[3] > box2[1] or box2[3] > box1[1] or box1[0] > box2[2] or box2[0] > box1[2]:
        return False
    else:
        return True


def new_compare_faces(know_encoding, u_poses, face_encoding, cur_loc, t):
    flag = None
    for idx, d_loc in enumerate(u_poses):
        intersect = if_intersect(d_loc, cur_loc)
        if intersect:
            flag = idx

    if flag:
        distance = np.linalg.norm(know_encoding[flag] - face_encoding)
        if distance < 0.75:
            return flag
        else:
            return None
    else:
        return compare_faces(know_encoding, face_encoding, t)


def calculate_rppg(face_coor, frames, fps):
    """
    Calculate rppg for the given face coordinates and frames.

    :param face_coor: Face coordinates
    :param frames: Frames in one second of video
    :param fps: Frame rate
    :return: Calculated rppg list
    """
    top, right, bottom, left = face_coor
    face_frame_resized = []
    for frame in frames:
        face_frame = frame[top:bottom, left:right]
        frame_resized = cv2.resize(face_frame, (72, 72))
        face_frame_resized.append(frame_resized)

    face_frame_resized = np.array(face_frame_resized)

    rppg_ica_bvp = ICA_POH(face_frame_resized, fps)
    rppg_ica_bvp = np.array(rppg_ica_bvp)
    rppg_ica_bvp = np.round(
        (rppg_ica_bvp - np.mean(rppg_ica_bvp)) / np.std(rppg_ica_bvp), 4
    )

    return list(rppg_ica_bvp)


def calculate_emotion(face_coor, frame, username=None, res_dir=None, res_mapping=None):
    """
    Calculate emotion for the given face coordinates and frame.

    :param face_coor: Face coordinates
    :param frame: Middle frame
    :param username: Username
    :param res_dir: Result directory
    :param res_mapping: Result mapping
    :return: Emotion analysis result
    """
    top, right, bottom, left = face_coor

    face_frame = frame[top:bottom, left:right]
    frame_resized = cv2.resize(face_frame, (256, 256))

    if username:
        save_frame = frame_resized[:, :, ::-1]
        res_path = os.path.join(res_dir, "{}.jpg".format(username))
        res_mapping["{}.jpg".format(username)] = res_path
        cv2.imwrite(res_path, save_frame)

    frame_resized = np.transpose(frame_resized, (2, 0, 1)) / 255.0

    frame_resized = np.array([frame_resized])
    frame_resized = torch.FloatTensor(frame_resized).to(device)
    out = net(frame_resized)
    valence = out["valence"].cpu().detach().numpy()[0]
    arousal = out["arousal"].cpu().detach().numpy()[0]
    return valence, arousal


def jade(X, m, Wprev):
    n = X.shape[0]
    T = X.shape[1]
    nem = m
    seuil = 1 / math.sqrt(T) / 100
    if m < n:
        D, U = np.linalg.eig(np.matmul(X, np.mat(X).H) / T)
        Diag = D
        k = np.argsort(Diag)
        pu = Diag[k]
        ibl = np.sqrt(pu[n - m : n] - np.mean(pu[0 : n - m]))
        bl = np.true_divide(np.ones(m, 1), ibl)
        W = np.matmul(np.diag(bl), np.transpose(U[0:n, k[n - m : n]]))
        IW = np.matmul(U[0:n, k[n - m : n]], np.diag(ibl))
    else:
        IW = linalg.sqrtm(np.matmul(X, X.H) / T)
        W = np.linalg.inv(IW)

    Y = np.mat(np.matmul(W, X))
    R = np.matmul(Y, Y.H) / T
    C = np.matmul(Y, Y.T) / T
    Q = np.zeros((m * m * m * m, 1))
    index = 0
    for lx in range(m):
        Y1 = Y[lx, :]
        for kx in range(m):
            Yk1 = np.multiply(Y1, np.conj(Y[kx, :]))
            for jx in range(m):
                Yjk1 = np.multiply(Yk1, np.conj(Y[jx, :]))
                for ix in range(m):
                    Q[index] = (
                        np.matmul(Yjk1 / math.sqrt(T), Y[ix, :].T / math.sqrt(T))
                        - R[ix, jx] * R[lx, kx]
                        - R[ix, kx] * R[lx, jx]
                        - C[ix, lx] * np.conj(C[jx, kx])
                    )
                    index += 1
    # Compute and Reshape the significant Eigen
    D, U = np.linalg.eig(Q.reshape(m * m, m * m))
    Diag = abs(D)
    K = np.argsort(Diag)
    la = Diag[K]
    M = np.zeros((m, nem * m), dtype=complex)
    Z = np.zeros(m)
    h = m * m - 1
    for u in range(0, nem * m, m):
        Z = U[:, K[h]].reshape((m, m))
        M[:, u : u + m] = la[h] * Z
        h = h - 1
    # Approximate the Diagonalization of the Eigen Matrices:
    B = np.array([[1, 0, 0], [0, 1, 1], [0, 0 - 1j, 0 + 1j]])
    Bt = np.mat(B).H

    encore = 1
    if Wprev == 0:
        V = np.eye(m).astype(complex)
    else:
        V = np.linalg.inv(Wprev)
    # Main Loop:
    while encore:
        encore = 0
        for p in range(m - 1):
            for q in range(p + 1, m):
                Ip = np.arange(p, nem * m, m)
                Iq = np.arange(q, nem * m, m)
                g = np.mat([M[p, Ip] - M[q, Iq], M[p, Iq], M[q, Ip]])
                temp1 = np.matmul(g, g.H)
                temp2 = np.matmul(B, temp1)
                temp = np.matmul(temp2, Bt)
                D, vcp = np.linalg.eig(np.real(temp))
                K = np.argsort(D)
                la = D[K]
                angles = vcp[:, K[2]]
                if angles[0, 0] < 0:
                    angles = -angles
                c = np.sqrt(0.5 + angles[0, 0] / 2)
                s = 0.5 * (angles[1, 0] - 1j * angles[2, 0]) / c

                if abs(s) > seuil:
                    encore = 1
                    pair = [p, q]
                    G = np.mat([[c, -np.conj(s)], [s, c]])  # Givens Rotation
                    V[:, pair] = np.matmul(V[:, pair], G)
                    M[pair, :] = np.matmul(G.H, M[pair, :])
                    temp1 = c * M[:, Ip] + s * M[:, Iq]
                    temp2 = -np.conj(s) * M[:, Ip] + c * M[:, Iq]
                    temp = np.concatenate((temp1, temp2), axis=1)
                    M[:, Ip] = temp1
                    M[:, Iq] = temp2

    # Whiten the Matrix
    # Estimation of the Mixing Matrix and Signal Separation
    A = np.matmul(IW, V)
    S = np.matmul(np.mat(V).H, Y)
    return A, S


def ica(X, Nsources, Wprev=0):
    nRows = X.shape[0]
    nCols = X.shape[1]
    if nRows > nCols:
        print(
            "Warning - The number of rows is cannot be greater than the number of columns."
        )
        print("Please transpose input.")

    if Nsources > min(nRows, nCols):
        Nsources = min(nRows, nCols)
        print(
            "Warning - The number of sources cannot exceed number of observation channels."
        )
        print(
            "The number of sources will be reduced to the number of observation channels ",
            Nsources,
        )

    Winv, Zhat = jade(X, Nsources, Wprev)
    W = np.linalg.pinv(Winv)
    return W, Zhat


def detrend(input_signal, lambda_value):
    signal_length = input_signal.shape[0]
    # observation matrix
    H = np.identity(signal_length)
    ones = np.ones(signal_length)
    minus_twos = -2 * np.ones(signal_length)
    diags_data = np.array([ones, minus_twos, ones])
    diags_index = np.array([0, 1, 2])
    D = sparse.spdiags(
        diags_data, diags_index, (signal_length - 2), signal_length
    ).toarray()
    filtered_signal = np.dot(
        (H - np.linalg.inv(H + (lambda_value**2) * np.dot(D.T, D))), input_signal
    )
    return filtered_signal


def common_process_video(frames):
    "Calculates the average value of each frame."
    RGB = []
    for frame in frames:
        sum = np.sum(np.sum(frame, axis=0), axis=0)
        RGB.append(sum / (frame.shape[0] * frame.shape[1]))
    return np.asarray(RGB)


def ICA_POH(frames, FS):
    # Cut off frequency.
    LPF = 0.7
    HPF = 2.5
    RGB = common_process_video(frames)

    NyquistF = 1 / 2 * FS
    BGRNorm = np.zeros(RGB.shape)
    Lambda = 100
    for c in range(3):
        BGRDetrend = detrend(RGB[:, c], Lambda)
        BGRNorm[:, c] = (BGRDetrend - np.mean(BGRDetrend)) / np.std(BGRDetrend)
    _, S = ica(np.mat(BGRNorm).H, 3)

    # select BVP Source
    MaxPx = np.zeros((1, 3))
    for c in range(3):
        FF = np.fft.fft(S[c, :])
        F = np.arange(0, FF.shape[1]) / FF.shape[1] * FS * 60
        FF = FF[:, 1:]
        FF = FF[0]
        N = FF.shape[0]
        Px = np.abs(FF[: math.floor(N / 2)])
        Px = np.multiply(Px, Px)
        Fx = np.arange(0, N / 2) / (N / 2) * NyquistF
        Px = Px / np.sum(Px, axis=0)
        MaxPx[0, c] = np.max(Px)
    MaxComp = np.argmax(MaxPx)
    BVP_I = S[MaxComp, :]
    B, A = signal.butter(3, [LPF / NyquistF, HPF / NyquistF], "bandpass")
    BVP_F = signal.filtfilt(B, A, np.real(BVP_I).astype(np.double))

    BVP = BVP_F[0]
    return BVP


def filter_locations(input_locs, exist_locs):
    new_locs = []
    for face in input_locs:
        for init_face in exist_locs:
            if face[0] > init_face[2]:
                new_locs.append(face)
            elif face[2] < init_face[0]:
                new_locs.append(face)
            elif face[3] > init_face[1]:
                new_locs.append(face)
            elif face[1] < init_face[3]:
                new_locs.append(face)
            else:
                continue

    return new_locs


def get_face_means(faces):
    face_sizes = []
    for face in faces:
        # top right bottom left
        face = list(face)
        face_size = (face[1] - face[3]) * (face[2] - face[0])
        face_sizes.append(face_size)

    return np.mean(face_sizes)


def ract_face_loc(faces):
    new_locs = []

    for face_t in faces:
        # top right bottom left
        face_l = list(face_t)
        face_l[0] = max(int(face_l[0] - 0.2 * (face_l[2] - face_l[0])), 0)
        face_l[1] = max(int(face_l[1] + 0.1 * (face_l[1] - face_l[3])), 0)
        # face_l[2] = int(face_l[2] + 0.1*(face_l[2]-face_l[0]))
        face_l[2] = max(int(face_l[2]), 0)
        face_l[3] = max(int(face_l[3] - 0.1 * (face_l[1] - face_l[3])), 0)
        new_locs.append(tuple(face_l))

    return new_locs


def compare_locs(dict_1, dict_2):
    if set(dict_1.keys()) == set(dict_2.keys()):
        for k, v in dict_1.items():
            box1 = v
            box2 = dict_2[k]
            if not if_intersect(box1, box2):
                return True
    else:
        return True


def remove_user_anchor(user, anchor_locs):
    for ix, anchors in enumerate(anchor_locs):
        if isinstance(anchors, str):
            data = json.loads(anchors)
        elif isinstance(anchors, dict):
            data = anchors
        else:
            raise ValueError("The type of anchor_locs is not supported!")

        try:
            del data[user]
            anchor_locs[ix] = data
        except Exception as e:
            continue
    return anchor_locs


def convert_face_location_2_xyxy(anchor_locations: dict):
    # Convert face_recognition face locations to x1, y1, x2, y2 format
    xyxy_locations = []
    for anchor_location in anchor_locations:
        if isinstance(anchor_location, str):
            anchor_location = json.loads(anchor_location)
        elif isinstance(anchor_location, dict):
            pass
        else:
            raise ValueError("The type of anchor_location must be dict or str")
        xyxy_location = {}
        for user, face_loc in anchor_location.items():
            x1, y1 = face_loc[3], face_loc[0]
            x2, y2 = face_loc[1], face_loc[2]
            xyxy_location[user] = [x1, y1, x2, y2]
        xyxy_locations.append(json.dumps(xyxy_location))
    return xyxy_locations


def go_rppg(video_path, res_dir, compare_thresh=0.7):
    try:
        """
        :param video_path: video file path
        :param rppg_savepath:  rppg results save path, .csv
        :param v_savepath:  valence results save path, .csv
        :param a_savepath:  arousal results save path, .csv
        :return: None
        """
        rppg_savepath = os.path.join(res_dir, "rppg_results.csv")
        anchor_savepath = os.path.join(res_dir, "anchor_results.csv")
        v_savepath = os.path.join(res_dir, "v_results.csv")
        a_savepath = os.path.join(res_dir, "a_results.csv")
        res_mapping = {
            "rppg_results.csv": rppg_savepath,
            "v_results.csv": v_savepath,
            "a_results.csv": a_savepath,
            "anchor_results.csv": anchor_savepath,
        }
        # User list
        user_list = []
        # User encodings
        user_encodings = []
        user_init_locations = []
        # know_encoding = []
        user_positions = []

        # Analysis results
        rppg_list = []
        valence_list = []
        arousal_list = []

        # Timestamp
        rppg_timestamp = []
        emotion_timestamp = []

        ### Additional parameters
        anchor_locations = []
        anchor_timestamps = []
        dict_location = {}

        videoCapture = cv2.VideoCapture(video_path)
        fps = videoCapture.get(cv2.CAP_PROP_FPS)

        ts = int(1 * fps)
        videoCapture.set(cv2.CAP_PROP_POS_FRAMES, int(fps) * 3)

        success, frame = videoCapture.read()

        # Store one second of video frames
        frames_by_second = []

        frames_num = 0
        seconds = 1
        total = 0
        time_cos = [0, 0, 0, 0]
        while True:
            # logging.info(f"Working on frame {frames_num}")
            frame = frame[:, :, ::-1]
            frame = np.asarray(frame)
            frames_by_second.append(frame)
            success, frame = videoCapture.read()
            if not success:
                break
            total += 1

            rppg_timestamp.append(int(total * 1 / fps * 1000))  # Add timestamp

            frames_num += 1

            # When the number of frames reaches one second, perform rppg and emotion analysis
            if (frames_num % ts == 0) & (frames_num > 0):
                temp_dict_location = {}
                frames = np.asarray(frames_by_second)
                # Use the middle frame for face detection
                frame_selected = frames[int(ts) // 2]
                _start = time.time()
                face_locations = face_recognition.face_locations(frame_selected)
                time_cos[0] += time.time() - _start
                # Calculate average face size

                # Filter detected faces
                if len(user_init_locations) > 0:
                    face_locations = filter_locations(face_locations, user_init_locations)

                _start = time.time()
                # Encode detected faces
                face_encodings = face_recognition.face_encodings(
                    frame_selected, face_locations
                )
                time_cos[1] += time.time() - _start

                # First time faces are detected
                if len(user_encodings) == 0:  # Init
                    user_encodings = list(face_encodings)
                    user_positions = list(face_locations)
                    # Fill forward
                    user_list = [f"user_{i}" for i in range(len(user_encodings))]

                    rppg_list = [
                        [0] * int((frames_num - ts)) for _ in range(len(user_encodings))
                    ]
                    valence_list = [
                        [0] * int((seconds - 1)) for _ in range(len(user_encodings))
                    ]
                    arousal_list = [
                        [0] * int((seconds - 1)) for _ in range(len(user_encodings))
                    ]

                    # Calculate rppg and emotion
                    for idx, face_location in enumerate(face_locations):
                        face = face_locations[idx]

                        ### Update temp_dict_location
                        temp_dict_location[user_list[idx]] = face_location
                        _start = time.time()
                        rppg = calculate_rppg(face, frames, fps)
                        time_cos[2] += time.time() - _start
                        _start = time.time()
                        valence, arousal = calculate_emotion(
                            face, frame_selected, user_list[idx], res_dir, res_mapping
                        )
                        time_cos[3] += time.time() - _start
                        rppg_list[idx] = rppg_list[idx] + rppg
                        arousal_list[idx].append(arousal)
                        valence_list[idx].append(valence)

                else:
                    users_num = np.arange(len(user_list))

                    # Users detected in the current frame
                    detected_users = []

                    for idx, face_encoding in enumerate(face_encodings):
                        face = face_locations[idx]

                        match = new_compare_faces(
                            user_encodings,
                            user_positions,
                            face_encoding,
                            face_locations[idx],
                            compare_thresh,
                        )

                        # If no match, consider as a new user
                        if match == -1:  # New user
                            user_encodings.append(face_encoding)
                            user_positions.append(face_locations[idx])

                            user_list.append(f"user_{len(user_list)}")

                            ### Update temp_dict_location
                            temp_dict_location[f"user_{len(user_list) - 1}"] = (
                                face_locations[idx]
                            )

                            try:
                                _start = time.time()
                                rppg = calculate_rppg(face, frames, fps)
                                time_cos[2] += time.time() - _start
                            except Exception as e:
                                # Fill forward
                                rppg = [0] * int(fps)
                            _start = time.time()
                            valence, arousal = calculate_emotion(
                                face, frame_selected, user_list[-1], res_dir, res_mapping
                            )
                            time_cos[3] += time.time() - _start
                            rppg_list.append([0] * int((frames_num - ts)) + rppg)
                            alist = [0] * int((seconds - 1))
                            vlist = [0] * int((seconds - 1))
                            alist.append(arousal)
                            vlist.append(valence)
                            arousal_list.append(alist)
                            valence_list.append(vlist)
                            detected_users.append(len(rppg_list) - 1)

                        elif match is None:
                            continue

                        else:  # Old user
                            # Users detected in the current second
                            if match not in detected_users:

                                ### Update temp_dict_location
                                temp_dict_location[f"user_{match}"] = face_locations[idx]

                                try:
                                    _start = time.time()
                                    rppg = calculate_rppg(face, frames, fps)
                                    time_cos[2] += time.time() - _start
                                except Exception as e:
                                    rppg = [0] * int(fps)
                                _start = time.time()
                                valence, arousal = calculate_emotion(face, frame_selected)
                                time_cos[3] += time.time() - _start
                                rppg_list[match] = rppg_list[match] + rppg
                                arousal_list[match].append(arousal)
                                valence_list[match].append(valence)
                                detected_users.append(match)

                    # Users not detected in the current second need to be filled forward
                    undetect_users = np.setdiff1d(users_num, detected_users)
                    for u in undetect_users:
                        rppg_list[u] += [0] * ts
                        arousal_list[u].append(0)
                        valence_list[u].append(0)

                emotion_timestamp.append(seconds * 1 * 1000)  # Add timestamp
                seconds += 1
                frames_by_second = []

                ### Add timestamps and face locations
                if compare_locs(dict_location, temp_dict_location):
                    anchor_timestamps.append(int(total * 1 / fps * 1000 - 500))
                    anchor_locations.append(json.dumps(temp_dict_location))
                dict_location = temp_dict_location

            # Test, monitor analysis progress
            if (total % (fps * 60) == 0) & (total > 0):
                logging.info(f"{total // (fps * 60)} minutes of data extraction was completed !")
        # logging.info("各步骤耗时(ms)：", time_cos)
        # Define save data
        rppg_columns = []
        valence_columns = []
        arousal_columns = []
        rppg_datas = []
        valence_datas = []
        arousal_datas = []
        for idx in range(len(user_list)):

            rppg = np.abs(rppg_list[idx])
            # Delete abnormal users - if more than 80% of the total time is not detected, consider as abnormal
            if np.percentile(rppg, 80) > 1e-3:
                rppg_columns.append(user_list[idx])
                valence_columns.append(user_list[idx])
                arousal_columns.append(user_list[idx])

                rppg_datas.append(rppg_list[idx])
                valence_datas.append(valence_list[idx])
                arousal_datas.append(arousal_list[idx])
            else:
                removing = res_mapping.pop("{}.jpg".format(user_list[idx]), None)
                anchor_locations = remove_user_anchor(user_list[idx], anchor_locations)
                if removing:
                    os.remove(removing)

        res_mapping["max_user_num"] = len(rppg_list)

        rppg_datas = np.array(rppg_datas).T
        valence_datas = np.array(valence_datas).T
        arousal_datas = np.array(arousal_datas).T

        rppg_timestamp = rppg_timestamp[: rppg_datas.shape[0]]
        emotion_timestamp = emotion_timestamp[: valence_datas.shape[0]]

        ### Save timestamps and face locations
        anchor_speakers_df = pd.DataFrame(index=anchor_timestamps)
        anchor_speakers_df["user_locs"] = convert_face_location_2_xyxy(anchor_locations)
        anchor_speakers_df.to_csv(anchor_savepath)

        rppg_df = pd.DataFrame(index=rppg_timestamp, columns=rppg_columns, data=rppg_datas)
        rppg_df.insert(loc=0, column="rppg_std", value=np.std(rppg_datas, axis=1))
        rppg_df.insert(loc=0, column="rppg_mean", value=np.mean(rppg_datas, axis=1))
        rppg_df.to_csv(rppg_savepath)

        v_df = pd.DataFrame(
            index=emotion_timestamp, columns=valence_columns, data=valence_datas
        )
        v_df.insert(loc=0, column="v_std", value=np.std(valence_datas, axis=1))
        v_df.insert(loc=0, column="v_mean", value=np.mean(valence_datas, axis=1))
        v_df.to_csv(v_savepath)

        a_df = pd.DataFrame(
            index=emotion_timestamp, columns=arousal_columns, data=arousal_datas
        )
        a_df.insert(loc=0, column="a_std", value=np.std(arousal_datas, axis=1))
        a_df.insert(loc=0, column="a_mean", value=np.mean(arousal_datas, axis=1))
        a_df.to_csv(a_savepath)
        return res_mapping

    except Exception as e:
            logging.error(f"Error in rrpg: {traceback.format_exc()}")
            return ""
