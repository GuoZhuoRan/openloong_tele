import math
import numpy as np

from constants_vuer import grd_yup2grd_zup, hand2inspire,zb_hand_rotation,grd_zup2grd_yup
from motion_utils import mat_update, fast_mat_inv


class VuerPreprocessor:
    def __init__(self):
        self.vuer_head_mat = np.array([[1, 0, 0, 0],
                                  [0, 1, 0, 1.5],
                                  [0, 0, 1, -0.2],
                                  [0, 0, 0, 1]])
        self.vuer_right_wrist_mat = np.array([[1, 0, 0, 0.5],
                                         [0, 1, 0, 1],
                                         [0, 0, 1, -0.5],
                                         [0, 0, 0, 1]])
        self.vuer_left_wrist_mat = np.array([[1, 0, 0, -0.5],
                                        [0, 1, 0, 1],
                                        [0, 0, 1, -0.5],
                                       [0, 0, 0, 1]])

    def process(self, tv):
        self.vuer_head_mat = mat_update(self.vuer_head_mat, tv.head_matrix.copy())
        self.vuer_right_wrist_mat = mat_update(self.vuer_right_wrist_mat, tv.right_hand.copy())
        self.vuer_left_wrist_mat = mat_update(self.vuer_left_wrist_mat, tv.left_hand.copy())

        # print("vuer_head_mat:", self.vuer_head_mat)

        # change of basis
        head_mat = grd_yup2grd_zup @ self.vuer_head_mat @ fast_mat_inv(grd_yup2grd_zup)  ##frame basis transformed to z_up frame, head_mat expressed in z_up frame
        right_wrist_mat = grd_yup2grd_zup @ self.vuer_right_wrist_mat @ fast_mat_inv(grd_yup2grd_zup)
        left_wrist_mat = grd_yup2grd_zup @ self.vuer_left_wrist_mat @ fast_mat_inv(grd_yup2grd_zup) #self.vuer_left_wrist_mat in y_up

        # rel_left_wrist_mat = left_wrist_mat @ hand2inspire
        rel_left_wrist_mat = left_wrist_mat@zb_hand_rotation
        rel_left_wrist_mat[0:3, 3] = rel_left_wrist_mat[0:3, 3] - head_mat[0:3, 3]

        # rel_right_wrist_mat = right_wrist_mat @ hand2inspireF  # wTr = wTh @ hTr
        rel_right_wrist_mat = right_wrist_mat@zb_hand_rotation
        rel_right_wrist_mat[0:3, 3] = rel_right_wrist_mat[0:3, 3] - head_mat[0:3, 3]

        # homogeneous
        left_fingers = np.concatenate([tv.left_landmarks.copy().T, np.ones((1, tv.left_landmarks.shape[0]))])  ## coordinate in y_up frame
        right_fingers = np.concatenate([tv.right_landmarks.copy().T, np.ones((1, tv.right_landmarks.shape[0]))])

        # print("left_fingers:", tv.left_landmarks) ## given test, left_fingers is in y_up
        # print("left finger vuer: " ,self.vuer_left_wrist_mat)

        # change of basis
        left_fingers = grd_yup2grd_zup @ left_fingers ## in z_up frame
        right_fingers = grd_yup2grd_zup @ right_fingers

        rel_left_fingers = fast_mat_inv(left_wrist_mat) @ left_fingers # fingers in palm coordinate
        rel_right_fingers = fast_mat_inv(right_wrist_mat) @ right_fingers
        rel_left_fingers = (zb_hand_rotation.T @ rel_left_fingers)[0:3, :].T
        rel_right_fingers = (zb_hand_rotation.T @ rel_right_fingers)[0:3, :].T

        return head_mat, rel_left_wrist_mat, rel_right_wrist_mat, rel_left_fingers, rel_right_fingers

    def get_hand_gesture(self, tv):
        self.vuer_right_wrist_mat = mat_update(self.vuer_right_wrist_mat, tv.right_hand.copy())
        self.vuer_left_wrist_mat = mat_update(self.vuer_left_wrist_mat, tv.left_hand.copy())

        # change of basis
        right_wrist_mat = grd_yup2grd_zup @ self.vuer_right_wrist_mat @ fast_mat_inv(grd_yup2grd_zup)
        left_wrist_mat = grd_yup2grd_zup @ self.vuer_left_wrist_mat @ fast_mat_inv(grd_yup2grd_zup)

        left_fingers = np.concatenate([tv.left_landmarks.copy().T, np.ones((1, tv.left_landmarks.shape[0]))])
        right_fingers = np.concatenate([tv.right_landmarks.copy().T, np.ones((1, tv.right_landmarks.shape[0]))])

        # change of basis
        left_fingers = grd_yup2grd_zup @ left_fingers
        right_fingers = grd_yup2grd_zup @ right_fingers

        rel_left_fingers = fast_mat_inv(left_wrist_mat) @ left_fingers
        rel_right_fingers = fast_mat_inv(right_wrist_mat) @ right_fingers
        rel_left_fingers = (hand2inspire.T @ rel_left_fingers)[0:3, :].T
        rel_right_fingers = (hand2inspire.T @ rel_right_fingers)[0:3, :].T
        all_fingers = np.concatenate([rel_left_fingers, rel_right_fingers], axis=0)

        return all_fingers


