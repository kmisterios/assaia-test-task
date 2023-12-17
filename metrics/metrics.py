"""Metrics for the task."""

from typing import List

import numpy as np
import pandas as pd

from utils import TimeInterval


class MetricCalculator:
    def __init__(
        self, num_frames: int, start_weight: float = 2, weight_ratio: float = 0.2
    ):
        self.num_frames = num_frames
        self.metric_df = None
        self.weight_ratio = weight_ratio
        self.start_weight = start_weight

    def __reset_df(self):
        d = {
            "num_frame": list(range(self.num_frames)),
            "labels": [0] * self.num_frames,
            "predict": [0] * self.num_frames,
        }
        self.metric_df = pd.DataFrame(d)
        self.metric_df.set_index("num_frame")

    def __get_ids(self, intervals: List[TimeInterval]):
        all_ids = []
        weighted_ids = []
        for interval in intervals:
            interval_ids = list(range(interval.min, interval.max + 1))
            all_ids += interval_ids
            num_weighted = max(1, int(len(interval_ids) * self.weight_ratio))
            weighted_ids += interval_ids[:num_weighted]
        return np.unique(all_ids), np.unique(weighted_ids)

    def precision(
        self,
        label_intervals: List[TimeInterval],
        pred_intervals: List[TimeInterval],
        reset_df=True,
    ):
        if reset_df or self.metric_df is None:
            self.__reset_df()
            label_ids, label_weighted_ids = self.__get_ids(label_intervals)
            pred_ids, pred_weighted_ids = self.__get_ids(pred_intervals)
            self.metric_df.loc[label_ids, "labels"] = 1
            self.metric_df.loc[pred_ids, "predict"] = 1
            self.metric_df.loc[label_weighted_ids, "labels"] = self.start_weight
            self.metric_df.loc[pred_weighted_ids, "predict"] = self.start_weight
        df_t = self.metric_df[self.metric_df["labels"] == self.metric_df["predict"]]
        df_f = self.metric_df[self.metric_df["labels"] != self.metric_df["predict"]]
        TP = df_t["labels"].sum()
        FP = df_f["predict"].sum()
        if TP + FP == 0:
            return 1
        return TP / (TP + FP)

    def recall(
        self,
        label_intervals: List[TimeInterval],
        pred_intervals: List[TimeInterval],
        reset_df=True,
    ):
        if reset_df or self.metric_df is None:
            self.__reset_df()
            label_ids, label_weighted_ids = self.__get_ids(label_intervals)
            pred_ids, pred_weighted_ids = self.__get_ids(pred_intervals)
            self.metric_df.loc[label_ids, "labels"] = 1
            self.metric_df.loc[pred_ids, "predict"] = 1
            self.metric_df.loc[label_weighted_ids, "labels"] = self.start_weight
            self.metric_df.loc[pred_weighted_ids, "predict"] = self.start_weight
        df_t = self.metric_df[self.metric_df["labels"] == self.metric_df["predict"]]
        df_f = self.metric_df[self.metric_df["labels"] != self.metric_df["predict"]]
        TP = df_t["labels"].sum()
        FN = df_f["labels"].sum()
        if TP + FN == 0:
            return 1
        return TP / (TP + FN)

    def f1(
        self,
        label_intervals: List[TimeInterval],
        pred_intervals: List[TimeInterval],
        reset_df=True,
    ):
        precision_score = self.precision(label_intervals, pred_intervals, reset_df)
        recall_score = self.recall(label_intervals, pred_intervals, reset_df=False)
        if np.isclose(precision_score + recall_score, 0):
            return 0
        return (2 * precision_score * recall_score) / (precision_score + recall_score)
