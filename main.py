"""Find vehicles in specified area."""

import cv2
from loguru import logger
from pathlib import Path
from typing import Optional
from tqdm import tqdm
import os
import numpy as np
from warnings import warn
from shapely.geometry import Point, box
from shapely.geometry.polygon import Polygon
from ultralytics import YOLO
import json
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

from utils import Bbox, TimeInterval, read_polygon, bounding_rectangle, transform_polygon


CLASSES_VEHICLES = {
    2: "car",
    4: "airplane",
    5: "bus",
    6: "train",
    7: "truck",  
}
MODEL_NAME = "yolov8n.pt"
IMGSZ = 1024
CONFIDENCE_DETECT = 0.05
IOU_NMS = 0.5
IOU_AREAS_THRESH_LOW = 0.1
IOU_AREAS_THRESH_HIGH = 0.1


class VideoProcesser:
    def __init__(self,
                 video_path: str,
                 output_path: str,
                 polygon_path: str,
                 save_folder: Optional[Path] = None,
                 save_frame: bool = False
                 ):
        ## Video Caption
        self.videocap = cv2.VideoCapture(video_path)
        if  self.videocap.isOpened() == False:
            raise Exception("Video is broken")
        self.num_frames = None
        self.shape = None
        self.fps = None       
        self.save_frame = save_frame
        self.save_path = None
        self.video_path = Path(video_path)
        self.video_info()
        if save_folder is not None:
            self.save_path = save_folder / self.video_path.stem
        if not os.path.exists(str(self.save_path)):
            os.makedirs(str(self.save_path))
        ## Output folder
        if not os.path.exists(output_path):
            out_path = Path(output_path)
            os.makedirs(out_path.parents[0])
        self.output_path = output_path
        ## Model
        self.model = YOLO(MODEL_NAME)
        ## Polygon
        polygon_dict = read_polygon(polygon_path)
        self.polygon = polygon_dict[self.video_path.name]
        self.polygon_rect = Bbox(
            points=bounding_rectangle(self.polygon),
            frame_dim=self.shape,
        )
        self.transformed_polygon = transform_polygon(self.polygon, self.polygon_rect)
        ## Results
        self.pred_intervals = None
        
    def video_info(self):
        self.fps = self.videocap.get(cv2.CAP_PROP_FPS)
        width = int(self.videocap.get(3))
        height = int(self.videocap.get(4))
        self.shape = (height, width)
        self.num_frames = int(self.videocap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(
            f"""
            --Info--
            Video: {str(self.video_path)}
            FPS: {self.fps}
            Shape: {self.shape}
            Number of frames: {self.num_frames}
            """
        )

    def detect(self, img: np.ndarray):
        img = img[self.polygon_rect.y_min : self.polygon_rect.y_max, self.polygon_rect.x_min : self.polygon_rect.x_max]
        results = self.model.predict(
            img,
            conf = CONFIDENCE_DETECT,
            verbose=False,
            iou = IOU_NMS,
            imgsz = IMGSZ,
        )

        predictions = results[0]
        boxes = predictions.boxes.xyxy.numpy()
        categories = predictions.boxes.cls.numpy()
        boxes_filtered = []
        for category, box_ in zip(categories, boxes):
            if category in CLASSES_VEHICLES:
                boxes_filtered.append(box_)
        if len(boxes_filtered) > 0:
            return np.vstack(boxes_filtered)
        return []
    
    def save_results(self):
        intervals_list = [interval.to_list() for interval in self.pred_intervals]
        res_dict = {
            self.video_path.name: intervals_list
        }
        with open(self.output_path, "w") as f:
            json.dump(res_dict, f)
        
    def process_frames(self):
        num_frames_skipped = 0
        polygon_shapely = Polygon(self.transformed_polygon)
        intervals_predict = []
        for n_frame in tqdm(range(
            self.num_frames
        ), desc = f"{self.video_path.name}"):
            ret, frame = self.videocap.read()
            if not ret:
                warn("Frame is not loaded")
                num_frames_skipped += 1
                continue
            if self.save_frame:
                cv2.imwrite(str(self.save_path / f"{n_frame}.png"), frame)

            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes_detect = self.detect(img)
            if len(boxes_detect) == 0:
                continue
            for _, bbox in enumerate(boxes_detect):
                bbox_ = Bbox(bbox, img.shape[:2])
                bbox_shapely = box(*bbox_.list)
                iou_areas = polygon_shapely.intersection(bbox_shapely).area / bbox_shapely.area
                right_max_point = Point(*bbox_.point_max)
                contains_point = polygon_shapely.contains(right_max_point)
                iou_areas_thresh = IOU_AREAS_THRESH_LOW if contains_point else IOU_AREAS_THRESH_HIGH
                if iou_areas > iou_areas_thresh:
                    if len(intervals_predict) == 0:
                        intervals_predict.append(TimeInterval([n_frame, None]))
                        break
                    if intervals_predict[-1].prev_num + 1 != n_frame:
                        intervals_predict[-1].max = intervals_predict[-1].prev_num
                        intervals_predict.append(TimeInterval([n_frame, None]))
                    else:
                        intervals_predict[-1].prev_num = intervals_predict[-1].prev_num + 1 
                    break

        logger.info(f"Frames skipped: {num_frames_skipped}")
        self.num_frames -= num_frames_skipped
        if num_frames_skipped > 0:
            logger.info(f"Number of frames processed: {self.num_frames}")
        if self.num_frames < 1:
            raise Exception("Video is empty")
        
        if len(intervals_predict) > 0:
            if intervals_predict[-1].max is None:
                intervals_predict[-1].max = intervals_predict[-1].prev_num

            if intervals_predict[-1].max + 1 < self.num_frames:
                intervals_predict[-1].max += 1
        
        cv2.destroyAllWindows()
        logger.info(f"Found {len(intervals_predict)} intervals with vehicles.")
        num_frames_vehicles = sum([x.num_frames() for x in intervals_predict])
        logger.info(f"Total number of frames with vehicles: {num_frames_vehicles}/{self.num_frames}")
        self.pred_intervals = intervals_predict
        return intervals_predict
    

def parse_arguments():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--video_path",
        type=str,
        help="Path to video to process",
    )
    parser.add_argument(
        "--polygon_path",
        type=str,
        help="Path to polygon JSON file",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="Path for results JSON file",
    )
    args = vars(parser.parse_args())
    return args
    

if __name__ == "__main__":
    args = parse_arguments()
    video_processer = VideoProcesser(
        video_path=args["video_path"],
        output_path=args["output_path"],
        polygon_path=args["polygon_path"],
    )
    pred_intervals = video_processer.process_frames()
    video_processer.save_results()