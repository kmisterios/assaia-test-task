# Test assignment for Assaia

## Run code
 <code>./run.sh \<video_path\> \<polygon_path\> \<output_path\></code>

 ## Description
 ### Method
 My method consisits of several major blocks:

 1. Crop frame with bounding rectangle created from the polygon boundary.
 2. Detect vehicles with [YoloV8](https://github.com/ultralytics/ultralytics).

    * I used <code>yolov8n</code> pretarined model trained on COCO. I selected only these classes from the $80$ COCO classes: <code>["car", "airplane", "bus", "truck", "train"]</code>
 3. Apply heuristics to check if the detected object id in the polygon boundary.

    * Check if bounding box intersects polygon region and apply threshold on IOU. If IOU is less than threshold, then it could be the situation that intersection is empty region of bbox, because objects can be not square. I don't count such detections.
    * There are cases when vehicle is standing in front of the boundary, but it's bounding box intersect the region. To exclude this case, I check, if lower right pint of bbox is in a region. If it is not, I increase IOU significantly.
    * I bbox is not filtered, then we have a detection.

I've chosen YoloV8 trained on COCO, because:
* It works real time which is required for using on cameras in an airport.
* It has convibient interface 
* It has great quality of prediction of the box
* It has big number of different vehicles classes.

### Limitations
The main limitations are:
1. I use model pretrained on standard dataset. Vehicles in the airport can be unusual. To overcome this limitations one can fine-tune the model on the dataset of different classes of vehicles from the airports.
2. Heuristics not always work fine. Also, there are boundaries wich are far away or have deep perspective. My algorythm can make errors in such cases. Here the perspective transforms and masking the area outside the given polygon can work out.
3. Sometimes detector losses the objects. Here tracking algorythm can be added to solve this problem. I was exxperimenting with SORT and YOLO trackers. They were giving results. But then I found videos with $2-3$ FPS in the dataset. For such cases tracker give additional error. So, tracker can be used depending on the FPS of the video.

### Metrics
I've chosen <code>precision</code>, <code>recall</code>, <code>f1-score</code> metrics. We basically solve the task, if there is vehicle in boundary area in a given frame. So, binary classification. That's why I create vector with length equal to the number of frames in the video, where "vehicle detected" in frame - $1$, othervise - $0$. Then I compute standard metrics with $1$. Since it is important to detect vehicle as fast as possible, I give weight of $2$ to the first $20$% of frames in the interval. In this case when we detect the vehicle later, the metric will be lower.

### Selecting the parameters
* During parameter selection I was using <code>f1-score</code> to estimate quality with current parameter set.
* Since I have no training, I slected parameters on train set and tested on the test set.
* To divide the dataset on train and test I used Stratidied technique to ensure that both sets have similar distributuion of differen videos. To achieve this I computed the ratio of "vechicle" frames for all the videos. Then I divided videos on 3 categories:

    * ratio less $0.3$
    * ratio between $0.3$ and $0.5$
    * ration is higher than $0.5$

The metric computes on each video of the test can be found in the [file](metrics/metrics_test.csv). The overall metric on the test set is $0.782$, which is quite good, however it was lower ($0.510$) on the train set. Maybe lucky me to get simple videos in the test set.

