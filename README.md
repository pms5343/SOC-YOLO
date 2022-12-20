# SOC-YOLO: Small and Overlapping Worker Detection at Construction Sites

This is the implementation of our Automation in Construction research paper:
> Minsoo Park, Dai Quoc Tran, Jinyeong Bak, Seunghee Park - SOC-YOLO: Small and Overlapping Worker Detection at Construction Sites

## Installation 
This code is based on YOLOv5. Please install the code according to the  [YOLOv5 tutorial](https://github.com/ultralytics/yolov5) first.

## SOC-YOLO Implementation
The updated code for implementing SOC-YOLO is displayed below:
1. Updated [DIoU-NMS](https://ojs.aaai.org/index.php/AAAI/article/view/6999) in [general.py](utils/general.py)
2. Updated DIoU loss function in [loss.py](utils/loss.py)
3. P2:Feature-level expansion is added in [P2.yaml](models/P2.yaml)
4. [SoftPool](https://doi.org/10.48550/arXiv.2101.00440) use [softpool.py](utils/softpool.py) and modify SPPF in [common.py](models/common.py)
5. We modify and add [Weighted-tiplet attetion](https://doi.org/10.48550/arXiv.2010.03045) in [common.py](models/common.py) with variable hyper parameters alpha, beta, gamma, and advanced yaml is added in [P2_Triple.yaml](models/P2_Triple.yaml)

## Referens
- https://github.com/ultralytics/yolov5
- Zheng, Zhaohui, et al. "Distance-IoU loss: Faster and better learning for bounding box regression." Proceedings of the AAAI conference on artificial intelligence. Vol. 34. No. 07. 2020.
- Stergiou, Alexandros, Ronald Poppe, and Grigorios Kalliatakis. "Refining activation downsampling with SoftPool." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2021.
- Misra, Diganta, et al. "Rotate to attend: Convolutional triplet attention module." Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision. 2021.
