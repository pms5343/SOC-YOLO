
#%%
import torch
import numpy as np
import cv2
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords
from utils.plots import Annotator

# 탐지모델
MODEL_PATH = '/home/pms5343/python/yolov5-original/runs/train/SOC-withoutTR/weights/best.pt'

img_size = 640
conf_thres = 0.4  # confidence threshold
iou_thres = 0.45  # NMS IOU threshold
max_det = 1000  # maximum detections per image
classes = None  # filter by class
agnostic_nms = False  # class-agnostic NMS

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
ckpt = torch.load(MODEL_PATH, map_location=device)
model = ckpt['ema' if ckpt.get('ema') else 'model'].float().fuse().eval()
class_names = ['Outriggers', 'Co-worker', 'Helmet', 'Hinge', 'Ladder', 'Other-worker', 'Hook', 'Unstable', 'Worker'] # model.names

stride = int(model.stride.max())
colors = ((50, 50, 50), (0, 0, 255), (0, 255, 0)) # (gray, red, green)
#%%
# 모델
CONFIDENCE = 0.4
THRESHOLD = 0.3
#LABELS = ['fdsf','sf']

#net = cv2.dnn.readNetFromDarknet('models/yolov4-ANPR.cfg', 'models/yolov4-ANPR.weights')

# 동영상 로드
cap = cv2.VideoCapture('/home/pms5343/python/yolov5-original/data/ladder/test/images/%1d.jpg')

#cap = cv2.imread('/home/pms5343/python/yolov5-original/data/ladder/test/images/Test_001.jpg')

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
out = cv2.VideoWriter('/home/pms5343/python/yolov5-original/data/output.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
#out = cv2.imwrite('/home/pms5343/python/yolov5-original/data/output.png', cap)
#%%
while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break

    H, W, _ = img.shape

    # preprocess
    img_input = letterbox(img, img_size, stride=stride)[0]
    img_input = img_input.transpose((2, 0, 1))[::-1]
    img_input = np.ascontiguousarray(img_input)
    img_input = torch.from_numpy(img_input).to(device)
    img_input = img_input.float()
    img_input /= 255.
    img_input = img_input.unsqueeze(0)

    # inference 
    pred = model(img_input, augment=False, visualize=False)[0]

    # postprocess
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)[0]

    pred = pred.cpu().numpy()

    pred[:, :4] = scale_coords(img_input.shape[2:], pred[:, :4], img.shape).round()


    # Visualize
    annotator = Annotator(img.copy(), line_width=3, example=str(class_names), font='/home/pms5343/python/Ladder-detection-yolov5-master/data/malgun.ttf') #한글로 할때
    #annotator = Annotator(img.copy(), line_width=3, example=str(class_names))
    cw_x1, cw_x2 = None, None # 좌측(cw_x1), 우측(cw_x2) 좌표

    for p in pred:
        class_name = class_names[int(p[5])]
        x, y, w, h = p[:4]

        annotator.box_label([x, y, w, h], '%s %d' % (class_name, float(p[4]) * 100), color=colors[int(p[5])])
        if class_name == 'Unstable':
            alert_text = '[!Unstable]'
            color = (255, 0, 0)
        annotator.box_label([x, y, w, h], '%s %d' % (alert_text+class_name), color=color)
    
    result_img = annotator.result()
    print(result_img)
    cv2.imshow('result', result_img)
    out.write(result_img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
out.release()

# %%
