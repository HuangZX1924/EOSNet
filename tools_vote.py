import json
import numpy as np
from pathlib import Path
from ultralytics.yolo.utils.metrics import ap_per_class

# 根据生成的初始json，过滤多余结果，每帧只保留最大值
def translate(box):
    # 在val保存中有一步骤是将x_c，y_c，w，h的结果保存为了x1,y1,w,h。因此这里再变回来
    x1, y1, w, h = box
    xc = x1 + w / 2
    yc = y1 + h / 2
    return [xc, yc, w, h]

def fliter(ori_json):
    with open(ori_json, 'r') as f:
        ori_data = json.load(f)
    sorted_data = sorted(ori_data, key=lambda x: (x['image_name'], -x['score_t3']))    # 先按name排序，name相同则按分数从大到小排序，确保分数最大的在每个子序列的开端
    jdict = []
    for ann in sorted_data:
        if len(jdict) == 0:
            jdict.append({
                'image_name': ann['image_name'],
                'bbox_t1': translate(ann['bbox_t1']),
                'score_t1': ann['score_t1'],
                'bbox_t2': translate(ann['bbox_t2']),
                'score_t2': ann['score_t2'],
                'bbox_t3': translate(ann['bbox_t3']),
                'score_t3': ann['score_t3']
            })
        elif ann['image_name'] != jdict[-1]['image_name']:
            jdict.append({
                'image_name': ann['image_name'],
                'bbox_t1': translate(ann['bbox_t1']),
                'score_t1': ann['score_t1'],
                'bbox_t2': translate(ann['bbox_t2']),
                'score_t2': ann['score_t2'],
                'bbox_t3': translate(ann['bbox_t3']),
                'score_t3': ann['score_t3']
            })
        else:
            continue
    with open(ori_json.replace('predictions.json','fliter.json'), 'w') as f:
        json.dump(jdict, f)

# 跟踪过滤后的json文件，通过三帧投票创建新的box
# 投票策略，这里先考虑第一种，创建一个长度为7的list，分别存放了C、1、2、CU1、CU2、1U2、CU1U2的结果，这个结果包含交集框的位置和新得分
def xyxy2xywh(box):
    x1, y1, x2, y2 = box
    xc = (x1 + x2) / 2
    yc = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return [xc, yc, w, h]

def xywh2xyxy(box):
    xc, yc, w, h = box
    x1 = xc - w/2
    x2 = xc + w/2
    y1 = yc - h/2
    y2 = yc + h/2
    return [x1, y1, x2, y2]

def overlap(boxA, boxB):
    # box的type为xywh
    boxA = xywh2xyxy(boxA)
    boxB = xywh2xyxy(boxB)
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # interArea = max(0, xB - xA) * max(0, yB - yA)
    interArea = IOU(boxA,boxB)
    box = [xA, yA, xB, yB]
    box = xyxy2xywh(box)
    if boxA == boxB:
        same = 1
    else:
        same = 0
    return box, interArea, same

def cal_score(score1, score2):
    return 1-(1-score1)*(1-score2)

def vote(fliter_json, iou_th):
    with open(fliter_json, 'r') as f:
        fliter_data = json.load(f)
    jdict = []
    length = len(fliter_data)
    for i in range(length):
        new_boxes = []
        id = fliter_data[i]['image_name'].split('/')[-1].split('.jpg')[0]
        tc = int(id)
        t1 = str(tc + 2).zfill(len(id))
        t2 = str(tc + 4).zfill(len(id))
        tc_name = fliter_data[i]['image_name']
        t1_name = tc_name.replace(id + '.jpg', t1 + '.jpg')
        t2_name = tc_name.replace(id + '.jpg', t2 + '.jpg')

        box_tc = fliter_data[i]['bbox_t3']
        score_tc = fliter_data[i]['score_t3']
        new_boxes.append({
            'image_name': fliter_data[i]['image_name'],
            'bbox': box_tc,
            'w': box_tc[2],
            'h': box_tc[3],
            'score': score_tc
        })

        if i+2<length and fliter_data[i+2]['image_name'] == t1_name:
            box_t1 = fliter_data[i+2]['bbox_t2']    # 注意这里是t2，因为现在顺序是tc、t1、t2，在label创建的时候是t1、t2、tc
            score_t1 = fliter_data[i+2]['score_t2']
        else:
            box_t1 = box_tc
            score_t1 = score_tc
        new_boxes.append({
            'image_name': fliter_data[i]['image_name'],
            'bbox': box_t1,
            'w': box_t1[2],
            'h': box_t1[3],
            'score': score_t1
        })

        if i+4<length and fliter_data[i+4]['image_name'] == t2_name:
            box_t2 = fliter_data[i+4]['bbox_t1']
            score_t2 = fliter_data[i+4]['score_t1']
        else:
            box_t2 = box_tc
            score_t2 = score_tc
        new_boxes.append({
            'image_name': fliter_data[i]['image_name'],
            'bbox': box_t2,
            'w': box_t2[2],
            'h': box_t2[3],
            'score': score_t2
        })

        box_c1, area_c1, same_c1 = overlap(box_tc, box_t1)
        if area_c1 < iou_th:
            score_c1 = 0
        elif same_c1 == 1:
            score_c1 = score_tc
        else:
            score_c1 = cal_score(score_tc, score_t1)
        new_boxes.append({
            'image_name': fliter_data[i]['image_name'],
            'bbox': box_c1,
            'w': (box_tc[2] + box_t1[2]) / 2,
            'h': (box_tc[3] + box_t1[3]) / 2,
            'score': score_c1
        })

        box_c2, area_c2, same_c2 = overlap(box_tc, box_t2)
        if area_c2 < iou_th:
            score_c2 = 0
        elif same_c2 == 1:
            score_c2 = score_tc
        else:
            score_c2 = cal_score(score_tc, score_t2)
        new_boxes.append({
            'image_name': fliter_data[i]['image_name'],
            'bbox': box_c2,
            'w': (box_tc[2] + box_t2[2]) / 2,
            'h': (box_tc[3] + box_t2[3]) / 2,
            'score': score_c2
        })

        box_12, area_12, same_12 = overlap(box_t1, box_t2)
        if area_12 < iou_th:
            score_12 = 0
        elif same_12 == 1:
            score_12 = score_t1
        else:
            score_12 = cal_score(score_t1, score_t2)
        new_boxes.append({
            'image_name': fliter_data[i]['image_name'],
            'bbox': box_12,
            'w': (box_t1[2] + box_t2[2]) / 2,
            'h': (box_t1[3] + box_t2[3]) / 2,
            'score': score_12
        })

        box_c12, area_c12, same_c12 = overlap(box_c1, box_t2)
        if area_c12 < iou_th:
            score_c12 = 0
        elif same_c12 == 1:
            score_c12 = score_c1
        else:
            score_c12 = cal_score(score_c1, score_t2)
        new_boxes.append({
            'image_name': fliter_data[i]['image_name'],
            'bbox': box_c12,
            'w': (box_tc[2] + box_t1[2] + box_t2[2]) / 3,
            'h': (box_tc[3] + box_t1[3] + box_t2[3]) / 3,
            'score': score_c12
        })

        new_bbox = sorted(new_boxes, key=lambda x: (-x['score']))[0]
        # new_bbox = new_boxes[0]

        jdict.append({
            'image_name': new_bbox['image_name'],
            'bbox': [new_bbox['bbox'][0], new_bbox['bbox'][1], new_bbox['w'], new_bbox['h']],
            'score': new_bbox['score']
        })

    with open(ori_json.replace('predictions.json','vote.json'), 'w') as f:
        json.dump(jdict, f)

# 根据生成的json文件，利用yolo/utils/metrics.py计算指标
def IOU(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    eps = 1e-7
    iou = interArea / float(boxAArea + boxBArea - interArea + eps)
    return iou

def cal_metrics(vote_json):
    with open(vote_json, 'r') as f:
        vote_data = json.load(f)
    niou = 10
    iouv = np.linspace(0.5, 0.95, 10)
    correct = np.zeros((len(vote_data), niou)).astype(bool)
    image_size = [256.0, 256.0, 256.0, 256.0]
    iou_mul = []
    score_mul = []
    for ann in vote_data:
        if ann['image_name'].split('/') [-2]== 'seq12':
            continue
        pre_bbox = ann['bbox']
        gt_name = gt_path  + image_type + ann['image_name'].split(image_type)[-1].split('.jpg')[0] + '.txt'
        with open(gt_name, 'r', encoding='utf-8') as f:
            gt_txt = f.readline().strip().split(' ')
        gt_bbox = list(map(float,gt_txt))[9:13] 
        gt_bbox = np.multiply(gt_bbox, image_size)
        iou_sin = IOU(xywh2xyxy(pre_bbox),xywh2xyxy(gt_bbox))
        iou_mul.append(iou_sin)
        score_mul.append(ann['score'])
    iou = np.array(iou_mul,dtype='float').T.reshape(-1, 1)
    score = np.array(score_mul,dtype='float').T
    for i in range(niou):
        x = np.where(iou >= iouv[i])
        correct[np.stack(x, 1)[:,0].astype(int), i] = True
    cls_tuple = {}
    cls_tuple[0]='uav'
    tp, fp, p, r, f1, ap, unique_classes = ap_per_class(tp=correct, conf=score, 
                                                        pred_cls=np.zeros(len(iou_mul)), target_cls=np.zeros(len(iou_mul)),
                                                        plot=True, on_plot=None, save_dir=Path(ori_json.split('predictions.json')[0]),
                                                        names=cls_tuple, prefix='vote_')
    return p, r, ap[:, 0], ap.mean()


gt_path = '/home1/users/tangyuan01/YOLOC/datasets/AntiUAV310/labels/'
# ori_json = '/home1/users/tangyuan01/YOLOC/runs/detect/train5/predictions.json'
ori_json = './predictions.json'
fliter_json = ori_json.replace('predictions.json','fliter.json')
vote_json = ori_json.replace('predictions.json','vote.json')
image_type = 'test'

iou_ths = np.linspace(0.1, 0.9, 17)
# iou_ths = [0.5]

if __name__ == '__main__':
    fliter(ori_json)
    for iou_th in iou_ths:
        vote(fliter_json, iou_th)
        Pre, Rec, mAP50, mAP50_95 = cal_metrics(vote_json)
        print("Pre: %f, Rec: %f, mAP50: %f, mAP50_95: %f \n" %(Pre, Rec, mAP50, mAP50_95))