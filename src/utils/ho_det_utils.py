import numpy as np
from tqdm import trange

def calculate_center(bb):
    x,y,w,h = bb
    return [x + w/2, y+h/2]

def filter_object(obj_dets, hand_dets, thresh=0.8):
    '''
    if a hand is holding portable objects, get the closet obj id
    '''
    if obj_dets is None or hand_dets is None:
        return None
    object_cc_list = []
    for j in range(obj_dets.shape[0]):
        object_cc_list.append(calculate_center(obj_dets[j,:4]))
    object_cc_list = np.array(object_cc_list)
    img_obj_id = []
    img_hand_id = []
    for i in range(hand_dets.shape[0]):
        if hand_dets[i, 5] != 3 and hand_dets[i, 4] < thresh: #state 3:'Portable Object'
            continue
        hand_cc = np.array(calculate_center(hand_dets[i,:4]))
        point_cc = np.array([(hand_cc[0]+hand_dets[i,6]*10000*hand_dets[i,7]), (hand_cc[1]+hand_dets[i,6]*10000*hand_dets[i,8])])
        dist = np.sum((object_cc_list - point_cc)**2,axis=1)
        dist_min = np.argmin(dist)
        img_obj_id.append(dist_min)
        img_hand_id.append(i)
    
    if len(img_obj_id) == 0:
        return None
    else:
        return img_obj_id, img_hand_id

def parse_det(det_res):
    # score_max = np.argmax(det_res[:, 4])
    res = {
        "bbox": det_res[:, :4],
        "score": det_res[:, 4],
        "is_right": det_res[:, -1], #(0: l, 1: r)
        "state": det_res[:, 5] #{0:'No Contact', 1:'Self Contact', 2:'Another Person', 3:'Portable Object', 4:'Stationary Object'}
    }
    return res

def intersect_box(*bboxes):
    bboxes = np.array(bboxes) # x,y,w,h = box
    x1 = bboxes[:, 0].max()
    y1 = bboxes[:, 1].max()
    x2 = (bboxes[:, 0] + bboxes[:, 2]).min()
    y2 = (bboxes[:, 1] + bboxes[:, 3]).min()
    
    return np.array([x1, y1, x2-x1, y2-y1])

def union_box(*bboxes):
    bboxes = np.array(bboxes)
    x1 = bboxes[:, 0].min()
    y1 = bboxes[:, 1].min()
    x2 = (bboxes[:, 0] + bboxes[:, 2]).max()
    y2 = (bboxes[:, 1] + bboxes[:, 3]).max()
    
    return np.array([x1, y1, x2-x1, y2-y1])

def compute_area(box):
    x,y,w,h = box
    return w * h

def compute_iou(box_a, box_b):
    # Ensure the boxes are numpy arrays
    box_a = np.array(box_a)
    box_b = np.array(box_b)
    
    inter_box = intersect_box([box_a, box_b])
    union_box = union_box([box_a, box_b])
    
    intersection_area = compute_area(inter_box)
    union_area = compute_area(union_box)
    
    union_area = compute_area(box_a) + compute_area(box_b) - intersection_area
    iou = intersection_area / union_area if union_area > 0 else 0

    return iou

def mask_to_bbox(mask, rate=2):
    """
    Args:
        mask (H, W)
    """
    
    H, W = mask.shape
    h_idx, w_idx = np.where(mask > 0)
    if len(h_idx) == 0:
        return np.array([0, 0, 0, 0])

    y1, y2 = h_idx.min(), h_idx.max()
    x1, x2 = w_idx.min(), w_idx.max()
    w = (x2 - x1) * rate
    h = (y2 - y1) * rate
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    box = [cx-w/2, cy-h/2, w, h]
    
    
    max_box = [0, 0, W, H]

    return intersect_box(max_box, box)