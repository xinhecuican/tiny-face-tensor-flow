import numpy as np
import torch


def nms(dets, thresh):
    """
    Courtesy of Ross Girshick
    [https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/nms/py_cpu_nms.py]
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return np.array(keep).astype(np.int)


def nms2(boxes, scores, overlap=0.5, top_k=2000):
    """
    输入:
        boxes: 存储一个图片的所有预测框。[num_positive,4].
        scores:置信度。如果为多分类则需要将nms函数套在一个循环内。[num_positive].
        overlap: nms抑制时iou的阈值.
        top_k: 先选取置信度前top_k个框再进行nms.
    返回:
        nms后剩余预测框的索引.
    """

    keep = scores.new(scores.size(0)).zero_().long()
    # 保存留下来的box的索引 [num_positive]
    # 函数new(): 构建一个有相同数据类型的tensor

    # 如果输入box为空则返回空Tensor
    if boxes.numel() == 0:
        return keep

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)  # 并行化计算所有框的面积
    v, idx = scores.sort(0)  # 升序排序
    idx = idx[-top_k:]  # 前top-k的索引，从小到大
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()

    count = 0
    while idx.numel() > 0:
        i = idx[-1]  # 目前最大score对应的索引
        keep[count] = i  # 存储在keep中
        count += 1
        if idx.size(0) == 1:  # 跳出循环条件：box被筛选完了
            break
        idx = idx[:-1]  # 去掉最后一个

        # 剩下boxes的信息存储在xx，yy中
        torch.index_select(x1, 0, idx, out=xx1)
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2)

        # 计算当前最大置信框与其他剩余框的交集，不知道clamp的同学确实容易被误导
        xx1 = torch.clamp(xx1, min=x1[i])  # max(x1,xx1)
        yy1 = torch.clamp(yy1, min=y1[i])  # max(y1,yy1)
        xx2 = torch.clamp(xx2, max=x2[i])  # min(x2,xx2)
        yy2 = torch.clamp(yy2, max=y2[i])  # min(y2,yy2)
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        w = xx2 - xx1  # w=min(x2,xx2)−max(x1,xx1)
        h = yy2 - yy1  # h=min(y2,yy2)−max(y1,yy1)
        w = torch.clamp(w, min=0.0)  # max(w,0)
        h = torch.clamp(h, min=0.0)  # max(h,0)
        inter = w * h

        # 计算当前最大置信框与其他剩余框的IOU
        # IoU = i / (area(a) + area(b) - i)
        rem_areas = torch.index_select(area, 0, idx)  # 剩余的框的面积
        union = rem_areas + area[i] - inter  # 并集
        IoU = inter / union  # 计算iou

        # 选出IoU <= overlap的boxes(注意le函数的使用)
        idx = idx[IoU.le(overlap)]
    return keep, count
    # [num_remain], num_remain
