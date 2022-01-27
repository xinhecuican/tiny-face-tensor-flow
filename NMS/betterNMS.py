import pickle
from evaluate import write_results
import numpy as np
from tqdm import tqdm
import json
from matplotlib import pyplot as plt
import copy
from threading import Thread
from widerface_evaluate import evaluation
from evaluate import write_results
from multiprocessing import  Process

resultDir = 'newNMS/'
JSON_PATH = './history.json'

# 缺陷:
#   根据算法的设计，如果一个物体处于预设的重叠阈值之内，可能会导致检测不到该待检测物体。
#    即当两个目标框接近时，分数更低的框就会因为与之重叠面积过大而被删掉。
def nms(dets, thresh):
    #dets = dets[np.where(dets[:, 4] > 0)[0]]
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # order 保存排序后的下标， order[0] 对应的索引是 score 最高元素的下标 
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        # 保存 score 最高的框
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
        # 留下 overlap 小于 thresh 的部分
        order = order[inds + 1]

    return np.array(keep).astype(np.int)

def softnms(dets, thresh, method = 0, sigma = 0.5, thresh2 = 0.1):
    factor = 0.15

    dets = dets[np.argsort(dets[:,4])[::-1]]
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # order 保存排序后的下标， order[0] 对应的索引是 score 最高元素的下标 

    for index in range(0, len(dets) - 1):
        # 计算第 i 个 与 (i + 1, eln) 个的 交集
        xx1 = np.maximum(x1[index], x1[index + 1:])
        yy1 = np.maximum(y1[index], y1[index + 1:])
        xx2 = np.minimum(x2[index], x2[index + 1:])
        yy2 = np.minimum(y2[index], y2[index + 1:])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        # iou = A 交 B / A 并 B
        ovr = inter / (areas[index] + areas[index + 1:] - inter)
        inds = np.where(ovr > thresh)[0]

        if method == 0:
            # 0.6 
            scores[inds + index + 1] *= (1 - ovr[inds]) * factor
        else :
            scores[inds + index + 1] *= np.exp(- (ovr[inds] * ovr[inds]) / sigma) * factor
            
    keep = np.where(scores > thresh2)[0]
    return dets[keep]

def softnms2(dets, Nt=0.3, sigma=0.5, thresh=0.001, method=2):
    """
    py_cpu_softnms
    :param dets:   boexs 坐标矩阵 format [y1, x1, y2, x2]
    :param Nt:     iou 交叠门限
    :param sigma:  使用 gaussian 函数的方差
    :param thresh: 最后的分数门限
    :param method: 使用的方法
    :return:       留下的 boxes 的 index
    """

    # indexes concatenate boxes with the last column
    N = dets.shape[0]
    indexes = np.array([np.arange(N)])
    dets = np.concatenate((dets, indexes.T), axis=1)

    # the order of boxes coordinate is [y1,x1,y2,x2]
    y1 = dets[:, 0]
    x1 = dets[:, 1]
    y2 = dets[:, 2]
    x2 = dets[:, 3]
    scores = dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    for i in range(N):
        # intermediate parameters for later parameters exchange
        tBD = dets[i, :].copy()
        tscore = scores[i].copy()
        tarea = areas[i].copy()
        pos = i + 1

        #
        if i != N-1:
            maxscore = np.max(scores[pos:], axis=0)
            maxpos = np.argmax(scores[pos:], axis=0)
        else:
            maxscore = scores[-1]
            maxpos = 0
        if tscore < maxscore:
            dets[i, :] = dets[maxpos + i + 1, :]
            dets[maxpos + i + 1, :] = tBD
            tBD = dets[i, :]

            scores[i] = scores[maxpos + i + 1]
            scores[maxpos + i + 1] = tscore
            tscore = scores[i]

            areas[i] = areas[maxpos + i + 1]
            areas[maxpos + i + 1] = tarea
            tarea = areas[i]

        # IoU calculate
        xx1 = np.maximum(dets[i, 1], dets[pos:, 1])
        yy1 = np.maximum(dets[i, 0], dets[pos:, 0])
        xx2 = np.minimum(dets[i, 3], dets[pos:, 3])
        yy2 = np.minimum(dets[i, 2], dets[pos:, 2])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[pos:] - inter)

        # Three methods: 1.linear 2.gaussian 3.original NMS
        if method == 1:  # linear
            weight = np.ones(ovr.shape)
            weight[ovr > Nt] = weight[ovr > Nt] - ovr[ovr > Nt]
        elif method == 2:  # gaussian
            weight = np.exp(-(ovr * ovr) / sigma)
        else:  # original NMS
            weight = np.ones(ovr.shape)
            weight[ovr > Nt] = 0

        scores[pos:] = weight * scores[pos:]

    # select the boxes and keep the corresponding indexes
    #inds = dets[:, 4][scores > thresh]
    inds = np.where(scores > thresh)[0]
    keep = inds.astype(int)

    return keep

def writeJson(path, checkPoint, method, thresh, result):
    with open(path, "r", encoding="utf-8") as f:
        oldData = json.load(f)
        isDuplicated = False
        for data in oldData:
            if data['weight'] == checkPoint and data['method'] == method and data['thresh'] == thresh:
                isDuplicated = True
                return

        if not isDuplicated :
            newData = {
                "weight" : checkPoint,
                "method" : method,
                "thresh" : thresh,
                "result" : {
                    "easy": result[0],
                    "medium": result[1],
                    "hard": result[2]
                }
            }
            oldData.append(newData)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(oldData, f, indent=4)
    return 

def normScore(box):
    _max = np.max(box[:, -1])
    _min = np.min(box[:, -1])
    diff = _max - _min
    box[:, -1] = (box[:, -1] - _min) / diff
    return box 

def nmsFunc(id, dets, thresh, thresh2):
    for elm in tqdm(dets, desc= str(id)):
        elmCopy = copy.deepcopy(elm)
        # '0--Parade/0_Parade_marchingband_1_465.jpg'
        fileName = elmCopy[0]
        box = elmCopy[1]

        # 只要 大于 0 的 box
        box = box[np.where(box[:, 4:] > -1 )[0]]
        if len(box) == 0 :
            write_results(box, fileName, 'val', resultDir)
            continue

        box = normScore(box)
        ##keep = nms(box, thresh)
        box = softnms(box, thresh, sigma= 0.5, method= 1, thresh2= thresh2)
        #keep = softnms2(box, Nt= thresh, thresh= thresh2)
        #box = box[keep]
        write_results(box, fileName, 'val', resultDir)
    return 

def main(dets):
    # thresh 从a 到b，分析 AP
    #METHOD = 'soft_nms_linear'
    METHOD = 'softnms1'
    WEIGHT = 'checkpoint12.pth'
    EVAL_ONLY = 0
    PROCESS_NUM = 6
    #normScore(dets)
    if EVAL_ONLY == 0 :
        # Python 不能实际上并行 ！！！
        # 多线程是 逻辑上 的 ！！！

        #for thresh in np.around(np.arange(0.05, 0.5, 0.05), 3):
        dataPerProcess = int(len(dets) / PROCESS_NUM)
        partition = dataPerProcess * np.array(range(1, PROCESS_NUM + 1))
        partition[PROCESS_NUM - 1] = len(dets)
        thresh = 0.3 

        #for thresh2 in [0.005, 0.01, 0.03, 0.08, 0.1, 0.2]:
        #    print("now thresh2 = ",thresh2)
        #    # 使用NMS
        #    processes = []
        #    for i in range(PROCESS_NUM):
        #        if i == 0 :
        #            processes.append(Process(target= nmsFunc, args= (0, dets[0: partition[i]], thresh, thresh2) ))
        #            continue
        #        processes.append(Process(target= nmsFunc, args= (i, dets[partition[i - 1]: partition[i]], thresh, thresh2) ))
        #    for thread in processes:
        #        thread.start()
        #    for thread in processes:
        #        thread.join()
        #    print("finish ")
        for thresh2 in [ 0.05, 0.1, 0.2, 0.3 ]:
            print("shresh2 = ", thresh2)
            nmsFunc(1, dets, thresh, thresh2= thresh2)
        # 进行evaluate，得到AP
            AP = evaluation(resultDir, './widerface_evaluate/ground_truth')
        # 写入结果
            writeJson(JSON_PATH, WEIGHT, METHOD + ' 0.2 gauss' +  str(thresh2), thresh, AP)

    else :
        AP = evaluation(resultDir, './widerface_evaluate/ground_truth')


def visualize(path):
    with open(path, "r", encoding="utf-8") as f:
        oldData = json.load(f)
    nms = []
    for elm in oldData:
        if elm['method'] == 'nms':
            nms.append((elm['thresh'], elm['result']))
    nms.sort(key= lambda x: x[0])
    xaxis = []
    yaxis = []
    for elm in nms:
        xaxis.append(elm[0])
        yaxis.append(elm[1]['easy'])
    plt.plot(xaxis, yaxis)
    plt.show()

if __name__ == "__main__":
    f = open('beforeNMS.pkl', 'rb')
    dets = pickle.loads(f.read())
    f.close()
    print("data loaded , len = ", len(dets))


    #writeJson(JSON_PATH, 'checkpoint12.pth', 'nms', 0.4, [1,1,1])
    main(dets)
    #visualize(path=JSON_PATH)


