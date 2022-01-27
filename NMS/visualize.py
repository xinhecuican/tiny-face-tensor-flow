import sys
import os
from PIL import Image, ImageDraw, ImageFont
import json
from pathlib import Path
import numpy as np

keepAll = 0
thresh = -0.5

imgRoot = ".\\data\\widerface\\WIDER_val\\images\\"
boxDir = '.\\val_results_all\\'
#boxDir = '.\\newNMS' 
saveRoot = '.\\eval' 
allBoxFiles = []

#for root, dirs, files in os.walk(boxDir, topdown= True):
#    for file in files:
#        allBoxFiles.append(os.path.join(root, file))
files = [
    "0--Parade\\0_Parade_marchingband_1_20.txt",
    "0--Parade\\0_Parade_marchingband_1_74.txt",
    "0--Parade\\0_Parade_marchingband_1_78.txt",
    "0--Parade\\0_Parade_marchingband_1_104.txt",
    "0--Parade\\0_Parade_marchingband_1_139.txt",
    "0--Parade\\0_Parade_marchingband_1_552.txt",
    "0--Parade\\0_Parade_marchingband_1_710.txt",
    "0--Parade\\0_Parade_marchingband_1_768.txt"
]
for file in files:
    allBoxFiles.append(os.path.join(boxDir, file))

print("img Number = ", len(allBoxFiles))
for boxFile in allBoxFiles:
    imgName, imgDir = boxFile.split(os.sep)[-1].split('.')[0], boxFile.split(os.sep)[-2]
    imgPath = os.path.join(imgRoot, imgDir, imgName + '.jpg' )
    saveDir = os.path.join(saveRoot, imgDir)
    savePath = os.path.join(saveDir, imgName + '.jpg')

    if not os.path.exists(saveDir):
        os.mkdir(saveDir)

    img = Image.open(imgPath)
    allBoxes = np.loadtxt(fname=boxFile, dtype= np.int32, skiprows= 2)
    if  len(allBoxes) == 0:
        print("no box")
        img.save(savePath)
        continue

    if keepAll:
        boxes = allBoxes
    else:
        boxes = []
        for index in range(len(allBoxes)):
            if allBoxes[index][4] > thresh :
                boxes.append(allBoxes[index])

    boxes = np.array(boxes)

    boxes = boxes[:, 0:4]


    draw = ImageDraw.Draw(img)
    for box in boxes:
        x1 = box[0]
        y1 = box[1]
        w  = box[2]
        h  = box[3]
        draw.rectangle((x1, y1, x1 + w, y1 + h ), outline= 'blue', width= 5)

    print('imgsave = ', savePath)
    img.save(savePath)


