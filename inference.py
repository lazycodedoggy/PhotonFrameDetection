import sys
import os
import re
import logging
import getopt

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models

from PIL import Image

from PhotonDataset import transform, PhotonDataset

logging.getLogger("PIL").setLevel(logging.ERROR)
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging

MODEL_BASE_PATH='./models'

JIAOZHI_MODEL_PATH=MODEL_BASE_PATH + "/one/"
COMPOSE_MODEL_PATH=MODEL_BASE_PATH + "/all/"

INSPECT2ID = {
    "Stratum_corneum" : 0,
    "DEJunction" : 1,
    "ELCOR" : 2,
}

IDS2INSPECT = {
    0 : "Stratum_corneum",
    1 : "DEJunction",
    2 : "ELCOR",
}


THRESHOLDS=[0.7, 0.7, 0.7]
IMAGE_SIZE=224

device = 'cuda' if torch.cuda.is_available() else 'cpu'
logger.info(f"Using {device} device")

def listModelVer(subDir):
    vers = []
    for tmpFile in os.listdir(subDir):
        if re.match("[a-zA-Z]+-\d+\.pth", tmpFile):
            _, saveTime, _ = re.split("-|\.", tmpFile)
            vers.append((tmpFile, int(saveTime)))
    vers.sort(key=lambda xx : xx[1], reverse=True)
    return [x for x, _ in vers]

def loadCheckPoint(model, checkpointPath):
    oldVers = listModelVer(checkpointPath)
    if len(oldVers) > 0:
        fullPath = "%s/%s"%(checkpointPath, oldVers[0])
        logger.info("@szh:load model and optimizer state from file: {}".format(fullPath))
        checkpointDict = torch.load(fullPath, map_location=device)

        if hasattr(model, "module"):
            model.module.load_state_dict(checkpointDict["model"])
        else:
            model.load_state_dict(checkpointDict["model"])
    else:
        logger.error(f"@szh: loading model {checkpointPath} fails!")

def pickLongMatch(hitDetails):
    midResults = [[] for _ in range(len(INSPECT2ID))]
    for argSeq in range(len(INSPECT2ID)):
        start, end = -1, -2
        for seq, value in zip(range(len(hitDetails)), hitDetails[:, argSeq]):
            if value > 0:
                if start < 0:
                    start, end = seq, seq
                else:
                    end = seq
            else:
                if start >= 0:
                    midResults[argSeq].append((start + 1, end + 1))
                    start, end = -1, -2
        if start > 0:
            midResults[argSeq].append((start + 1, seq + 1))

    results = [(0, 0) for _ in range(len(INSPECT2ID))]
    for argSeq in range(len(INSPECT2ID)):
        for start, end in midResults[argSeq]:
            if (end - start) > (results[argSeq][1] - results[argSeq][0]):
                results[argSeq] = (start, end)
    
    return results

def prepareModels():
    vgg16Compose = models.vgg16(weights=torchvision.models.VGG16_Weights.DEFAULT)
    num_classes = len(IDS2INSPECT)
    vgg16Compose.classifier[6] = nn.Linear(4096, num_classes)
    loadCheckPoint(vgg16Compose, COMPOSE_MODEL_PATH)
    vgg16Compose.to(device)

    vgg16Jiaozhi = models.vgg16(weights=torchvision.models.VGG16_Weights.DEFAULT)
    vgg16Jiaozhi.classifier[6] = nn.Linear(4096, 1)
    loadCheckPoint(vgg16Jiaozhi, JIAOZHI_MODEL_PATH)
    vgg16Jiaozhi.to(device)

    return [vgg16Compose, vgg16Jiaozhi]

def inference(models, tifFiles):
    sigMoid = nn.Sigmoid()
    imgs = [Image.open(x) for x in tifFiles]
    if len(imgs) > 1 and imgs[0].n_frames != imgs[1].n_frames:
        logger.error(f"@szh:The frame numbers does not match: {filePathes}")
        return

    frameNum = imgs[0].n_frames
    imgTensors = []
    try:
        readPos = 1
        for i in range(frameNum):
            frameTensors = [np.array(x.copy().resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BICUBIC).getdata(), dtype='uint8') for x in imgs]
            for frameTensor in frameTensors:
                if frameTensor.shape != (IMAGE_SIZE * IMAGE_SIZE, 3):
                    logger.error(f"@szh:The file size is not 512 * 512 * 3:  {filePathes}, {frameTensor.size()}")
                    break
            frameTensors = [np.reshape(x, (IMAGE_SIZE, IMAGE_SIZE, 3)) for x in frameTensors]
            if len(frameTensors) > 1:
                imgTensor = sum(frameTensors)
            else:
                imgTensor = frameTensors[0]

            imgTensors.append(transform(imgTensor))
            for x in imgs:
                x.seek(readPos)
            readPos += 1
    except EOFError:
        for x in imgs:
            x.close()

    imgTensors = torch.stack(imgTensors, 0)
    details = np.zeros((frameNum, len(INSPECT2ID)))
    with torch.no_grad():
        imgTensors = imgTensors.to(device)
        
        outsAll = models[0](imgTensors).cpu()
        outsOne = models[1](imgTensors).cpu()
        outsAll[:, 0] = outsOne.squeeze(1)
        
        outsAll = sigMoid(outsAll)
        for i, out in zip(range(frameNum), outsAll):
            for k in range(out.shape[0]):
                value = out[k]
                if value > THRESHOLDS[k]:
                    details[i][k] = i + 1
                else:
                    details[i][k] = -1 * (i + 1)

    results = pickLongMatch(details)
    resInfo = {}
    for i in range(len(IDS2INSPECT)):
        resInfo[IDS2INSPECT[i]] = (results[i][0], results[i][1])
    return resInfo, details

def extractAFAndSHGFile(dir):
    try:
        return [dir + "/" + x for x in os.listdir(dir) if ("AF_Color" in x or "SHG_Color" in x) and x.endswith(".tif")]
    except FileNotFoundError:
        logger.error(f"@ERROR:extractAFAndSHGFile: dir does not exists: {dir}")
        return None
    
def main(projPathes, outputFile):
    tifFilesPair = []
    for projPath in projPathes:
        for dataGroup in os.listdir(projPath):
            personPath = projPath + "/" + dataGroup + "/原始"
            if not os.path.isdir(personPath):
                logger.info(f"@szh: skip path: {personPath}")
                continue

            for personId in os.listdir(personPath):
                midPath = personPath + "/" + personId + "/三维快速扫描/"
                if not os.path.isdir(midPath):
                    logger.info(f"@szh: skip path: {midPath}")
                    continue

                for projName in os.listdir(midPath):
                    fullPath = midPath + "/" + projName
                    if not os.path.isdir(fullPath):
                        logger.info(f"@szh: skip path: {fullPath}")
                        continue

                    afshgFiles = extractAFAndSHGFile(fullPath)
                    if afshgFiles is None or len(afshgFiles) == 0:
                        logger.info(f"@szh: skip path: {fullPath}")
                        continue

                    tifFilesPair.append((fullPath, dataGroup, projName, afshgFiles))

    models = prepareModels()
    with open(outputFile, "a") as appendFile:
        for projPath, day, projName, tifFiles in tifFilesPair:
            resInfo, details = inference(models, tifFiles)
            logger.info(f"{day}-{projName}: {projPath}, resultInfo = {resInfo}")
            appendFile.write(f"{day}-{projName}: {projPath}, resultInfo = {resInfo}\n")
            appendFile.flush()


if __name__ == '__main__':
    try:
      opts, args = getopt.getopt(sys.argv[1:], "hi:o:", ["ipath=","ofile="])
    except getopt.GetoptError:
      print('inference.py -i <proj pathes> -o <outputfile>')
      sys.exit(2)

    projPathes = []
    outputfile = None
    for opt, arg in opts:
        if opt == '-h':
            print('inference.py -i <proj pathes> -o <outputfile>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            projPathes.append(arg)
        elif opt in ("-o", "--ofile"):
            outputfile = arg
    
    if len(projPathes) == 0:
        print('inference.py -i <proj pathes> -o <outputfile>')
        sys.exit(2)
    if outputfile is None:
        outputfile = "inference-result.txt"
    
    main(projPathes, outputfile)

    