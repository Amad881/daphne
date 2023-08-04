import torch
import numpy as np
from pdb import set_trace as bp
from transformers import AutoTokenizer, AutoModel
import json
from torch.utils.data import Dataset
import pickle
import os
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class bertTokenizedData(Dataset):
    def __init__(self, data, dataType='original', params=None):
        self.params = params
        self.data = data
        self.dataType = dataType

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        exampleId = self.dataType + "_" + str(idx) 
        return self.data[idx], exampleId

class pairData(Dataset):
    def __init__(self, data, nClasses):
        self.data = data
        self.nClasses = nClasses
        self.labelFrqs, self.exampleIdxByLabel = self.getFreqsAndUttByLabel()
        self.getQuintAndBeta()
        self.posPairs, self.negPairs = self.getPairs()
        self.perEpochSample()

    def getFreqsAndUttByLabel(self):
        freqs = [0] * self.nClasses
        exampleIdxByLabel = [[] for i in range(self.nClasses)]
        for idx, row in enumerate(self.data):
            for label in row['label']:
                freqs[label] += 1
                exampleIdxByLabel[label].append(idx)
        freqsArr = np.array(freqs)
        return freqsArr, exampleIdxByLabel

    def getQuintAndBeta(self):
        # Assign each label to a quintile based on frequency
        quintiles = np.array_split(np.argsort(self.labelFrqs), 5)
        quintileByLabel = np.zeros(self.nClasses)
        for quintileIdx, quintile in enumerate(quintiles):
            for label in quintile:
                quintileByLabel[label] = quintileIdx

        self.quintileByLabel = quintileByLabel
        self.betaByQuintile = {0: 20, 1: 5, 2: 1, 3: 0.5, 4: 0.05}
        self.dataQuints = np.array([self.quintileByLabel[row['label']] for row in self.data])
        self.dataBetas = np.array([self.betaByQuintile[self.quintileByLabel[random.choice(row['label'])]] for row in self.data])



    def getPairs(self, numPos=50000, numNeg=100000):
        seenPairs = set()
        posPairs = [[] for i in range(self.nClasses)]
        negPairs = [[] for i in range(self.nClasses)]
        for dataIdx, row in enumerate(self.data):
            labels = row['label']
            for label in labels:
                for exampleIdx in self.exampleIdxByLabel[label]:
                    if exampleIdx != dataIdx:
                        if str((dataIdx, exampleIdx)) not in seenPairs:
                            posPairs[label].append((dataIdx, exampleIdx))
                            seenPairs.add(str((dataIdx, exampleIdx)))
            for otherLabel in range(self.nClasses):
                if otherLabel not in labels:
                    for exampleIdx in self.exampleIdxByLabel[otherLabel]:
                        if str((dataIdx, exampleIdx)) not in seenPairs:
                            negPairs[label].append((dataIdx, exampleIdx))
                            seenPairs.add(str((dataIdx, exampleIdx)))

        def samplePairs(allPairs, numSamples, nClasses):
            print('Sampling pairs')
            pairNumPerLabel = numSamples // nClasses
            classIdxs = np.arange(nClasses)
            random.shuffle(classIdxs)
            outPairs = []
            firstLoop = True
            growth = True
            oldLen = 0
            while len(outPairs) < numSamples and growth:
                for label in classIdxs:
                    if len(allPairs[label]) > 0:
                        if firstLoop:
                            random.shuffle(allPairs[label])
                            for i in range(pairNumPerLabel):
                                if len(allPairs[label]) > 0:
                                    outPairs.append(allPairs[label].pop())
                        else:
                            outPairs.append(allPairs[label].pop())
                firstLoop = False
                random.shuffle(classIdxs)
                if len(outPairs) == oldLen:
                    growth = False
                else:
                    oldLen = len(outPairs)
            print('Done sampling pairs')
            return outPairs

        outPosPairs = samplePairs(posPairs, numPos, self.nClasses)
        outNegPairs = samplePairs(negPairs, numNeg, self.nClasses)

        random.shuffle(outPosPairs)
        random.shuffle(outNegPairs)
        return outPosPairs, outNegPairs     
    
    def perEpochSample(self, model=None, posSize=25000, negSize=50000, tokenizer=None, batchSize=64):
        if model == None:
            # Randomly sample pos and neg pairs
            random.shuffle(self.posPairs)
            random.shuffle(self.negPairs)
            outData = self.posPairs[:posSize] + self.negPairs[:negSize]
        else:
            model.eval()
            # Sort pos pairs by similarity
            posPairScores = []

            # Batch over pairs and get scores
            for i in range(0, len(self.posPairs), batchSize):
                text1 = [self.data[pair[0]]['text'] for pair in self.posPairs[i:i+batchSize]]
                text2 = [self.data[pair[1]]['text'] for pair in self.posPairs[i:i+batchSize]]
                with torch.no_grad():
                    x1 = tokenizer(text1, return_tensors="pt", padding=True, truncation=True).to(device)
                    x2 = tokenizer(text2, return_tensors="pt", padding=True, truncation=True).to(device)
                    # Get L2 norm of difference between embeddings
                    scores = torch.linalg.norm(model.encode(x1) - model.encode(x2), dim=1)
                posPairScores.extend(scores.tolist())

            posPairScores = np.array(posPairScores)
            posPairIdxs = np.argsort(-posPairScores)
            posPairIdxs = posPairIdxs[::-1]
            posPairIdxs = posPairIdxs[:posSize]
            posPairs = [self.posPairs[idx] for idx in posPairIdxs]
            # Sort neg pairs by similarity
            negPairScores = []

            # Batch over pairs and get scores
            for i in range(0, len(self.negPairs), batchSize):
                text1 = [self.data[pair[0]]['text'] for pair in self.negPairs[i:i+batchSize]]
                text2 = [self.data[pair[1]]['text'] for pair in self.negPairs[i:i+batchSize]]
                with torch.no_grad():
                    x1 = tokenizer(text1, return_tensors="pt", padding=True, truncation=True).to(device)
                    x2 = tokenizer(text2, return_tensors="pt", padding=True, truncation=True).to(device)
                    scores = torch.linalg.norm(model.encode(x1) - model.encode(x2), dim=1)
                negPairScores.extend(scores.tolist())

            negPairScores = np.array(negPairScores)
            negPairIdxs = np.argsort(negPairScores)
            negPairIdxs = negPairIdxs[:negSize]
            negPairs = [self.negPairs[idx] for idx in negPairIdxs]
            outData = posPairs + negPairs
        self.pairedExamples = outData

    def __len__(self):
        return len(self.pairedExamples)

    def __getitem__(self, idx):
        pair = self.pairedExamples[idx]
        data0 = self.data[pair[0]]
        data1 = self.data[pair[1]]
        
        # Get beta for each pair using the max of the two betas
        beta = max(self.dataBetas[pair[0]], self.dataBetas[pair[1]])
        return data0, data1, beta

def jsonFileToArr(file):
    data = []
    with open(file, 'r') as fin:
        for line in fin.readlines():
            lineDict = json.loads(line)
            dataRow = [lineDict['label'], " ".join(lineDict['text'])]
            data.append(dataRow)
    return data

def jsonToDataset(jsonFile, return_label=False, params=None, fresh=True, dataType='original'):
    data = jsonFileToArr(jsonFile)
    ds = bertTokenizedData(data, dataType=dataType, params=params)
    if return_label:
        return ds, [row[0] for row in data]
    else:
        return ds

def combineDatasets(datasetList, nClasses):
    allData = []
    allDataTypes = []
    allExampleIds = []
    for ds in datasetList:
        for idx, dataPoint in enumerate(ds.data):
            exampleId = ds.dataType + "_" + str(idx)
            dataPointDict = {'label': dataPoint[0], 'text': dataPoint[1], 'exampleId': exampleId}
            allData.append(dataPointDict)
    outDs = pairData(allData, nClasses)
    return outDs