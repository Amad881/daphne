import argparse
import pandas as pd
import json
import re
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from pdb import set_trace as bp

def getParameters():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str, default='',
                        help='location of the training data, should be a json file from vp-tokenizer')
    parser.add_argument('--test_data', type=str, default='n',
                        help='location of the test data, should be a json file from vp-tokenizer')
    parser.add_argument('--val_data', type=str, default='',
                        help='location of the validation data, should be a json file from vp-tokenizer')
    parser.add_argument('--label_data', type=str, default='',
                        help='location of the label map (int -> sentence) in json format')
    parser.add_argument('--dev_ratio', type=float, default=0.1,
                        help='fraction of train-dev data to use for dev')
    parser.add_argument('--lr', type=float, default=.0001,
                        help='initial learning rate')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--knn', action='store_true',
                        help='use KNN in test time')
    parser.add_argument('--max_epochs', type=int, default=40,
                        help='upper epoch limit')
    parser.add_argument('--train_seed', type=int, default=42,
                        help='random seed for training')
    parser.add_argument('--data_seed', type=int, default=42,
                    help='random seed for splitting data into train and dev')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size')
    parser.add_argument('--logger_name', type=str, default='test_pairwise',
                        help='name for wandB logger')
    parser.add_argument('--notes', type=str, default='',
                        help='notes for wandB logger')
    parser.add_argument('--gpu_num', type=int, default=1,
                        help='gpu number to use')
    parser.add_argument('--eval', action='store_true',
                        help='evaluate the model')
    parser.add_argument('--model_path', type=str, default='',
                        help='path to model to evaluate')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='number of workers for data loader')
    parser.add_argument('--patience', type=int, default=5,
                        help='patience for early stopping and learning rate decay if used')
    parser.add_argument('--para_data', type=str, default='',
                        help='location of the paraphrase data, should be a json file from vp-tokenizer')
    parser.add_argument('--para_task', type=str, default='train',
                        help='what to do with the paraphrase data: train, epoch_sample')
    parser.add_argument('--pairwise', action='store_true',
                        help='whether to use pairwise loss')
    parser.add_argument('--cartography', type=bool, default=False,
                        help='whether to use cartography loss')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='beta for pairwise loss')
    parser.add_argument('--infer', action='store_true',
                        help='whether to use inference mode')
    parser.add_argument('--infer_utterance', type=str, default='',
                        help='utterance to infer')
    parser.add_argument('--multi_label', action='store_true',
                        help='whether to use multilabel loss')
    parser.set_defaults(knn=False)
    parser.set_defaults(eval=False)
    parser.set_defaults(pairwise=False)
    args = parser.parse_args()
    return vars(args)

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    """
    string = re.sub("[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub("\'s", " \'s", string)
    string = re.sub("\'m", " \'m", string)
    string = re.sub("\'ve", " \'ve", string)
    string = re.sub("n\'t", " n\'t", string)
    string = re.sub("\'re", " \'re", string)
    string = re.sub("\'d", " \'d", string)
    string = re.sub("\'ll", " \'ll", string)
    string = re.sub(",", " , ", string)
    string = re.sub("!", " ! ", string)
    string = re.sub("\(", " ( ", string)
    string = re.sub("\)", " ) ", string)
    string = re.sub("\?", " ? ", string)
    string = re.sub("\s{2,}", " ", string)
    return string.strip().lower().split(" ")

def paraDfPklToJsonl(inFile, outFile):
    df = pd.read_pickle(inFile)
    dictList = []
    for idx, row in df.iterrows():
        paras = row['genSamples'][-1]
        for para in paras:
            dictList.append({'label': row['label'], 'text': clean_str(para)})
    with open(outFile, 'w') as fout:
        for example in dictList:
            fout.write(json.dumps(example) + "\n")

if __name__ == "__main__":
    val = pkl.load(open('vp_cartoOrig_March27_1111_metricsDict.pkl', 'rb'))
    val = val['cartography']
    origVals = []
    paraVals = []
    maxLen = 60
    for example, lossVals in val.items():
        lossVal = lossVals[:maxLen]
        if 'original' in example:
            origVals.append(np.array(lossVal))
        else:
            paraVals.append(np.array(lossVal))
    
    origVals = np.array(origVals)
    paraVals = np.array(paraVals)

    print("Overall std and mean of original: ", np.std(origVals), "__", np.mean(origVals))
    print("Overall std and mean of paraphrase: ", np.std(paraVals), "__", np.mean(paraVals))
    # Get values per 20 epochs
    quartile = int(maxLen / 4)
    for i in range(0, maxLen, quartile):
        print("Epoch ", i)
        print("Original: ", np.std(origVals[:, i]), "__", np.mean(origVals[:, i]))
        print("Paraphrase: ", np.std(paraVals[:, i]), "__", np.mean(paraVals[:, i]))

    # origVals = origVals[:300]
    # paraVals = paraVals[:300]
    origMeans = np.mean(origVals, axis=1)
    origStd = np.std(origVals, axis=1)
    plt.scatter(origStd, origMeans, label='original')
    paraMeans = np.mean(paraVals, axis=1)
    paraStd = np.std(paraVals, axis=1)
    plt.scatter(paraStd, paraMeans, label='paraphrase')
    plt.legend()
    # plt to image
    # plt.savefig('vp_cartoNLI_March27_1111_metricsDict.png')
    bp()