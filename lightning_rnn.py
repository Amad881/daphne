import torch
from torch import nn
import pytorch_lightning as pl
import torch.nn.functional as F
import rnnModel_dot as rnnModel
import dataProcess
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score
from transformers import AutoTokenizer
import faiss
from pdb import set_trace as bp
import numpy as np

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """

    def __init__(self, margin_pos, margin_neg):
        super(ContrastiveLoss, self).__init__()
        self.margin_pos = margin_pos
        self.margin_neg = margin_neg
        self.eps = 1e-9

    def forward(self, output1, output2, target, size_average=True):
        distances = torch.norm(F.normalize(output1,dim=1)-F.normalize(output2,dim=1), dim=1)
        losses = 1.0 * (target.float() * torch.pow(torch.clamp(distances - self.margin_pos, min=0.0), 2) +
                                  (1 + -1 * target).float() * torch.pow(torch.clamp(self.margin_neg - distances, min=0.0), 2))
        return losses.mean() if size_average else losses.sum()

class Classifier(pl.LightningModule):
    def __init__(self, args):
        super(Classifier, self).__init__()
        self.automatic_optimization = False
        self.args = args
        self.metricsDict = dict()
        self.knn = args['knn']
        self.pairwise = args['pairwise']
        if self.args['cartography']:
            self.metricsDict['cartography'] = dict()

        mPos = 0.8
        mNeg = 1.2
        self.Lcon = ContrastiveLoss(mPos, mNeg)
        self.Lmix = nn.KLDivLoss(reduction='batchmean')

        self.dsTokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        self.labelDs, self.labels = dataProcess.jsonToDataset(args['label_data'], return_label=True, params=args, dataType='label')
        labelsList = []
        for label in self.labels:
            labelsList.extend(label)
            self.labels = set(labelsList)
        self.n_classes = len(self.labels)
        self.trainDs = dataProcess.jsonToDataset(args['train_data'], params=args)
        self.testDs = dataProcess.jsonToDataset(args['test_data'], params=args)
        if len(args['val_data']) > 0:
            self.valDs = dataProcess.jsonToDataset(args['val_data'], params=args)
        else:
            valRatio = args['dev_ratio']
            trainRatio = 1 - valRatio
            self.trainDs, self.valDs = torch.utils.data.random_split(self.trainDs, [trainRatio, valRatio], generator=torch.Generator().manual_seed(args['train_seed']))

        if args['eval']:
            self.paraDs = None
        elif args['para_task'] != '' and args['para_task'] != 'None':
            self.paraDs = dataProcess.jsonToDataset(args['para_data'], params=args, dataType='para')

        if args['pairwise']:
            datasets = [self.trainDs, self.labelDs]
            if self.args['para_task'] == 'train':
                datasets.append(self.paraDs)
            self.pairDs = dataProcess.combineDatasets(datasets, self.n_classes)


        self.model = rnnModel.Classifier(self.n_classes, dropout=args['dropout'])
        if not args['eval']:
            self.save_hyperparameters()

    def getLossPerExample(self, batch, batch_idx):
        x, y, dataIds = batch
        y_pred = self.model(x)
        loss = F.cross_entropy(y_pred, y, reduction='none')
        for idx, dataId in enumerate(dataIds):
            if dataId not in self.metricsDict['cartography']:
                self.metricsDict['cartography'][dataId] = []
            self.metricsDict['cartography'][dataId].append(loss[idx].item())

    def training_step(self, batch, batch_idx):

        if self.args['cartography']:
            self.getLossPerExample(batch, batch_idx)

        opt = self.optimizers()
        opt.zero_grad()

        if self.args['pairwise']:
            x1, x2, y1, y2, ids1, ids2, betas = batch
            # Convert targets to one-hot

            # yPair = (y1 == y2).float()
            # y1 = F.one_hot(y1, self.n_classes)
            # y2 = F.one_hot(y2, self.n_classes)

            # Convert multilabel targets to one-hot
            y1_oh = torch.zeros(len(y1), self.n_classes)
            y2_oh = torch.zeros(len(y2), self.n_classes)
            for idx, label in enumerate(y1):
                for l in label:
                    y1_oh[idx][l] = 1
            for idx, label in enumerate(y2):
                for l in label:
                    y2_oh[idx][l] = 1

            # normalize one-hot vectors
            y1_oh = F.normalize(y1_oh, dim=1).to(self.device)
            y2_oh = F.normalize(y2_oh, dim=1).to(self.device)
            # yPair is the cosine similarities between each pair of examples
            yPair = torch.cosine_similarity(y1_oh, y2_oh, dim=1)
            

            # Encode examples
            x1 = self.model.encode(x1)
            x2 = self.model.encode(x2)

            # Sample lambda from beta distribution
            lam = torch.distributions.beta.Beta(betas, betas).sample()
            lam = lam.unsqueeze(1).float()
            # Interpolate targets and examples using lambda
            y = lam * y1_oh + (1 - lam) * y2_oh
            x = lam * x1 + (1 - lam) * x2

            # Classify mixed example
            _, y_pred = self.model.classify(x)

            # Compute KL divergence loss between mixed example and targets
            Lmix = self.Lmix(F.log_softmax(y_pred, dim=1), y)
            # Compute contrastive loss between encoded examples
            Lcon = self.Lcon(x1, x2, yPair)

            beta = self.args['beta']
            loss = beta * -Lcon + (1 - beta) * Lmix

        else: 
            x, y, ids = batch
            y_pred = self.model(x)
            loss = F.cross_entropy(y_pred, y)

        self.manual_backward(loss)
        opt.step()

        self.log('train_loss', loss)
        return loss
    
    def on_train_epoch_start(self):
        if self.args['pairwise']:
            # self.pairDs.perEpochSample(model=self.model, tokenizer=self.dsTokenizer)
            # self.model.train()
            self.pairDs.perEpochSample()

    def training_epoch_end(self, outputs):
        sch = self.lr_schedulers()

        # If the selected scheduler is a ReduceLROnPlateau scheduler.
        if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
            sch.step(self.trainer.callback_metrics["val_f1"])

    def validation_step(self, batch, batch_idx):
        x, y, dataIds = batch
        y_pred = self.model(x)
        # loss = F.cross_entropy(y_pred, y)
        
        # self.log('val_loss', loss)
        # self.log('val_accuracy', accuracy_score(y.cpu().data, torch.argmax(y_pred, dim=1).cpu()), prog_bar=True, on_epoch=True)
        # self.log('val_f1', f1_score(y.cpu().data, torch.argmax(y_pred, dim=1).cpu(), average='macro'), prog_bar=True, on_epoch=True)
        
        threshold = 1/self.n_classes
        y_pred = torch.where(y_pred > threshold, 1, 0)

        # Average example accuracy
        exampleAccuracy = 0
        exampleF1 = 0
        for idx in range(len(y)):
            exampleAccuracy += accuracy_score(y[idx].cpu().data, y_pred[idx].cpu())
            exampleF1 += f1_score(y[idx].cpu().data, y_pred[idx].cpu(), average='macro')
        exampleAccuracy /= len(y)
        exampleF1 /= len(y)

        self.log('val_accuracy', accuracy_score(y.cpu().data, y_pred.cpu()), prog_bar=True, on_epoch=True)
        self.log('val_f1', f1_score(y.cpu().data, y_pred.cpu(), average='macro'), prog_bar=True, on_epoch=True)

        self.log('val_example_accuracy', exampleAccuracy, prog_bar=True, on_epoch=True)
        self.log('val_example_f1', exampleF1, prog_bar=True, on_epoch=True)

        # self.log('val_accuracy', accuracy_score(y.cpu, torch.argmax(y_pred, dim=1).cpu()), prog_bar=True, on_epoch=True)
        # self.log('val_f1', f1_score(y, torch.argmax(y_pred, dim=1).cpu(), average='macro'), prog_bar=True, on_epoch=True)
        return 0

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.args['lr'])
        patience = int(self.args['patience']/2)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=patience, verbose=True)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    # Only used for inference
    def forward(self, x):
        return self.model(x)

    def on_test_start(self, multi=True):
        self.metricsDict = {}
        if self.args['knn']:
            print("Building KNN index...")
            exData = torch.utils.data.ConcatDataset([self.trainDs, self.labelDs])
            exLoader = DataLoader(exData, batch_size=self.args['batch_size'], num_workers=self.args['num_workers'], collate_fn=self.collator)

            key_rep = []
            for batch in exLoader:
                x = batch[0]
                y = batch[1]
                dataIds = batch[2]
                x = x.to(self.device)
                key_rep.append(self.model.encode(x))
            keys = F.normalize(torch.cat(key_rep, dim=0), dim=1)
            

            label_map = [[] for i in range(self.n_classes)]
            try:
                for idx, val in enumerate(exData):
                    y, x = val[0]
                    labelId = val[1]
                    label_map[y].append(idx)
            except:
                print("Error: Label map not found")

            self.class_keys = []
            for lst in label_map:
                self.class_keys.append(keys[lst])

    def knn_predict(self, queries):
        queries = F.normalize(queries, dim=1).cpu().numpy()
        scores = []
        for keys in self.class_keys:
            scores.append(self.use_faiss(queries,keys))
        return torch.cat(scores, dim=1).to(self.device)

    # For KNN
    def use_faiss(self, queries, keys):
        index = faiss.IndexFlatL2(keys.shape[1])
        index.add(keys.detach().cpu().numpy())
        k = keys.shape[0]
        D, I = index.search(queries, k)
        return 1./ (torch.sqrt(torch.from_numpy(D[:,0]))+1e-6).unsqueeze(1)

    def test_step(self, batch, batch_idx):
        x, y, dataIds = batch
        y_pred = self.model(x)
        
        if self.knn:
            sm_wt = 0.9
            queries = self.model.encode(x)
            pred_nn = self.knn_predict(queries)
            y_pred = sm_wt*pred_sm + (1 - sm_wt)*pred_nn
        else:
            y_pred = pred_sm

        threshold = 1/self.n_classes
        y_pred = torch.where(y_pred > threshold, 1, 0)

        # Average example accuracy
        exampleAccuracy = 0
        exampleF1 = 0
        for idx in range(len(y)):
            exampleAccuracy += accuracy_score(y[idx].cpu().data, y_pred[idx].cpu())
            exampleF1 += f1_score(y[idx].cpu().data, y_pred[idx].cpu(), average='macro')
        exampleAccuracy /= len(y)
        exampleF1 /= len(y)

        if 'test_accuracy' not in metricsDict:
            metricsDict['test_accuracy'] = []
        metricsDict['test_accuracy'].append(accuracy_score(y.cpu().data, torch.argmax(y_pred, dim=1).cpu()))
        if 'test_f1' not in metricsDict:
            metricsDict['test_f1'] = []
        metricsDict['test_f1'].append(f1_score(y.cpu().data, torch.argmax(y_pred, dim=1).cpu(), average='macro'))

        self.log('test_accuracy', accuracy_score(y.cpu().data, torch.argmax(y_pred, dim=1).cpu()), prog_bar=True, on_epoch=True)
        self.log('test_f1', f1_score(y.cpu().data, torch.argmax(y_pred, dim=1).cpu(), average='macro'), prog_bar=True, on_epoch=True)

        self.log('test_example_accuracy', exampleAccuracy, prog_bar=True, on_epoch=True)
        self.log('test_example_f1', exampleF1, prog_bar=True, on_epoch=True)

        return 0

    def test_step_single(self, batch, batch_idx):
        x, y, dataIds = batch
        pred_sm = self.model(x)

        if self.knn:
            sm_wt = 0.9
            queries = self.model.encode(x)
            pred_nn = self.knn_predict(queries)
            y_pred = sm_wt*pred_sm + (1 - sm_wt)*pred_nn
        else:
            y_pred = pred_sm

        loss = F.cross_entropy(y_pred, y)

        metricsDict = self.metricsDict
        if 'test_loss' not in metricsDict:
            metricsDict['test_loss'] = []
        metricsDict['test_loss'].append(loss.item())
        if 'test_accuracy' not in metricsDict:
            metricsDict['test_accuracy'] = []
        metricsDict['test_accuracy'].append(accuracy_score(y.cpu().data, torch.argmax(y_pred, dim=1).cpu()))
        if 'test_f1' not in metricsDict:
            metricsDict['test_f1'] = []
        metricsDict['test_f1'].append(f1_score(y.cpu().data, torch.argmax(y_pred, dim=1).cpu(), average='macro'))

        self.log('test_loss', loss)
        self.log('test_accuracy', accuracy_score(y.cpu().data, torch.argmax(y_pred, dim=1).cpu()), prog_bar=True, on_epoch=True)
        self.log('test_f1', f1_score(y.cpu().data, torch.argmax(y_pred, dim=1).cpu(), average='macro'), prog_bar=True, on_epoch=True)
        return loss
    
    def on_test_end(self):
        metricsDict = self.metricsDict
        for key, value in metricsDict.items():
            metricsDict[key] = np.array(value).mean()

    def train_dataloader(self):
        if self.args['pairwise']:
            return DataLoader(self.pairDs, batch_size=self.args['batch_size'], shuffle=True, num_workers=self.args['num_workers'], collate_fn=self.pairwiseCollator)
        else:
            trainFull = torch.utils.data.ConcatDataset([self.trainDs, self.labelDs])
            if self.args['para_task'] == 'train':
                trainFull = torch.utils.data.ConcatDataset([trainFull, self.paraDs])
            return DataLoader(trainFull, batch_size=self.args['batch_size'], shuffle=True, num_workers=self.args['num_workers'], collate_fn=self.collator)
    
    def val_dataloader(self):
        return DataLoader(self.valDs, batch_size=self.args['batch_size'], num_workers=self.args['num_workers'], collate_fn=self.collator)
    
    def test_dataloader(self):
        return DataLoader(self.testDs, batch_size=self.args['batch_size'], num_workers=self.args['num_workers'], collate_fn=self.collator)

    def collator(self, batch):
        dataIds = [x[1] for x in batch]
        batch = [x[0] for x in batch]

        texts = [x[1] for x in batch]
        # Convert multilabel to binary indicator matrix
        labelsArr = []
        for x in batch:
            labelsRow = [0]*self.n_classes
            for label in x[0]:
                labelsRow[label] = 1
            labelsArr.append(labelsRow)
        
        # labels = torch.tensor([x[0] for x in batch])
        labels = torch.tensor(labelsArr)
        # labels = [x[0] for x in batch]
        encoded = self.dsTokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        return encoded, labels, dataIds
    
    def pairwiseCollator(self, batch):
        pair1 = [x[0] for x in batch]
        pair2 = [x[1] for x in batch]
        betas = torch.tensor([x[2] for x in batch])

        texts1 = [x['text'] for x in pair1]
        encoded1 = self.dsTokenizer(texts1, return_tensors="pt", padding=True, truncation=True)
        texts2 = [x['text'] for x in pair2]
        encoded2 = self.dsTokenizer(texts2, return_tensors="pt", padding=True, truncation=True)

        # labels1 = torch.tensor([x['label'] for x in pair1])
        # labels2 = torch.tensor([x['label'] for x in pair2])
        labels1 = [x['label'] for x in pair1]
        labels2 = [x['label'] for x in pair2]

        exampleIds1 = [x['exampleId'] for x in pair1]
        exampleIds2 = [x['exampleId'] for x in pair2]
        return encoded1, encoded2, labels1, labels2, exampleIds1, exampleIds2, betas

    def infer(self, utterance, return_labelNum=False, multilabel=True):
        # Create dictionary relating label values to label text
        try:
            self.label_dict
        except:
            label_dict = dict()
            for example in self.labelDs:
                labelStr = example[0][1]
                if multilabel:
                    labelId = example[0][0][0]
                else:
                    labelId = example[0][0]
                label_dict[labelId] = labelStr
            self.label_dict = label_dict

        x = self.dsTokenizer(utterance, return_tensors="pt", padding=True, truncation=True)
        pred_sm = self.model(x)

        if self.knn:
            sm_wt = 0.9
            queries = self.model.encode(x)
            pred_nn = self.knn_predict(queries.detach())
            y_pred = sm_wt*pred_sm + (1 - sm_wt)*pred_nn
        else:
            y_pred = pred_sm

        if multilabel:
            threshold = 1/self.n_classes
            # Return indices of labels with probability greater than threshold
            bestPred = torch.where(y_pred > threshold)[1].detach().cpu().numpy()
            output = []
            for pred in bestPred:
                output.append(self.label_dict[pred])
            if return_labelNum:
                output = (output, bestPred, y_pred)
        else:
            bestPred = torch.argmax(y_pred, dim=1).detach().cpu().numpy()[0]
            output = self.label_dict[bestPred]
            y_pred = y_pred.detach().cpu().numpy()[0]
            if y_pred[bestPred] < 4.0:
                output = "Sorry I don't understand, can you please repeat your statement in some other way?"
            if return_labelNum:
                output = (output, bestPred, y_pred)
        return output

    def batch_infer(self, utterances, return_labelNum=True):
        # Create dictionary relating label values to label text
        try:
            self.label_dict
        except:
            label_dict = dict()
            for example in self.labelDs:
                labelStr = example[0][1]
                labelId = example[0][0]
                label_dict[labelId] = labelStr
            self.label_dict = label_dict

        x = self.dsTokenizer(utterances, return_tensors="pt", padding=True, truncation=True)
        pred_sm = self.model(x)

        if self.knn:
            sm_wt = 0.9
            queries = self.model.encode(x)
            pred_nn = self.knn_predict(queries.detach())
            y_pred = sm_wt*pred_sm + (1 - sm_wt)*pred_nn
        else:
            y_pred = pred_sm

        bestPred = torch.argmax(y_pred, dim=1).detach().cpu().numpy()

        output = [self.label_dict[x] for x in bestPred]

        y_pred = y_pred.detach().cpu().numpy()
        for i in range(len(output)):
            if y_pred[i][bestPred[i]] < 4.0:
                output[i] = "Sorry I don't understand, can you please repeat your statement in some other way?"

        # Normalize logits and present as probabilities
        if return_labelNum:
            output = (output, bestPred, y_pred)
        
        return output