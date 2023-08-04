import torch
import pytorch_lightning as pl
import torch.nn.functional as F
# import rnnModel_qkv as rnnModel
import rnnModel_dot as rnnModel
import dataProcess
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score
from pdb import set_trace as bp

class Classifier(pl.LightningModule):
    def __init__(self, args):
        super(Classifier, self).__init__()
        self.automatic_optimization = False
        self.lr = args['lr']
        self.args = args

        self.labelDs, labels = dataProcess.jsonToDataset(args['label_data'], return_label=True, params=args)
        self.n_classes = len(set(labels))
        self.trainDs = dataProcess.jsonToPairData(args['train_data'], params=args)
        self.testDs = dataProcess.jsonToDataset(args['test_data'], params=args)
        if len(args['val_data']) > 0:
            self.valDs = dataProcess.jsonToDataset(args['val_data'], params=args)
        else:
            valRatio = args['dev_ratio']
            trainRatio = 1 - valRatio
            self.trainDs, self.valDs = torch.utils.data.random_split(self.trainDs, [trainRatio, valRatio])
        
        self.model = rnnModel.Classifier(self.n_classes)
        self.save_hyperparameters()

    # Interpolate between examples in a batch
    def get_pairs(self, batch):

        # Method to convert y from scalar label to one-hot vector representation
        def get_one_hot(y):
            otgt = torch.zeros(y.size(0),self.n_classes).to(y.device) # Need to set device due to instantiation outside of init
            otgt.scatter_(1,y.unsqueeze(1),1)
            return otgt

        alphaLow = 0.2
        alphaHigh = 0.8
        lam = torch.distributions.Beta(torch.tensor([alphaLow]), torch.tensor([alphaHigh])).sample().to(batch[0].device)
        x, y = batch
        randIdx = torch.randperm(x.shape[0])
        x = lam*x + (1-lam)*x[randIdx]
        y = lam*get_one_hot(y) + (1-lam)*get_one_hot(y[randIdx])

        return x, y

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        opt.zero_grad()
        x, y = self.get_pairs(batch)
        x, y = batch

        y_pred = self.model(x)
        loss = F.cross_entropy(y_pred, y)

        self.manual_backward(loss)
        opt.step()

        self.log('train_loss', loss)
        return loss
    
    def training_epoch_end(self, outputs):
        sch = self.lr_schedulers()

        # If the selected scheduler is a ReduceLROnPlateau scheduler.
        if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
            sch.step(self.trainer.callback_metrics["val_loss"])

    def validation_step(self, batch, batch_idx):
        x, y, mask = batch
        y_pred = self.model(x)
        loss = F.cross_entropy(y_pred, y)
        
        self.log('val_loss', loss)
        self.log('val_accuracy', accuracy_score(y.cpu().data, torch.argmax(y_pred, dim=1).cpu()), prog_bar=True, on_epoch=True)
        self.log('val_f1', f1_score(y.cpu().data, torch.argmax(y_pred, dim=1).cpu(), average='macro'), prog_bar=True, on_epoch=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.model(x)
        loss = F.cross_entropy(y_pred, y)

        self.log('test_loss', loss)
        self.log('test_accuracy', accuracy_score(y.cpu().data, torch.argmax(y_pred, dim=1).cpu()), prog_bar=True, on_epoch=True)
        self.log('test_f1', f1_score(y.cpu().data, torch.argmax(y_pred, dim=1).cpu(), average='macro'), prog_bar=True, on_epoch=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    # Only used for inference
    def forward(self, x):
        return self.model(x)

    def train_dataloader(self):
        trainFull = torch.utils.data.ConcatDataset([self.trainDs, self.labelDs])
        return DataLoader(trainFull, batch_size=32, shuffle=True, num_workers=8)
    
    def val_dataloader(self):
        return DataLoader(self.valDs, batch_size=32, num_workers=8)
    
    def test_dataloader(self):
        return DataLoader(self.testDs, batch_size=32, num_workers=8)