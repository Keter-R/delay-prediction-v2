import os

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import StochasticWeightAveraging
from torch import Tensor, optim
import torchmetrics as tm
from torcheval.metrics.aggregation.auc import AUC
from pytorch_lightning import loggers as pl_loggers
from codes import loader


def Task(seed, config):
    pl.seed_everything(seed=seed, workers=True)
    data = loader.load_data(seed, config['data'],
                            using_temporal=config['using_temporal_graph'],
                            using_knn=config['using_knn_graph'])
    models = loader.load_model(config['models'], data)
    metrics = dict()
    curves = dict()
    for name, model in models.items():
        if name == 'gcn_temporal':
            task = torch_model_task(model, name, seed, data, config)
            curves[name] = task.validation_metrics
        elif name == 'gcn_knn':
            task = torch_model_task(model, name, seed, data, config)
            curves[name] = task.validation_metrics
        elif name == 'std_gcn_temporal':
            task = torch_model_task(model, name, seed, data, config)
            curves[name] = task.validation_metrics
        elif name == 'mlp':
            task = torch_model_task(model, name, seed, data, config)
            curves[name] = task.validation_metrics
    for name, curve in curves.items():
        metrics[name] = curve[-1]
    return metrics, curves


def torch_model_task(model, name, seed, data, config):
    model = model.to("cuda")
    loss_fn = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([data['np_ratio'] * config['loss_weight_alpha']]).to("cuda"))
    task = pl_Task(model=model, loss_fn=loss_fn, epoch=config['epoch'], lr=config['lr'],
                   lr_gamma=config['lr_gamma'], lr_step_size=config['lr_step_size'],
                   weight_decay=config['weight_decay'], data_len=data['data_length'],
                   data_feature=data['data_feature'])
    ckpt_callback = pl.callbacks.ModelCheckpoint(dirpath=f'./checkpoints/{name}',
                                                 filename='{epoch}-{val_loss:.2f}-{AUC:.2f}',
                                                 monitor='AUC', mode='max')
    TI = config["data"]["temporal_graph"]["time_interval"]
    SW = config["data"]["temporal_graph"]["self_weight"]
    LD = config["models"][name]["layer_dim"]
    version = None
    if name == "gcn_temporal":
        version = f"TI_{TI}_SW_{SW}_LD_{LD}"
    elif name == "mlp":
        version = f"LD_{LD}"
    trainer = pl.Trainer(accelerator="gpu", devices="1", max_epochs=task.epoch,
                         deterministic="warn", log_every_n_steps=1,
                         callbacks=[ckpt_callback],
                         logger=pl_loggers.TensorBoardLogger(f"./lightning_logs/{name}",
                                                             name=f'seed_{seed}',
                                                             version=version))
    trainer.fit(task, data['data_module'])
    trainer.validate(task, data['data_module'], ckpt_path="best")
    return task


class pl_Task(pl.LightningModule):
    def __init__(self, model, loss_fn, epoch, lr, lr_gamma, lr_step_size, weight_decay, data_len, data_feature):
        super(pl_Task, self).__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.epoch = epoch
        self.lr = lr
        self.lr_gamma = lr_gamma
        self.lr_step_size = lr_step_size
        self.weight_decay = weight_decay
        self.data_len = data_len
        self.data_feature = data_feature
        self.validation_metrics = []

    def loss(self, pred, target):
        if pred.device != 'cuda':
            pred = pred.to('cuda')
        if target.device != 'cuda':
            target = target.to('cuda')
        return self.loss_fn(pred, target)

    def forward(self, x, y):
        x = x.flatten()
        feat = x.reshape(self.data_len, self.data_feature)
        # temporal_adj = (x[self.data_len * self.data_feature:self.data_len * (self.data_feature + self.data_len)]
        #                 .reshape(self.data_len, self.data_len))
        # knn_adj = x[self.data_len * (self.data_feature + self.data_len):].reshape(self.data_len, self.data_len)
        pred = self.model(feat)
        pred = pred.flatten()
        y = y.flatten()
        n = y.shape[0] // 2
        # first half of y is target indexes, second half is target values
        indexes = y[:n].long()
        y = y[n:]
        pred = pred[indexes]
        return pred, y

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred, y = self(x, y)
        loss = self.loss(pred, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred, y = self(x, y)
        loss = self.loss(pred, y)
        metrics = calculate_metrics(pred, y)
        metrics['val_loss'] = loss.item()
        self.validation_metrics.append(metrics)
        for key, value in self.validation_metrics[-1].items():
            self.log(key, value)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        # lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.lr_gamma)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.lr_step_size, gamma=self.lr_gamma)
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}


def calculate_metrics(y_hat: Tensor, y: Tensor) -> dict:
    auc = tm.classification.BinaryAUROC().to("cuda")
    acc = tm.classification.BinaryAccuracy().to("cuda")
    recall = tm.classification.BinaryRecall().to("cuda")
    spec = tm.classification.BinarySpecificity().to("cuda")
    prec = tm.classification.BinaryPrecision().to("cuda")
    f1 = tm.classification.BinaryF1Score().to("cuda")
    auc = auc(y_hat, y).item()
    Accuracy = acc(y_hat, y).item()
    Sensitivity = recall(y_hat, y).item()
    Specificity = spec(y_hat, y).item()
    Precision = prec(y_hat, y).item()
    F1_score = f1(y_hat, y).item()
    GMean = (Sensitivity * Specificity) ** 0.5
    # if y_hat out of range [0, 1]
    # apply sigmoid function to y_hat
    if y_hat.max() > 1 or y_hat.min() < 0:
        y_hat = torch.sigmoid(y_hat)
    y_hat = (y_hat >= 0.5).int()
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in range(y_hat.shape[0]):
        if y[i] == 1 and y_hat[i] == 1:
            TP += 1
        elif y[i] == 0 and y_hat[i] == 0:
            TN += 1
        elif y[i] == 0 and y_hat[i] == 1:
            FP += 1
        elif y[i] == 1 and y_hat[i] == 0:
            FN += 1
    FPR = FP / (FP + TN)
    FNR = FN / (TP + FN)
    TPR = TP / (TP + FN)
    TNR = TN / (TN + FP)
    metrics = {"Accuracy": Accuracy, "AUC": auc, "Sensitivity": Sensitivity, "Specificity": Specificity,
               "F1_score": F1_score, "GMean": GMean, "Precision": Precision,
               "FPR": FPR, "FNR": FNR, "TPR": TPR, "TNR": TNR,
               "TP": TP, "TN": TN, "FP": FP, "FN": FN}
    print(metrics)
    return metrics
