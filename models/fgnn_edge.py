import pytorch_lightning as pl
import torch
from models.base_model import GNN_Abstract_Base_Class

class EdgeClassifLoss(torch.nn.Module):
    def __init__(self, normalize=torch.nn.Sigmoid(), loss=torch.nn.BCELoss(reduction='mean')):
        super(EdgeClassifLoss, self).__init__()
        if isinstance(loss, torch.nn.BCELoss):
            self.loss = lambda preds,target: loss(preds, target.to(torch.float))
        else:
            self.loss = loss
        self.normalize = normalize

    def forward(self, raw_scores, target):
        """
        outputs is the output of siamese network (bs,n_vertices,n_vertices)
        """
        preds = self.normalize(raw_scores)
        loss = self.loss(preds,target)
        return torch.mean(loss)

class FGNN_Edge(GNN_Abstract_Base_Class):
    
    def __init__(self,model, optim_args, **kwargs):
        super().__init__(model, optim_args, **kwargs)
        self.loss = EdgeClassifLoss()
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        g, target = batch
        x = self(g)
        raw_scores = x.squeeze(-1)
        loss_value = self.loss(raw_scores, target)
        self.log('train_loss', loss_value, sync_dist=self.sync_dist)
        self.log_metric('train', data=g, raw_scores=raw_scores, target=target)
        return loss_value
    
    def validation_step(self, batch, batch_idx):
        g, target = batch
        x = self(g)
        raw_scores = x.squeeze(-1)
        loss_value = self.loss(raw_scores, target)
        self.log('val_loss', loss_value, sync_dist=self.sync_dist)
        self.log_metric('val', data=g, raw_scores=raw_scores, target=target)
        return loss_value
    
    def test_step(self, batch, batch_idx, dataloader_idx=None):
        g, target = batch
        x = self(g)
        raw_scores = x.squeeze(-1)
        loss_value = self.loss(raw_scores, target)
        self.log('test_loss', loss_value, sync_dist=self.sync_dist)
        self.log_metric('test', data=g, raw_scores=raw_scores, target=target)
        return loss_value