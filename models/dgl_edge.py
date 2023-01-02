import torch
from models.base_model import GNN_Abstract_Base_Class

class DGLEdgeLoss(torch.nn.Module):
    def __init__(self, normalize=torch.nn.Sigmoid(), loss=torch.nn.CrossEntropyLoss(reduction='mean')):
        super(DGLEdgeLoss, self).__init__()
        self.loss = loss
        self.normalize = normalize

    def forward(self, raw_scores, target):
        target = target.edata['solution'].squeeze(-1) #Because features are 2D column vectors
        preds = self.normalize(raw_scores)
        loss = self.loss(preds,target)
        return torch.mean(loss)

class DGL_Edge(GNN_Abstract_Base_Class):
    """
    Base class for any GNN using DGL with edge embeddings.
    """
    
    def __init__(self,model, optim_args, **kwargs):
        super().__init__(model, optim_args, **kwargs)
        self.loss = DGLEdgeLoss()
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        g, target = batch
        raw_scores = self(g)
        loss_value = self.loss(raw_scores, target)
        self.log('train_loss', loss_value, sync_dist=self.sync_dist)
        self.log_metric('train', data=g, raw_scores=raw_scores, target=target)
        return loss_value
    
    def validation_step(self, batch, batch_idx):
        g, target = batch
        raw_scores = self(g)
        loss_value = self.loss(raw_scores, target)
        self.log('val_loss', loss_value, sync_dist=self.sync_dist)
        self.log_metric('val', data=g, raw_scores=raw_scores, target=target)
        return loss_value
    
    def test_step(self, batch, batch_idx, dataloader_idx=None):
        g, target = batch
        raw_scores = self(g)
        loss_value = self.loss(raw_scores, target)
        self.log('test_loss', loss_value, sync_dist=self.sync_dist)
        self.log_metric('test', data=g, raw_scores=raw_scores, target=target)
        return loss_value