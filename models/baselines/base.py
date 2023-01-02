from models.base_model import DummyClass
from dgl import DGLGraph
import torch
from models.dgl_edge import DGLEdgeLoss

class UntrainableClass(DummyClass):
    """
    Base class for any untrainable models (for instance, any baseline)
    """
    def __init__(self, batch_size=None, sync_dist=True):
        super().__init__(batch_size, sync_dist)

    def on_train_batch_start(self, batch, batch_idx: int, unused: int = 0) -> int:
        return -1 #To prevent any training
    
    def training_step(self, *args, **kwargs):
        pass

    def train_dataloader(self, *args, **kwargs):
        pass

    def configure_optimizers(self, *args, **kwargs):
        pass

class Edge_NodeDegree(UntrainableClass):

    def __init__(self, batch_size=None, sync_dist=True):
        super().__init__(batch_size, sync_dist)
        self.loss = DGLEdgeLoss(normalize=torch.nn.Identity())
    
    def forward(self, x: DGLGraph):
        assert torch.all(x.in_degrees()==x.out_degrees()), "Graph is not symmetric !"
        degrees = x.in_degrees().to(float)
        degrees_norm = (degrees - degrees.mean())/degrees.std()
        degrees_sigm = degrees_norm.sigmoid()
        degrees_sigm = degrees_sigm.unsqueeze(-1)
        degrees_combi = degrees_sigm@degrees_sigm.transpose(-2,-1)
        proba_edge = degrees_combi[x.edges()].unsqueeze(-1)
        final = torch.cat((1-proba_edge, proba_edge), dim=-1)
        return final
    
    def test_step(self, batch, batch_idx, dataloader_idx=None):
        g, target = batch
        raw_scores = self(g)
        loss_value = self.loss(raw_scores, target)
        self.log('test_loss', loss_value, sync_dist=self.sync_dist)
        self.log_metric('test', data=g, raw_scores=raw_scores, target=target)
        return loss_value

        