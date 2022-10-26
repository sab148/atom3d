import torch
from pytorch_lightning import LightningModule
from torchmetrics import MinMetric
from torchmetrics import MeanSquaredError
from typing import Any, List

class SMPLitModule(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["net"])

        self.net = net
        
        # loss function
        self.criterion = torch.nn.MSELoss()



    def forward(self, x: torch.Tensor):
        print("model",nextself.net.parameters()).is_cuda)
        return self.net(x)


    def step(self, batch: Any):

        preds = self.forward(batch)
        loss = self.criterion(preds, batch.y)
        return loss, preds, batch.y

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log train metrics
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size= preds.size()[0])

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log val metrics
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size= preds.size()[0])

        return {"loss": loss, "preds": preds, "targets": targets}


    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # # log test metrics
        # self.log("test/loss", loss, on_step=False, on_epoch=True, batch_size= preds.size()[0])
        # self.log("preds", preds, on_step=False, on_epoch=True, prog_bar=True, batch_size= preds.size()[0])
        # self.log("targets", targets, on_step=False, on_epoch=True, prog_bar=True, batch_size= preds.size()[0])

        return {"loss": loss, "preds": preds, "targets": targets}

    def predict_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log test metrics
        # self.log("test/loss", loss, on_step=False, on_epoch=True, batch_size= preds.size()[0])

        return {"loss": loss, "preds": preds, "targets": targets}
    

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer