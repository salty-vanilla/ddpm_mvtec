import copy
from functools import partial

import torch
import torchvision
import pytorch_lightning as pl

from ema import EMA
from helper import loss_backwards

try:
    from apex import amp
    APEX_AVAILABLE = True
except:
    APEX_AVAILABLE = False


class DDPM(pl.LightningModule):
    def __init__(
        self,
        diffusion_model: torch.nn.Module,
        lr: float=2e-5,
        ema_decay: float=0.995,
        gradient_accumulate_every: int=2,
        fp16: bool=False,
        start_ema_epoch:int =2000,
        update_ema_every: int =10,
        sample_every_n_epochs: int =1000,
    ):
        super().__init__()
        self.diffusion_model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.diffusion_model)
        self.update_ema_every = update_ema_every
        self.lr = lr

        self.start_ema_epoch = start_ema_epoch
        self.sample_every_n_epochs = sample_every_n_epochs

        self.image_size = diffusion_model.image_size
        self.gradient_accumulate_every = gradient_accumulate_every

        assert not fp16 or fp16 and APEX_AVAILABLE, 'Apex must be installed in order for mixed precision training to be turned on'

        self.fp16 = fp16
        if fp16:
            (self.diffusion_model, self.ema_model), self.opt = amp.initialize([self.diffusion_model, self.ema_model], self.opt, opt_level='O1')

        self.automatic_optimization = False
        self.reset_parameters()

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.diffusion_model.state_dict())

    def step_ema(self):
        if self.current_epoch < self.start_ema_epoch:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.diffusion_model)

    def training_step(self, batch):
        opt = self.optimizers()

        x = batch.cuda()
        loss = self.diffusion_model.compute_loss(x)

        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()

        self.log_dict({'loss': loss}, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.diffusion_model.parameters(), lr=self.lr)
        return optimizer

    def manual_backward(self, loss: torch.Tensor, *args, **kwargs) -> None:
        backwards = partial(loss_backwards, self.fp16)
        return backwards(loss, self.optimizers())

    def on_train_epoch_end(self) -> None:
        if self.current_epoch % self.update_ema_every == 0:
            self.step_ema()
        if self.sample_every_n_epochs and \
           self.current_epoch != 0 and self.current_epoch % self.sample_every_n_epochs == 0:
            milestone = self.current_epoch // self.sample_every_n_epochs
            all_images = self.ema_model.sample(batch_size=16)
            all_images = (all_images + 1) * 0.5
            grid = torchvision.utils.make_grid(all_images)
            self.logger.experiment.add_image("generated_images", grid, self.current_epoch)
            # self.save(milestone)

            # self.step += 1
        return super().training_step_end()

    def training_epoch_end(self, outputs) -> None:
        return super().training_epoch_end(outputs)