import hydra
import pytorch_lightning as pl

from unet import Unet
from gaussian_diffusion import GaussianDiffusion
from ddpm import DDPM
from dataset.mvtec import MVTecDataModule


@hydra.main(config_name='config')
def main(cfg):
    unet = Unet(dim=cfg.unet_dim)
    diffusion = GaussianDiffusion(unet, image_size=(cfg.height, cfg.width))
    ddpm = DDPM(diffusion, sample_every_n_epochs=cfg.sample_every_n_epochs,)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(every_n_epochs=cfg.save_every_n_epochs)
    datamodule = MVTecDataModule(cfg.mvtec_root, cfg.category, cfg.batchsize)

    trainer = pl.Trainer(
        min_epochs=cfg.epochs,
        max_epochs=cfg.epochs,
        gpus=1,
        callbacks=[checkpoint_callback]
    )

    trainer.fit(ddpm, datamodule)


if __name__ == '__main__':
    main()