import torch
import yaml
from argparse import ArgumentParser
from pytorch_lightning import Trainer
from datasets.text_pc_dm import TextPCDataModule
from models import CLIPPCWrapper
from torchvision.models import resnet50
from transformers import AutoTokenizer, AutoModel
from datasets.shapenet_data_pc import ShapeNet15kPointClouds
from models.model import CLIPPC,CLIP
import clip


def main(hparams):
    config_dir = 'models/configs/PC.yaml'
    with open(config_dir) as fin:
        config = yaml.safe_load(fin)[hparams.model_name]
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    pretrained_model, preprocess = clip.load("ViT-B/32",device=device,jit=False)

    txt_encoder = pretrained_model.transformer

    if hparams.minibatch_size < 1:
        hparams.minibatch_size = hparams.batch_size

    model = CLIPPCWrapper(hparams.model_name, config, txt_encoder, hparams.minibatch_size)
    del hparams.model_name

    dm = TextPCDataModule.from_argparse_args(hparams)
   
    trainer = Trainer.from_argparse_args(hparams, max_epochs=32, accelerator="gpu")
    trainer.fit(model, dm)


if __name__ == '__main__':
    model = CLIPPC(512,2048,512,8,12,77,49408,512,8,12)
    # model = CLIP(512,224,12,768,16,77,49408,512,8,12)
    print(model.visual.resblocks[0].attn.in_proj_weight.dtype)