import torch
import yaml
from argparse import ArgumentParser
from pytorch_lightning import Trainer
from datasets.text_pc_dm import TextPCDataModule
from models import CLIPPCWrapper
from torchvision.models import resnet50
from transformers import AutoTokenizer, AutoModel
import clip

def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float() 

def main(hparams):
    config_dir = 'models/configs/PC.yaml'
    with open(config_dir) as fin:
        config = yaml.safe_load(fin)[hparams.model_name]
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    pretrained_model, preprocess = clip.load("ViT-B/32",device=device,jit=False)
    convert_models_to_fp32(pretrained_model)

    txt_encoder = pretrained_model.transformer
    # txt_encoder = None

    if hparams.minibatch_size < 1:
        hparams.minibatch_size = hparams.batch_size

    model = CLIPPCWrapper(hparams.model_name, config, txt_encoder, hparams.minibatch_size)
    del hparams.model_name

    dm = TextPCDataModule.from_argparse_args(hparams)
   
    trainer = Trainer.from_argparse_args(hparams, precision=32, max_epochs=100, accelerator="gpu")
    trainer.fit(model, dm)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--minibatch_size', type=int, default=0)
    parser = TextPCDataModule.add_argparse_args(parser)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    main(args)
