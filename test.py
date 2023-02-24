import torch
import yaml
from argparse import ArgumentParser
from pytorch_lightning import Trainer
from datasets.text_pc_dm import TextPCDataModule
from models import CLIPPCWrapper
from torchvision.models import resnet50
from transformers import AutoTokenizer, AutoModel
import clip
from pytorch_lightning.callbacks import ModelCheckpoint
from datasets.shapenet_data_pc import ShapeNet15kPointClouds
import os
from utils.visualize import *
from collections import Counter
from models.model import CLIPPC

synsetid_to_cate = {
    '02691156': 'airplane', '02773838': 'bag', '02801938': 'basket',
    '02808440': 'bathtub', '02818832': 'bed', '02828884': 'bench',
    '02876657': 'bottle', '02880940': 'bowl', '02924116': 'bus',
    '02933112': 'cabinet', '02747177': 'can', '02942699': 'camera',
    '02954340': 'cap', '02958343': 'car', '03001627': 'chair',
    '03046257': 'clock', '03207941': 'dishwasher', '03211117': 'monitor',
    '04379243': 'table', '04401088': 'telephone', '02946921': 'tin_can',
    '04460130': 'tower', '04468005': 'train', '03085013': 'keyboard',
    '03261776': 'earphone', '03325088': 'faucet', '03337140': 'file',
    '03467517': 'guitar', '03513137': 'helmet', '03593526': 'jar',
    '03624134': 'knife', '03636649': 'lamp', '03642806': 'laptop',
    '03691459': 'speaker', '03710193': 'mailbox', '03759954': 'microphone',
    '03761084': 'microwave', '03790512': 'motorcycle', '03797390': 'mug',
    '03928116': 'piano', '03938244': 'pillow', '03948459': 'pistol',
    '03991062': 'pot', '04004475': 'printer', '04074963': 'remote_control',
    '04090263': 'rifle', '04099429': 'rocket', '04225987': 'skateboard',
    '04256520': 'sofa', '04330267': 'stove', '04530566': 'vessel',
    '04554684': 'washer', '02992529': 'cellphone',
    '02843684': 'birdhouse', '02871439': 'bookshelf',
    # '02858304': 'boat', no boat in our dataset, merged into vessels
    # '02834778': 'bicycle', not in our taxonomy
}
classes = list(synsetid_to_cate.values())
if __name__ == '__main__':
    # pl_model = CLIPPCWrapper.load_from_checkpoint(
        # "lightning_logs/version_13754438/checkpoints/epoch=4-step=44635.ckpt")
    sssss = "cuda" if torch.cuda.is_available() else "cpu"   
    config_dir = 'models/configs/PC.yaml'
    with open(config_dir) as fin:
        config = yaml.safe_load(fin)["PC-B"]
    
    # ckpt = "lightning_logs/version_13812725/checkpoints/epoch=9-step=89270.ckpt"
    ckpt = "lightning_logs/version_2/checkpoints/epoch=50-step=113781.ckpt"
    # ckpt = "lightning_logs/version_13812725/checkpoints/epoch=9-step=89270.ckpt"
    # ckpt = "lightning_logs/version_13942716/checkpoints/epoch=10-step=49093.ckpt"
    checkpoint = torch.load(ckpt)
    # pl_model = CLIPPCWrapper("PC-B", config, None, 4)
    # pl_model.load_state_dict(checkpoint['state_dict'])
    # model = pl_model.model.to(device)
    model = CLIPPC(**config)
    state_dict = {}
    for k, v in checkpoint['state_dict'].items():
        if k.startswith('model'):
            state_dict[k[6:]] = v
    # print(checkpoint['state_dict'].keys())
    model.load_state_dict(state_dict)
    model = model.to(device)
    print(len(state_dict))
    print(len([*model.parameters()]))
    exit(0)
    

    # data_dir = os.path.join(os.getenv("SLURM_TMPDIR"), "ShapeNetCore.v2.PC15k")
    data_dir = "../ShapeNetCore.v2.PC15k"
    dataset = ShapeNet15kPointClouds(root_dir= data_dir,
            categories=['all'], split='val',
            tr_sample_size=2048,
            te_sample_size=2048,
            scale=1.,
            normalize_per_shape=False,
            normalize_std_per_axis=False,
            random_subsample=False)

    sum = Counter()
    acc = Counter()
    acc_all = 0
    print(len(dataset))


    for i in range(100):
        pc, text = dataset[i]
        # image, class_id = cifar100[3637]
        image_input = pc.unsqueeze(0).to(device)
        text_inputs = torch.cat([clip.tokenize(f"a 3d model of {c}") for c in classes]).to(device)

        # Calculate features
        with torch.no_grad():
            image_features = model.encode_pc(image_input)
            text_features = model.encode_text(text_inputs)

        # Pick the top 5 most similar labels for the image
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        values, indices = similarity[0].topk(5)

        # Print the result

        # print(text)
        # print(text)
        # # print("\nTop predictions:\n")
        # # for value, index in zip(values, indices):
        # #     print(f"{classes[index]:>16s}: {100 * value.item():.2f}%")

        if classes[indices[0]] == text:
            acc[text] = acc[text] + 1
            acc_all = acc_all + 1
        sum[text] = sum[text] + 1
        
        # # visualize_pointcloud_batch('test.png' ,
        # #                     pc.repeat(25, 1, 1).transpose(1,2), None, None,
        # #                     None)
        if i % 100 == 0:
            print(i)
    for key in sum.keys():
        print(f"{key:>16s} : {100 * acc[key] / sum[key]:.2f}%")
        
    print(acc)
    print(sum)
    print(acc_all)
