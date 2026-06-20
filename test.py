from argparse import ArgumentParser
import os


from tqdm import tqdm
import cv2
import logging
import numpy as np

import torch
from torch.utils.data import DataLoader

from datasets.seg_dataset import SegDataset
from utils import build_model
from metric import ConfuseMatrixMeter

# Hainan palette 5 classes
# palette = {
#     0: (35, 145, 255),     # Water 水系
#     1: (235, 55, 55),      # Road 道路
#     2: (255, 185, 35),     # Residential 居民地
#     3: (50, 175, 70),      # Vegetation 植被
#     4: (170, 120, 85)      # Background 背景
# }

# # UAVid palette 8 classes
# palette = {0 : (128, 0, 0),         # Building
#            1 : (128, 64, 128),      # Road
#            2 : (0, 128, 0),         # Tree
#            3 : (128, 128, 0),       # Low vegetation
#            4 : (64, 0, 128),        # Moving car
#            5 : (192, 0, 192),       # Static car
#            6 : (64, 64, 0),         # Human
#            7 : (0, 0, 0)}           # Background clutter

# # LoveDA palette 7 classes
# palette = {0 : (255, 255, 255),     # Background
#            1 : (255, 0, 0),         # Building
#            2 : (255, 255, 0),       # Road
#            3 : (0, 0, 255),         # Water
#            4 : (159, 129, 183),     # Barren
#            5 : (0, 255, 0),         # Forest
#            6 : (255, 195, 128)}     # Agricultural

# Vaihingen/Potsdam palette 6 classes
# palette = {0 : (255, 255, 255),     # Impervious surfaces
#            1 : (0, 0, 255),         # Buildings
#            2 : (0, 255, 255),       # Low vegetation
#            3 : (0, 255, 0),         # Tree
#            4 : (255, 204, 0),       # Car
#            5 : (255, 0, 0)}         # Clutter/background  

def label2rgb(mask):
    rgb_image = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for k, v in palette.items():
        rgb_image[mask == k] = v
    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    return rgb_image


def parse_arguments():
    parser = ArgumentParser()
    
    parser.add_argument('--test_dir', default='/testdata', type=str)
    parser.add_argument('--result_root', default='results', type=str)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--vis', default=True, type=bool)

    parser.add_argument('--checkpoint_root', default='checkpoints', type=str)
    parser.add_argument('--project_name', default=None, type=str)
    parser.add_argument('--checkpoint_name', default='best_test_mF1.pt', type=str)

    args = parser.parse_args()

    return args


if __name__== "__main__" :
    args = parse_arguments()

    args.checkpoint_path = os.path.join(args.checkpoint_root, args.project_name, args.checkpoint_name)
    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError('no such checkpoint %s' % args.checkpoint_path)

    if args.vis:
        args.result_dir = os.path.join(args.result_root, args.project_name)
        os.makedirs(args.result_dir, exist_ok=True)
        os.makedirs(args.result_dir + '/rgb', exist_ok=True)
        
        logging.basicConfig(filename=os.path.join(args.result_dir, 'test.log'),
                        level=logging.DEBUG, format='%(asctime)s %(levelname)s: %(message)s',
                        filemode='w', force=True)
        logging.info(args)

    test_data = SegDataset(args.test_dir, is_train=False)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    date_str, time_str, *encoder, architecture = args.project_name.split('_')
    encoder = '_'.join(encoder)

    model = build_model(encoder, architecture, in_channels=3, n_class=5)
    # model = HyperGraphUNet(encoder, num_classes=8)
    # model = SAM2HGUNet(encoder, num_classes=8)
    # model = load_model()
    # model = pvig_b_1024_gelu_fpn()


    model.load_state_dict(torch.load(args.checkpoint_path, weights_only=True))
    model.to(device)

    model.eval()

    running_metric = ConfuseMatrixMeter(n_class=5)
    
    with torch.no_grad():
        for batch in tqdm(test_loader):
            images, labels = batch['image'].to(device), batch['label'].to(device)
       
            pred = model(images)

            pred = pred.argmax(dim=1, keepdim=True)

            running_metric.update_cm(pr=pred.cpu().numpy(), gt=labels.cpu().numpy())

            if args.vis:
                nBatch, _, _, _ = pred.shape
                for i in range(nBatch):
                    result_path = os.path.join(args.result_dir, batch['name'][i])
                    result = (pred[i]).squeeze().cpu().numpy().astype('uint8')
                    cv2.imwrite(result_path, result)

                    rgb_path = os.path.join(args.result_dir + '/rgb', batch['name'][i])
                    cv2.imwrite(rgb_path, label2rgb(result)) 
    
        running_metric.show_scores(logger=logging)


          