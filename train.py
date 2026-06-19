from argparse import ArgumentParser
import os


import datetime
import time
import logging
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets.seg_dataset import SegDataset
from utils import train, test, build_model, get_optimizer, get_scheduler
from loss import *


def parse_arguments():
    parser = ArgumentParser()

    parser.add_argument('--encoder', default='dinov3_vitl16', type=str)
    parser.add_argument('--architecture', default='HyperGraphUPerNet', type=str)
    parser.add_argument('--checkpoint_root', default='checkpoints', type=str)
    parser.add_argument('--pretrained', default=None, type=str)
    parser.add_argument('--checkpoint_name', default='best_test_mIoU.pt', type=str)

    parser.add_argument('--data_name', default='uavid', type=str)
    parser.add_argument('--train_dir', default='/train', type=str)
    parser.add_argument('--test_dir', default='/test', type=str)

    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--num_epochs', default=80, type=int)
    parser.add_argument('--learning_rate', default=3e-4, type=float)
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--scheduler', default='cos', type=str)
    
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_arguments()

    args.project_name = args.encoder + '_' + args.architecture

    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    args.checkpoint_dir = os.path.join(args.checkpoint_root, current_time + '_' + args.project_name)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    logging.basicConfig(filename=os.path.join(args.checkpoint_dir, 'train.log'),
                        level=logging.DEBUG, format='%(asctime)s %(levelname)s: %(message)s',
                        filemode='w', force=True)
    logging.info(args)

    train_data = SegDataset(args.train_dir)
    test_data = SegDataset(args.test_dir, is_train=False)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    model = build_model(encoder=args.encoder, architecture=args.architecture, in_channels=3, n_class=8)

    if args.pretrained:
        args.pretrained_path = os.path.join(args.checkpoint_root, args.pretrained, args.checkpoint_name)
        if os.path.exists(args.pretrained_path):
            model.load_state_dict(torch.load(args.pretrained_path))
            print('Successfully loaded pre-trained model: ' + args.pretrained_path)

   
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)   
    # model = nn.DataParallel(model)

    optimizer = get_optimizer(model, name=args.optimizer, lr=args.learning_rate)
    scheduler = get_scheduler(optimizer, name=args.scheduler, max_epochs=args.num_epochs)

    criterion = JointLoss(SoftCrossEntropyLoss(smooth_factor=0.05), DiceLoss(smooth=0.05), 1.0, 1.0)

    best_test_miou = 0
    for epoch in range(args.num_epochs):
        start_time = time.time()

        current_lr = scheduler.get_last_lr()[0]
        print(f'Epoch {epoch+1}, Learning Rate: {current_lr:.6f}') 

        train_loss = train(model, train_loader, optimizer, criterion, device)
        test_loss, scores_dict = test(model, test_loader, criterion, device)
        print('Epoch: {}, Train Loss: {:.4f}, Test Loss: {:.4f}'.format(epoch+1, train_loss, test_loss))

        scheduler.step()

        test_miou = scores_dict['miou']
        if best_test_miou < test_miou:        
            best_test_miou = test_miou
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, args.checkpoint_name ))
        print('Epoch: {}, current epoch mIoU: {:.4f}, Best test mIoU: {:.4f}'.format(epoch+1, test_miou, best_test_miou))
        torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, 'last.pt'))

        end_time = time.time()
        logging.info('Epoch: {}, Train Loss: {:.4f}, Test Loss: {:.4f}, current epoch mIoU: {:.4f}, Best test mIoU: {:.4f}, Time: {:.2f}s'
                     .format(epoch+1, train_loss, test_loss, test_miou, best_test_miou, end_time-start_time))
