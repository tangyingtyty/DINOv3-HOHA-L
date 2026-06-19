import os
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler

from metric import ConfuseMatrixMeter

import segmentation_models_pytorch as smp

from dinov3_hypergraph_upernet import DINOv3_Segmentation
from dinov2_hypergraph_upernet import DINOv2_Segmentation
from sam_hypergraph_upernet import SAM_Segmentation
from sam3_hypergraph_upernet import SAM3_Segmentation
from timm_hypergraph_upernet import TIMM_Segmentation

import ttach as tta
"""
supported encoders: 
['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 
 'resnext50_32x4d', 'resnext101_32x4d', 'resnext101_32x8d', 
 'resnext101_32x16d', 'resnext101_32x32d', 'resnext101_32x48d', 
 'dpn68', 'dpn68b', 'dpn92', 'dpn98', 'dpn107', 'dpn131', 
 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn', 
 'senet154', 'se_resnet50', 'se_resnet101', 'se_resnet152', 'se_resnext50_32x4d', 'se_resnext101_32x4d', 
 'densenet121', 'densenet169', 'densenet201', 'densenet161', 
 'inceptionresnetv2', 'inceptionv4', 
 'efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3', 
 'efficientnet-b4', 'efficientnet-b5', 'efficientnet-b6', 'efficientnet-b7',
 'mobilenet_v2', 'xception', 
 'timm-efficientnet-b0', 'timm-efficientnet-b1', 'timm-efficientnet-b2', 'timm-efficientnet-b3', 
 'timm-efficientnet-b4', 'timm-efficientnet-b5', 'timm-efficientnet-b6', 'timm-efficientnet-b7',
 'timm-efficientnet-b8', 'timm-efficientnet-l2', 
 'timm-tf_efficientnet_lite0', 'timm-tf_efficientnet_lite1', 'timm-tf_efficientnet_lite2',
 'timm-tf_efficientnet_lite3', 'timm-tf_efficientnet_lite4', 
 'timm-resnest14d', 'timm-resnest26d', 'timm-resnest50d', 'timm-resnest101e',
 'timm-resnest200e', 'timm-resnest269e', 'timm-resnest50d_4s2x40d', 
 'timm-resnest50d_1s4x24d', 'timm-res2net50_26w_4s', 'timm-res2net101_26w_4s', 
 'timm-res2net50_26w_6s', 'timm-res2net50_26w_8s', 'timm-res2net50_48w_2s',
 'timm-res2net50_14w_8s', 'timm-res2next50', 'timm-regnetx_002', 'timm-regnetx_004', 
 'timm-regnetx_006', 'timm-regnetx_008', 'timm-regnetx_016', 'timm-regnetx_032',
 'timm-regnetx_040', 'timm-regnetx_064', 'timm-regnetx_080', 'timm-regnetx_120',
 'timm-regnetx_160', 'timm-regnetx_320', 'timm-regnety_002', 'timm-regnety_004', 
 'timm-regnety_006', 'timm-regnety_008', 'timm-regnety_016', 'timm-regnety_032',
 'timm-regnety_040', 'timm-regnety_064', 'timm-regnety_080', 'timm-regnety_120',
 'timm-regnety_160', 'timm-regnety_320', 
 'timm-skresnet18', 'timm-skresnet34', 'timm-skresnext50_32x4d', 
 'timm-mobilenetv3_large_075', 'timm-mobilenetv3_large_100', 
 'timm-mobilenetv3_large_minimal_100', 'timm-mobilenetv3_small_075', 
 'timm-mobilenetv3_small_100', 'timm-mobilenetv3_small_minimal_100',
 'timm-gernet_s', 'timm-gernet_m', 'timm-gernet_l', 
 'mit_b0', 'mit_b1', 'mit_b2', 'mit_b3', 'mit_b4', 'mit_b5',
 'mobileone_s0', 'mobileone_s1', 'mobileone_s2', 'mobileone_s3', 'mobileone_s4']

supported architectures: 
['Unet', 'UnetPlusPlus', 'MAnet', 'Linknet', 'FPN',
 'PSPNet', 'DeepLabV3', 'DeepLabV3Plus', 'PAN']"
"""

def build_model(encoder, architecture, in_channels = 3, n_class = 8):
    if encoder == 'sam3':
        model = SAM3_Segmentation(backbone_name=encoder, num_classes=n_class)
    elif encoder.startswith('sam'):
        model = SAM_Segmentation(backbone_name=encoder, num_classes=n_class)
    elif encoder.startswith('dinov3'):
        model = DINOv3_Segmentation(backbone_name=encoder, num_classes=n_class)
    elif encoder.startswith('dinov2'):
        model = DINOv2_Segmentation(backbone_name=encoder, num_classes=n_class)
    elif encoder.startswith('vit'):
        model = TIMM_Segmentation(backbone_name=encoder, num_classes=n_class)
    elif architecture == 'Unet':
        model = smp.Unet(encoder_name= encoder, encoder_weights= "imagenet", in_channels=in_channels, classes=n_class)
    elif architecture == 'UnetPlusPlus':
        model = smp.UnetPlusPlus(encoder_name= encoder, encoder_weights= "imagenet", in_channels=in_channels, classes=n_class)
    elif architecture == 'MAnet':
        model = smp.MAnet(encoder_name= encoder, encoder_weights= "imagenet", in_channels=in_channels, classes=n_class)
    elif architecture == 'Linknet':
        model = smp.Linknet(encoder_name= encoder, encoder_weights= "imagenet", in_channels=in_channels, classes=n_class)
    elif architecture == 'FPN':
        model = smp.FPN(encoder_name= encoder, encoder_weights= "imagenet", in_channels=in_channels, classes=n_class)
    elif architecture == 'PSPNet':
        model = smp.PSPNet(encoder_name= encoder, encoder_weights= "imagenet", in_channels=in_channels, classes=n_class)
    elif architecture == 'DeepLabV3':
        model = smp.DeepLabV3(encoder_name= encoder, encoder_weights= "imagenet", in_channels=in_channels, classes=n_class)
    elif architecture == 'DeepLabV3Plus':
        model = smp.DeepLabV3Plus(encoder_name= encoder, encoder_weights= "imagenet", in_channels=in_channels, classes=n_class)
    elif architecture == 'PAN':
        model = smp.PAN(encoder_name= encoder, encoder_weights= "imagenet", in_channels=in_channels, classes=n_class)
    else:
        raise ValueError(f"Unsupported architecture: {architecture}. "
                        f"Supported architectures are: ['Unet', 'UnetPlusPlus', 'MAnet', 'Linknet', 'FPN', 'PSPNet', 'DeepLabV3', 'DeepLabV3Plus', 'PAN']")
    return model


def get_optimizer(model, name, lr):
        # define optimizers
    if name == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr,
                              momentum=0.9, weight_decay=5e-4)
    elif name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr,
                               weight_decay=0)
    elif name == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=lr,
                                betas=(0.9, 0.999), weight_decay=0.01)
    else:
        raise NotImplementedError('optimizer [%s] is not recognized' % name)
 
    return optimizer


def get_scheduler(optimizer, name, max_epochs):
    if name == 'linear':
        scheduler = lr_scheduler.LambdaLR(optimizer, 
                                          lr_lambda=lambda epoch: (1.0 - epoch / float(max_epochs + 1)))
    elif name == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=max_epochs // 3, gamma=0.1)
    elif name == 'poly':
        scheduler = lr_scheduler.LambdaLR(optimizer, 
                                          lr_lambda=lambda epoch: (1 - epoch / max_epochs) ** 0.9)
    elif name == 'cos':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    elif name == 'constant':
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', name)
    
    return scheduler


def train(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for batch in tqdm(train_loader):
        images, labels = batch['image'].to(device), batch['label'].to(device)
        optimizer.zero_grad()

        pred = model(images)

        loss_se = criterion(pred, labels.long())
    
        loss = loss_se 

        loss.backward()
        optimizer.step()
  
        running_loss += loss.item()
 
    return running_loss / len(train_loader)


def test(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_metric = ConfuseMatrixMeter(n_class=5)

    with torch.no_grad():
        for batch in tqdm(test_loader):
            images, labels = batch['image'].to(device), batch['label'].to(device)
            pred = model(images)
            loss =  criterion(pred, labels.long())
            running_loss += loss.item()         
            pred = pred.argmax(dim=1, keepdim=True)
            running_metric.update_cm(pr=pred.cpu().numpy(), gt=labels.cpu().numpy())

    return running_loss / len(test_loader), running_metric.get_scores()