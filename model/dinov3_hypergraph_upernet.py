import torch
import torch.nn as nn
import torch.nn.functional as F

from dinov3_adapter import DINOv3_Adapter

DINOV3_REPO_DIR = 'dinov3'
DINOV3_CHECKPOINTS = DINOV3_REPO_DIR + '/checkpoints/'

BACKBONE_INTERMEDIATE_LAYERS = {
    "dinov3_vits16": [2, 5, 8, 11],
    "dinov3_vitb16": [2, 5, 8, 11],
    "dinov3_vitl16": [4, 11, 17, 23],
    "dinov3_vit7b16": [9, 19, 29, 39],
}

DINOV3_WEIGHTS = {
    # Pretrained models

    # ViT models pretrained on web dataset (LVD-1689M)
    'dinov3_vits16': 'dinov3_vits16_pretrain_lvd1689m-08c60483.pth',
    'dinov3_vits16plus': 'dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth',
    'dinov3_vitb16': 'dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth',
    'dinov3_vitl16': 'dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth',
    'dinov3_vith16plus': 'dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth',
    'dinov3_vit7b16': 'dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth',

    # ConvNeXt models pretrained on web dataset (LVD-1689M)
    'dinov3_convnext_tiny': 'dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth',
    'dinov3_convnext_small': 'dinov3_convnext_small_pretrain_lvd1689m-296db49d.pth',
    'dinov3_convnext_base': 'dinov3_convnext_base_pretrain_lvd1689m-801f2ba9.pth',
    'dinov3_convnext_large': 'dinov3_convnext_large_pretrain_lvd1689m-61fa432d.pth',

    # Vision Transformer models on satellite dataset (SAT-493M)
    'dinov3_vitl16_sat': 'dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth',
    'dinov3_vit7b16_sat': 'dinov3_vit7b16_pretrain_sat493m-a6675841.pth',

    # Pretrained heads - Image classification
    'dinov3_vit7b16_imagenet1k_linear_head': 'dinov3_vit7b16_imagenet1k_linear_head-90d8ed92.pth',

    # Pretrained heads - Depth trained on SYNTHMIX dataset
    'dinov3_vit7b16_synthmix_dpt_head': 'dinov3_vit7b16_synthmix_dpt_head-02040be1.pth',

    # Pretrained heads - Detector trained on COCO2017 dataset
    'dinov3_vit7b16_coco_detr_head': 'dinov3_vit7b16_coco_detr_head-b0235ff7.pth',

    # Pretrained heads - Segmentor trained on ADE20K dataset
    'dinov3_vit7b16_ade20k_m2f': 'dinov3_vit7b16_ade20k_m2f_head-bf307cb1.pth',

    # Pretrained heads - Zero-shot tasks with dino.txt
    'dinov3_vitl16_dinotxt_tet1280d20h24l': 'dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5',
}


class PPM(nn.Module):
    def __init__(self, in_channels, pool_sizes=[1, 2, 3, 6], reduction_dim=512):
        super(PPM, self).__init__()
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(ps),
                nn.Conv2d(in_channels, reduction_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduction_dim),
                nn.ReLU(inplace=True)
            ) for ps in pool_sizes
        ])

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        pyramids = [x]
        for stage in self.stages:
            y = stage(x)
            y = F.interpolate(y, size=(h, w), mode='bilinear', align_corners=False)
            pyramids.append(y)
        out = torch.cat(pyramids, dim=1)
        return out
  

class FPN(nn.Module):
    def __init__(self, feature_channels, out_channels=256):
        super(FPN, self).__init__()
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for in_channels in feature_channels:
            self.lateral_convs.append(nn.Conv2d(in_channels, out_channels, kernel_size=1))
            self.fpn_convs.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))

    def forward(self, features):
        laterals = [l_conv(f) for l_conv, f in zip(self.lateral_convs, features)]

        for i in range(len(laterals) - 1, 0, -1):
            upsample = F.interpolate(laterals[i], size=laterals[i - 1].shape[2:], mode='bilinear', align_corners=False)
            laterals[i - 1] += upsample

        outs = [fpn_conv(lateral) for fpn_conv, lateral in zip(self.fpn_convs, laterals)]
        return outs
    

class DINOv3_Segmentation(nn.Module):
    def __init__(self, *, backbone_name="dinov3_vitl16", num_classes=2):
        super().__init__()
        backbone_model = torch.hub.load(DINOV3_REPO_DIR, backbone_name, source='local',
                                        weights=DINOV3_CHECKPOINTS + DINOV3_WEIGHTS[backbone_name])

        self.backbone = DINOv3_Adapter(
            backbone_model,
            interaction_indexes=BACKBONE_INTERMEDIATE_LAYERS[backbone_name],
        )

        self.num_classes = num_classes
        embed_dim = self.backbone.embed_dim
        print("Backbone embed_dim:", embed_dim)
        feature_channels = [embed_dim, embed_dim, embed_dim, embed_dim]

        self.ppm = PPM(in_channels=feature_channels[-1], reduction_dim=512)
        self.ppm_out = feature_channels[-1] + 4 * 512  # original + 4 pooled features

        # FPN
        self.fpn = FPN(feature_channels[:-1] + [self.ppm_out], out_channels=256)

        # Segmentation head
        self.classifier = nn.Sequential(
            nn.Conv2d(256 * len(feature_channels), 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, num_classes, kernel_size=1)
        )


    def forward(self, x):
        f1, f2, f3, f4 = self.backbone(x)
    
        # PPM
        ppm_out = self.ppm(f4)

        # FPN
        features = [f1, f2, f3, ppm_out]
        fpn_outs = self.fpn(features)

        # Upsample to same size
        target_size = fpn_outs[0].shape[2:]
        upsampled = [F.interpolate(f, size=target_size, mode='bilinear', align_corners=False) for f in fpn_outs]

        fused = torch.cat(upsampled, dim=1)
        out = self.classifier(fused)

        # Final upsample to input size
        out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)

        return out


class DINOv3_Base_Segmentation(nn.Module):
    def __init__(self, *, backbone_name="dinov3_vitl16", num_classes=8):
        super().__init__()
        backbone_model = torch.hub.load(DINOV3_REPO_DIR, backbone_name, source='local',
                                        weights=DINOV3_CHECKPOINTS + DINOV3_WEIGHTS[backbone_name])

        self.backbone = DINOv3_Adapter(
            backbone_model,
            interaction_indexes=BACKBONE_INTERMEDIATE_LAYERS[backbone_name],
        )

        self.num_classes = num_classes
        embed_dim = self.backbone.embed_dim
        print("Backbone embed_dim:", embed_dim)

        self.classifier = nn.Conv2d(embed_dim, num_classes, kernel_size=3, padding=1)

    def forward(self, x):
        f1, f2, f3, f4 = self.backbone(x)
    
        out = self.classifier(f4)

        # Final upsample to input size
        out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)

        return out

if __name__ == "__main__":
    model = DINOv3_Segmentation(backbone_name="dinov3_vitl16", num_classes=8)
    model
    x = torch.randn(2, 3, 1024, 1024)
    output = model(x)
    print(output.shape)

    def count_parameters(model):
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"Total parameters: {total/1e6:.0f}M")
        print(f"Trainable parameters: {trainable/1e6:.2f}M")
        print(f"Frozen parameters: {(total - trainable)/1e6:.0f}M")

    count_parameters(model)
