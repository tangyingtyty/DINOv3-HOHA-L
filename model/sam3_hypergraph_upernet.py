import torch
import torch.nn as nn
import torch.nn.functional as F

from sam3.model_builder import build_sam3_image_model



SAM_WEIGHTS = {
    # Pretrained models
    'sam3': '/sam3/checkpoints/sam3.pt',
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
    

class SAM3_Segmentation(nn.Module):
    def __init__(self, *, backbone_name='sam3', num_classes=8):
        super().__init__()

        model = build_sam3_image_model(checkpoint_path=SAM_WEIGHTS['sam3'])

        del model.transformer
        del model.geometry_encoder
        del model.segmentation_head
        del model.dot_prod_scoring
        del model.backbone.language_backbone

        self.encoder = model.backbone.vision_backbone.trunk
        self.encoder.return_interm_layers = True

        self.backbone = SAM3_Adapter(self.encoder)

        self.num_classes = num_classes
        embed_dim = self.encoder.channel_list[0]
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


if __name__ == "__main__":
    model = SAM3_Segmentation(num_classes=8).cuda()
    x = torch.randn(2, 3, 1024, 1024).cuda()
    output = model(x)
    print(output.shape)

    def count_parameters(model):
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"Total parameters: {total/1e6:.0f}M")
        print(f"Trainable parameters: {trainable/1e6:.0f}M")
        print(f"Frozen parameters: {(total - trainable)/1e6:.0f}M")

    count_parameters(model)
