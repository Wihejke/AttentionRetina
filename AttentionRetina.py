"""
AttenRetina — DGX Training Script (Clean Final Version)
ResNet50 + FPN + CBAM + Focal Loss + SmoothL1 + Attention Loss
Dataset: MS COCO 2017
"""

import os, math, json, time, warnings
warnings.filterwarnings('ignore')

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
import torchvision
import torchvision.transforms.functional as TF
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict

print(">> All imports OK", flush=True)

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
class Config:
    TRAIN_IMAGES   = '/nfsshare/users/arunbalaji/train2017'
    VAL_IMAGES     = '/nfsshare/users/arunbalaji/val2017'
    TRAIN_ANN      = '/nfsshare/users/arunbalaji/annotations/instances_train2017.json'
    VAL_ANN        = '/nfsshare/users/arunbalaji/annotations/instances_val2017.json'
    CHECKPOINT_DIR = '/nfsshare/users/arunbalaji/rahul_project/checkpoints'

    IMAGE_SIZE     = 512
    NUM_CLASSES    = 80
    NUM_ANCHORS    = 9
    FPN_CHANNELS   = 256

    ANCHOR_SIZES   = [32, 64, 128, 256, 512]
    ANCHOR_SCALES  = [1.0, 2**(1/3), 2**(2/3)]
    ANCHOR_RATIOS  = [0.5, 1.0, 2.0]

    POS_IOU_THRESH = 0.5
    NEG_IOU_THRESH = 0.4

    FOCAL_ALPHA    = 0.25
    FOCAL_GAMMA    = 2.0

    BATCH_SIZE     = 16
    NUM_WORKERS    = 4
    LR             = 1e-4
    WEIGHT_DECAY   = 1e-4
    NUM_EPOCHS     = 50
    GRAD_CLIP      = 1.0
    PRINT_EVERY    = 50
    EVAL_EVERY     = 5
    SAVE_EVERY     = 1

    SCORE_THRESH   = 0.05
    NMS_THRESH     = 0.5
    MAX_DETS       = 100

cfg = Config()
os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# DEVICE SETUP
# ──────────────────────────────────────────────────────────────────────────────
print(">> Setting up device...", flush=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}", flush=True)
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB", flush=True)

# Verify paths
print(">> Checking paths...", flush=True)
all_ok = True
for name, p in [('Train images', cfg.TRAIN_IMAGES), ('Val images', cfg.VAL_IMAGES),
                ('Train ann',    cfg.TRAIN_ANN),    ('Val ann',   cfg.VAL_ANN)]:
    status = 'OK' if os.path.exists(p) else 'MISSING'
    if status == 'MISSING': all_ok = False
    print(f'[{status}] {name}: {p}', flush=True)
if not all_ok:
    raise FileNotFoundError("One or more dataset paths are missing.")

# ──────────────────────────────────────────────────────────────────────────────
# DATASET
# ──────────────────────────────────────────────────────────────────────────────
def build_coco_label_map(coco):
    cat_ids = sorted(coco.getCatIds())
    return {cat_id: idx for idx, cat_id in enumerate(cat_ids)}


class COCODataset(Dataset):
    def __init__(self, images_path, annotation_path, image_size=512, is_train=True):
        self.images_path = images_path
        self.image_size  = image_size
        self.is_train    = is_train
        print(f'Loading annotations from {annotation_path} ...', flush=True)
        self.coco      = COCO(annotation_path)
        self.label_map = build_coco_label_map(self.coco)
        all_ids = list(self.coco.imgs.keys())
        self.image_ids = []
        for img_id in all_ids:
            info    = self.coco.loadImgs(img_id)[0]
            path    = os.path.join(self.images_path, info['file_name'])
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            if os.path.exists(path) and len(ann_ids) > 0:
                self.image_ids.append(img_id)
        print(f'Usable images: {len(self.image_ids)}', flush=True)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id   = self.image_ids[idx]
        info     = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.images_path, info['file_name'])
        image    = cv2.imread(img_path)
        image    = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image.shape[:2]
        image    = cv2.resize(image, (self.image_size, self.image_size))
        scale_x  = self.image_size / orig_w
        scale_y  = self.image_size / orig_h
        ann_ids  = self.coco.getAnnIds(imgIds=img_id)
        anns     = self.coco.loadAnns(ann_ids)
        boxes, labels = [], []
        for ann in anns:
            if ann.get('iscrowd', 0): continue
            x, y, w, h = ann['bbox']
            x1 = max(0.0, x * scale_x)
            y1 = max(0.0, y * scale_y)
            x2 = min(self.image_size, (x + w) * scale_x)
            y2 = min(self.image_size, (y + h) * scale_y)
            if x2 - x1 < 1 or y2 - y1 < 1: continue
            boxes.append([x1, y1, x2, y2])
            labels.append(self.label_map[ann['category_id']])
        if len(boxes) == 0:
            boxes  = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,),   dtype=torch.int64)
        else:
            boxes  = torch.tensor(boxes,  dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
        image = torch.as_tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
        mean  = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std   = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image = (image - mean) / std
        return image, {'boxes': boxes, 'labels': labels, 'image_id': img_id}


def collate_fn(batch):
    images  = torch.stack([item[0] for item in batch], dim=0)
    targets = [item[1] for item in batch]
    return images, targets


# ──────────────────────────────────────────────────────────────────────────────
# MODEL
# ──────────────────────────────────────────────────────────────────────────────
class ResNet50Backbone(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        backbone    = torchvision.models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)
        self.stem   = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        for m in [self.stem, self.layer1]:
            for p in m.parameters():
                p.requires_grad = False

    def forward(self, x):
        x  = self.stem(x)
        x  = self.layer1(x)
        c3 = self.layer2(x)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return c3, c4, c5


class FPN(nn.Module):
    def __init__(self, channels=256):
        super().__init__()
        self.lat_c3   = nn.Conv2d(512,  channels, 1)
        self.lat_c4   = nn.Conv2d(1024, channels, 1)
        self.lat_c5   = nn.Conv2d(2048, channels, 1)
        self.alpha_p4 = nn.Parameter(torch.tensor(0.5))
        self.alpha_p3 = nn.Parameter(torch.tensor(0.5))
        self.out_p3   = nn.Conv2d(channels, channels, 3, padding=1)
        self.out_p4   = nn.Conv2d(channels, channels, 3, padding=1)
        self.out_p5   = nn.Conv2d(channels, channels, 3, padding=1)
        self.p6_conv  = nn.Conv2d(2048,    channels, 3, stride=2, padding=1)
        self.p7_conv  = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, c3, c4, c5):
        lat3 = self.lat_c3(c3)
        lat4 = self.lat_c4(c4)
        lat5 = self.lat_c5(c5)
        a4   = self.alpha_p4.clamp(0, 1)
        a3   = self.alpha_p3.clamp(0, 1)
        p5   = lat5
        p4   = a4 * lat4 + (1 - a4) * F.interpolate(p5, size=lat4.shape[-2:], mode='nearest')
        p3   = a3 * lat3 + (1 - a3) * F.interpolate(p4, size=lat3.shape[-2:], mode='nearest')
        p3   = self.out_p3(p3)
        p4   = self.out_p4(p4)
        p5   = self.out_p5(p5)
        p6   = self.p6_conv(c5)
        p7   = self.p7_conv(F.relu(p6))
        return p3, p4, p5, p6, p7


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        mid = max(channels // reduction, 1)
        self.fc = nn.Sequential(nn.Conv2d(channels, mid, 1, bias=False), nn.ReLU(),
                                nn.Conv2d(mid, channels, 1, bias=False))

    def forward(self, x):
        return torch.sigmoid(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x)))


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x):
        avg_out = x.mean(dim=1, keepdim=True)
        max_out = x.max(dim=1, keepdim=True).values
        return torch.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channel_att = ChannelAttention(channels, reduction)
        self.spatial_att = SpatialAttention()

    def forward(self, x):
        x = x * self.channel_att(x)
        x = x * self.spatial_att(x)
        return x


class PyramidAttention(nn.Module):
    def __init__(self, channels=256):
        super().__init__()
        self.cbam_p3 = CBAM(channels)
        self.cbam_p4 = CBAM(channels)
        self.cbam_p5 = CBAM(channels)
        self.cbam_p6 = CBAM(channels)
        self.cbam_p7 = CBAM(channels)


class ClassificationHead(nn.Module):
    def __init__(self, num_classes=80, num_anchors=9, in_channels=256):
        super().__init__()
        layers = []
        for _ in range(4):
            layers += [nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
                       nn.GroupNorm(32, in_channels), nn.ReLU(inplace=True)]
        self.conv   = nn.Sequential(*layers)
        self.output = nn.Conv2d(in_channels, num_anchors * num_classes, 3, padding=1)
        nn.init.constant_(self.output.bias, -math.log((1 - 0.01) / 0.01))

    def forward(self, x):
        return self.output(self.conv(x))


class RegressionHead(nn.Module):
    def __init__(self, num_anchors=9, in_channels=256):
        super().__init__()
        layers = []
        for _ in range(4):
            layers += [nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
                       nn.GroupNorm(32, in_channels), nn.ReLU(inplace=True)]
        self.conv   = nn.Sequential(*layers)
        self.output = nn.Conv2d(in_channels, num_anchors * 4, 3, padding=1)

    def forward(self, x):
        return self.output(self.conv(x))


class AttenRetinaNet(nn.Module):
    def __init__(self, num_classes=80, num_anchors=9):
        super().__init__()
        self.backbone  = ResNet50Backbone(pretrained=True)
        self.fpn       = FPN(channels=cfg.FPN_CHANNELS)
        self.attention = PyramidAttention(channels=cfg.FPN_CHANNELS)
        self.cls_head  = ClassificationHead(num_classes, num_anchors, cfg.FPN_CHANNELS)
        self.reg_head  = RegressionHead(num_anchors, cfg.FPN_CHANNELS)

    def forward(self, x, return_att_maps=False):
        c3, c4, c5     = self.backbone(x)
        p3, p4, p5, p6, p7 = self.fpn(c3, c4, c5)
        pyramid_in     = [p3, p4, p5, p6, p7]
        cbams          = [self.attention.cbam_p3, self.attention.cbam_p4,
                          self.attention.cbam_p5, self.attention.cbam_p6,
                          self.attention.cbam_p7]
        pyramid_out    = []
        spatial_att_maps = []
        for i, cbam in enumerate(cbams):
            feat  = cbam.channel_att(pyramid_in[i]) * pyramid_in[i]
            s_map = cbam.spatial_att(feat)
            feat  = feat * s_map
            pyramid_out.append(feat)
            spatial_att_maps.append(s_map)
        cls_outputs = [self.cls_head(p) for p in pyramid_out]
        reg_outputs = [self.reg_head(p) for p in pyramid_out]
        if return_att_maps:
            return cls_outputs, reg_outputs, spatial_att_maps
        return cls_outputs, reg_outputs


# ──────────────────────────────────────────────────────────────────────────────
# ANCHORS  — generated ONCE, reused every batch
# ──────────────────────────────────────────────────────────────────────────────
def generate_anchors_for_level(feat_h, feat_w, stride, base_size, scales, ratios):
    anchors = []
    for scale in scales:
        for ratio in ratios:
            size = base_size * scale
            w    = size * math.sqrt(ratio)
            h    = size / math.sqrt(ratio)
            for gy in range(feat_h):
                for gx in range(feat_w):
                    cx = (gx + 0.5) * stride
                    cy = (gy + 0.5) * stride
                    anchors.append([cx - w/2, cy - h/2, cx + w/2, cy + h/2])
    return torch.tensor(anchors, dtype=torch.float32)


def precompute_anchors(image_size, cfg):
    """Call once before training. Returns (anchors_cpu, anchors_gpu)."""
    strides   = [8, 16, 32, 64, 128]
    feat_sizes = [image_size // s for s in strides]
    all_anchors = []
    for i, (feat_h, feat_w) in enumerate(zip(feat_sizes, feat_sizes)):
        anchors = generate_anchors_for_level(
            feat_h, feat_w, strides[i],
            cfg.ANCHOR_SIZES[i], cfg.ANCHOR_SCALES, cfg.ANCHOR_RATIOS
        )
        all_anchors.append(anchors)
    anchors_cpu = torch.cat(all_anchors, dim=0)
    anchors_gpu = anchors_cpu.to(device)
    return anchors_cpu, anchors_gpu


def flatten_predictions(cls_outputs, reg_outputs, num_classes, num_anchors):
    all_cls, all_reg = [], []
    for cls_out, reg_out in zip(cls_outputs, reg_outputs):
        B, _, H, W = cls_out.shape
        all_cls.append(cls_out.permute(0, 2, 3, 1).contiguous().view(B, H * W * num_anchors, num_classes))
        all_reg.append(reg_out.permute(0, 2, 3, 1).contiguous().view(B, H * W * num_anchors, 4))
    return torch.cat(all_cls, dim=1), torch.cat(all_reg, dim=1)


# ──────────────────────────────────────────────────────────────────────────────
# ANCHOR MATCHING & BOX ENCODING  — run on CPU
# ──────────────────────────────────────────────────────────────────────────────
def compute_iou_cpu(anchors, gt_boxes):
    """Both tensors must be on CPU."""
    ax1, ay1, ax2, ay2 = anchors[:, 0], anchors[:, 1], anchors[:, 2], anchors[:, 3]
    gx1, gy1, gx2, gy2 = gt_boxes[:, 0], gt_boxes[:, 1], gt_boxes[:, 2], gt_boxes[:, 3]
    ix1 = torch.max(ax1.unsqueeze(1), gx1.unsqueeze(0))
    iy1 = torch.max(ay1.unsqueeze(1), gy1.unsqueeze(0))
    ix2 = torch.min(ax2.unsqueeze(1), gx2.unsqueeze(0))
    iy2 = torch.min(ay2.unsqueeze(1), gy2.unsqueeze(0))
    inter = (ix2 - ix1).clamp(0) * (iy2 - iy1).clamp(0)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_g = (gx2 - gx1) * (gy2 - gy1)
    union  = area_a.unsqueeze(1) + area_g.unsqueeze(0) - inter
    return inter / union.clamp(min=1e-6)


def match_anchors(anchors_cpu, gt_boxes_cpu, gt_labels_cpu, pos_thresh, neg_thresh):
    """All inputs on CPU. Returns CPU tensors."""
    N = anchors_cpu.shape[0]
    assigned_labels = torch.full((N,), -1, dtype=torch.int64)
    assigned_boxes  = torch.zeros((N, 4), dtype=torch.float32)
    if gt_boxes_cpu.shape[0] == 0:
        assigned_labels[:] = 0
        return assigned_labels, assigned_boxes
    iou = compute_iou_cpu(anchors_cpu, gt_boxes_cpu)
    max_iou, best_gt = iou.max(dim=1)
    assigned_labels[max_iou < neg_thresh] = 0
    pos_mask = max_iou >= pos_thresh
    assigned_labels[pos_mask] = gt_labels_cpu[best_gt[pos_mask]] + 1
    assigned_boxes[pos_mask]  = gt_boxes_cpu[best_gt[pos_mask]]
    best_anchor_per_gt = iou.argmax(dim=0)
    for gt_idx in range(gt_boxes_cpu.shape[0]):
        a_idx = best_anchor_per_gt[gt_idx]
        assigned_labels[a_idx] = gt_labels_cpu[gt_idx] + 1
        assigned_boxes[a_idx]  = gt_boxes_cpu[gt_idx]
    return assigned_labels, assigned_boxes


def encode_boxes(anchors, gt_boxes):
    """anchors and gt_boxes on same device (GPU)."""
    aw = anchors[:, 2] - anchors[:, 0]; ah = anchors[:, 3] - anchors[:, 1]
    ax = anchors[:, 0] + 0.5 * aw;     ay = anchors[:, 1] + 0.5 * ah
    gw = gt_boxes[:, 2] - gt_boxes[:, 0]; gh = gt_boxes[:, 3] - gt_boxes[:, 1]
    gx = gt_boxes[:, 0] + 0.5 * gw;       gy = gt_boxes[:, 1] + 0.5 * gh
    tx = (gx - ax) / aw;  ty = (gy - ay) / ah
    tw = torch.log(gw / aw.clamp(min=1e-6))
    th = torch.log(gh / ah.clamp(min=1e-6))
    return torch.stack([tx, ty, tw, th], dim=1)


def decode_boxes(anchors, deltas):
    aw = anchors[:, 2] - anchors[:, 0]; ah = anchors[:, 3] - anchors[:, 1]
    ax = anchors[:, 0] + 0.5 * aw;     ay = anchors[:, 1] + 0.5 * ah
    tx, ty = deltas[:, 0], deltas[:, 1]
    tw = deltas[:, 2].clamp(max=4.0);  th = deltas[:, 3].clamp(max=4.0)
    cx = tx * aw + ax; cy = ty * ah + ay
    w  = torch.exp(tw) * aw; h = torch.exp(th) * ah
    return torch.stack([cx - 0.5*w, cy - 0.5*h, cx + 0.5*w, cy + 0.5*h], dim=1)


# ──────────────────────────────────────────────────────────────────────────────
# LOSS FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, num_classes=80):
        super().__init__()
        self.alpha = alpha; self.gamma = gamma; self.num_classes = num_classes

    def forward(self, logits, labels):
        # logits: [N, C] on GPU,  labels: [N] on GPU  (0=bg, 1-80=fg, -1=ignore)
        valid_mask = labels >= 0
        logits = logits[valid_mask]
        labels = labels[valid_mask]
        targets = torch.zeros_like(logits)
        fg_mask = labels > 0
        if fg_mask.sum() > 0:
            targets[fg_mask, labels[fg_mask] - 1] = 1.0
        prob    = torch.sigmoid(logits)
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t     = prob * targets + (1 - prob) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_w = alpha_t * (1 - p_t) ** self.gamma
        num_pos = fg_mask.sum().clamp(min=1).float()
        return (focal_w * ce_loss).sum() / num_pos


class RegressionLoss(nn.Module):
    def forward(self, reg_pred, reg_targets, assigned_labels_gpu):
        pos_mask = assigned_labels_gpu > 0
        if pos_mask.sum() == 0:
            return torch.tensor(0.0, device=reg_pred.device, requires_grad=True)
        return F.smooth_l1_loss(reg_pred[pos_mask], reg_targets[pos_mask], beta=1.0/9, reduction='mean')


class AttentionLoss(nn.Module):
    def forward(self, att_maps, anchors_gpu, assigned_labels_batch_gpu, cls_outputs, cfg_ref):
        total_loss = torch.tensor(0.0, device=att_maps[0].device)
        image_size = cfg_ref.IMAGE_SIZE
        anchor_cx  = ((anchors_gpu[:, 0] + anchors_gpu[:, 2]) / 2) / image_size * 2 - 1
        anchor_cy  = ((anchors_gpu[:, 1] + anchors_gpu[:, 3]) / 2) / image_size * 2 - 1
        level_sizes = [cls_outputs[i].shape[2] * cls_outputs[i].shape[3] * cfg_ref.NUM_ANCHORS
                       for i in range(len(att_maps))]
        splits = [0]
        for s in level_sizes:
            splits.append(splits[-1] + s)
        batch_size = att_maps[0].shape[0]
        num_counted = 0
        for b_idx in range(batch_size):
            assigned  = assigned_labels_batch_gpu[b_idx]
            level_losses = []
            for lvl, att in enumerate(att_maps):
                a_start, a_end = splits[lvl], splits[lvl + 1]
                lbl = assigned[a_start:a_end]
                cx  = anchor_cx[a_start:a_end]
                cy  = anchor_cy[a_start:a_end]
                valid_lvl = lbl >= 0
                if valid_lvl.sum() == 0: continue
                grid = torch.stack([cx[valid_lvl], cy[valid_lvl]], dim=1).unsqueeze(0).unsqueeze(0)
                sampled = F.grid_sample(att[b_idx:b_idx+1], grid, align_corners=True,
                                        mode='bilinear', padding_mode='border').squeeze()
                if sampled.dim() == 0:
                    sampled = sampled.unsqueeze(0)
                target = (lbl[valid_lvl] > 0).float()
                # Use autocast(enabled=False) — BCE is unsafe inside autocast
                with torch.cuda.amp.autocast(enabled=False):
                    level_losses.append(F.binary_cross_entropy(
                        sampled.float().clamp(1e-6, 1 - 1e-6),
                        target.float(),
                        reduction='mean'
                    ))
            if level_losses:
                total_loss = total_loss + torch.stack(level_losses).mean()
                num_counted += 1
        if num_counted > 0:
            total_loss = total_loss / num_counted
        return total_loss


class AdaptiveLossWeights(nn.Module):
    def __init__(self, momentum=0.99):
        super().__init__()
        self.momentum = momentum
        self.register_buffer('sigma2_cls', torch.tensor(1.0))
        self.register_buffer('sigma2_reg', torch.tensor(1.0))
        self.register_buffer('sigma2_att', torch.tensor(1.0))

    def forward(self, l_cls, l_reg, l_att):
        if self.training:
            self.sigma2_cls = (self.momentum * self.sigma2_cls + (1 - self.momentum) * l_cls.detach()**2).clamp(min=1e-4)
            self.sigma2_reg = (self.momentum * self.sigma2_reg + (1 - self.momentum) * l_reg.detach()**2).clamp(min=1e-4)
            self.sigma2_att = (self.momentum * self.sigma2_att + (1 - self.momentum) * l_att.detach()**2).clamp(min=1e-4)
        lam1 = 1.0 / self.sigma2_cls
        lam2 = 1.0 / self.sigma2_reg
        lam3 = 1.0 / self.sigma2_att
        return lam1 * l_cls + lam2 * l_reg + lam3 * l_att, lam1.item(), lam2.item(), lam3.item()


# ──────────────────────────────────────────────────────────────────────────────
# CHECKPOINT UTILITIES
# ──────────────────────────────────────────────────────────────────────────────
def save_checkpoint(model, optimizer, scaler, epoch, best_map, history, path):
    torch.save({
        'epoch'          : epoch,
        'model_state'    : model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scaler_state'   : scaler.state_dict(),
        'best_map'       : best_map,
        'history'        : history
    }, path)
    print(f'  [CKPT] Saved → {path}', flush=True)


def load_checkpoint(path, model, optimizer=None, scaler=None):
    ckpt    = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    if optimizer     and 'optimizer_state' in ckpt: optimizer.load_state_dict(ckpt['optimizer_state'])
    if scaler        and 'scaler_state'    in ckpt: scaler.load_state_dict(ckpt['scaler_state'])
    history = ckpt.get('history', None)
    print(f'  [CKPT] Resumed from epoch {ckpt["epoch"]+1}  |  best_mAP={ckpt["best_map"]:.4f}', flush=True)
    return ckpt['epoch'], ckpt['best_map'], history


# ──────────────────────────────────────────────────────────────────────────────
# TRAINING STEP
# ──────────────────────────────────────────────────────────────────────────────
def training_step(images, targets, model, optimizer, scaler,
                  cls_loss_fn, reg_loss_fn, att_loss_fn,
                  anchors_cpu, anchors_gpu):
    model.train()
    images = images.to(device)
    optimizer.zero_grad()

    with autocast():
        cls_outputs, reg_outputs, att_maps = model(images, return_att_maps=True)
        all_cls, all_reg = flatten_predictions(cls_outputs, reg_outputs, cfg.NUM_CLASSES, cfg.NUM_ANCHORS)

        total_cls_loss        = torch.tensor(0.0, device=device)
        total_reg_loss        = torch.tensor(0.0, device=device)
        assigned_labels_batch = []   # list of GPU tensors

        for i in range(len(targets)):
            # Keep GT on CPU for matching
            gt_boxes_cpu  = targets[i]['boxes']    # CPU
            gt_labels_cpu = targets[i]['labels']   # CPU

            # Match on CPU
            assigned_labels_cpu, assigned_boxes_cpu = match_anchors(
                anchors_cpu, gt_boxes_cpu, gt_labels_cpu,
                cfg.POS_IOU_THRESH, cfg.NEG_IOU_THRESH
            )

            # Move results to GPU for loss computation
            assigned_labels_gpu = assigned_labels_cpu.to(device)
            assigned_boxes_gpu  = assigned_boxes_cpu.to(device)
            assigned_labels_batch.append(assigned_labels_gpu)

            # Encode regression targets on GPU
            reg_targets = encode_boxes(anchors_gpu, assigned_boxes_gpu)

            total_cls_loss += cls_loss_fn(all_cls[i], assigned_labels_gpu)
            total_reg_loss += reg_loss_fn(all_reg[i], reg_targets, assigned_labels_gpu)

        total_cls_loss /= len(targets)
        total_reg_loss /= len(targets)

        total_att_loss = att_loss_fn(
            att_maps, anchors_gpu, assigned_labels_batch, cls_outputs, cfg
        )

        loss = total_cls_loss + total_reg_loss + total_att_loss
        lam1, lam2, lam3 = 1.0, 1.0, 1.0

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
    scaler.step(optimizer)
    scaler.update()

    return loss.item(), total_cls_loss.item(), total_reg_loss.item(), total_att_loss.item()


# ──────────────────────────────────────────────────────────────────────────────
# COCO EVALUATION
# ──────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate_coco(model, val_loader, val_dataset, cfg, anchors_gpu):
    model.eval()
    results    = []
    cat_ids    = sorted(val_dataset.coco.getCatIds())
    idx_to_cat = {idx: cat_id for idx, cat_id in enumerate(cat_ids)}

    for images, targets in tqdm(val_loader, desc='  Evaluating', leave=False):
        images = images.to(device)
        with autocast():
            cls_outputs, reg_outputs = model(images)
        all_cls, all_reg = flatten_predictions(cls_outputs, reg_outputs, cfg.NUM_CLASSES, cfg.NUM_ANCHORS)
        for i in range(images.shape[0]):
            img_id   = targets[i]['image_id']
            boxes    = decode_boxes(anchors_gpu, all_reg[i]).clamp(0, cfg.IMAGE_SIZE)
            scores   = torch.sigmoid(all_cls[i])
            for cls_idx in range(cfg.NUM_CLASSES):
                cls_scores = scores[:, cls_idx]
                mask = cls_scores > cfg.SCORE_THRESH
                if mask.sum() == 0: continue
                kb   = boxes[mask]; ks = cls_scores[mask]
                keep = torchvision.ops.nms(kb, ks, cfg.NMS_THRESH)[:cfg.MAX_DETS]
                kb   = kb[keep].cpu(); ks = ks[keep].cpu()
                for box, score in zip(kb, ks):
                    x1, y1, x2, y2 = box.tolist()
                    results.append({'image_id': int(img_id), 'category_id': int(idx_to_cat[cls_idx]),
                                    'bbox': [round(x1,2), round(y1,2), round(x2-x1,2), round(y2-y1,2)],
                                    'score': round(float(score), 4)})

    if len(results) == 0:
        print('  No detections — all metrics = 0', flush=True)
        return {'mAP': 0.0, 'mAP50': 0.0, 'Precision': 0.0, 'Recall': 0.0, 'F1': 0.0}

    coco_dt   = val_dataset.coco.loadRes(results)
    coco_eval = COCOeval(val_dataset.coco, coco_dt, 'bbox')
    coco_eval.evaluate(); coco_eval.accumulate(); coco_eval.summarize()
    map50_95 = float(coco_eval.stats[0])
    map50    = float(coco_eval.stats[1])

    # Precision / Recall / F1
    dets_by_img = defaultdict(list)
    for r in results: dets_by_img[r['image_id']].append(r)
    total_tp = total_fp = total_fn = 0
    for img_id, dets in dets_by_img.items():
        ann_ids  = val_dataset.coco.getAnnIds(imgIds=img_id)
        gts      = val_dataset.coco.loadAnns(ann_ids)
        gt_boxes = np.array([[g['bbox'][0], g['bbox'][1], g['bbox'][0]+g['bbox'][2], g['bbox'][1]+g['bbox'][3]]
                              for g in gts if not g.get('iscrowd', 0)], dtype=np.float32)
        dets_sorted = sorted(dets, key=lambda x: -x['score'])
        det_boxes   = np.array([[d['bbox'][0], d['bbox'][1], d['bbox'][0]+d['bbox'][2], d['bbox'][1]+d['bbox'][3]]
                                 for d in dets_sorted], dtype=np.float32)
        n_gt = len(gt_boxes); n_det = len(det_boxes)
        if n_gt == 0 and n_det == 0: continue
        if n_gt == 0: total_fp += n_det; continue
        if n_det == 0: total_fn += n_gt; continue
        iou_mat    = compute_iou_cpu(torch.tensor(det_boxes), torch.tensor(gt_boxes)).numpy()
        matched_gt = set()
        for d_idx in range(n_det):
            best_iou, best_gt_j = cfg.POS_IOU_THRESH, -1
            for g_idx in range(n_gt):
                if g_idx not in matched_gt and iou_mat[d_idx, g_idx] >= best_iou:
                    best_iou = iou_mat[d_idx, g_idx]; best_gt_j = g_idx
            if best_gt_j >= 0: total_tp += 1; matched_gt.add(best_gt_j)
            else: total_fp += 1
        total_fn += n_gt - len(matched_gt)

    precision = total_tp / max(total_tp + total_fp, 1)
    recall    = total_tp / max(total_tp + total_fn, 1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-9)
    metrics   = {'mAP': round(map50_95,4), 'mAP50': round(map50,4),
                 'Precision': round(precision,4), 'Recall': round(recall,4), 'F1': round(f1,4)}
    print(f'  Precision={precision:.4f}  Recall={recall:.4f}  F1={f1:.4f}  '
          f'mAP@0.5={map50:.4f}  mAP@0.5:0.95={map50_95:.4f}', flush=True)
    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
def main():
    print(">> Building datasets...", flush=True)
    train_dataset = COCODataset(cfg.TRAIN_IMAGES, cfg.TRAIN_ANN, cfg.IMAGE_SIZE, is_train=True)
    val_dataset   = COCODataset(cfg.VAL_IMAGES,   cfg.VAL_ANN,   cfg.IMAGE_SIZE, is_train=False)

    print(">> Building dataloaders...", flush=True)
    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True,
                              num_workers=cfg.NUM_WORKERS, collate_fn=collate_fn, pin_memory=True)
    val_loader   = DataLoader(val_dataset, batch_size=4, shuffle=False,
                              num_workers=cfg.NUM_WORKERS, collate_fn=collate_fn, pin_memory=True)
    print(f'Train batches: {len(train_loader)}  |  Val batches: {len(val_loader)}', flush=True)

    print(">> Building model...", flush=True)
    model        = AttenRetinaNet(cfg.NUM_CLASSES, cfg.NUM_ANCHORS).to(device)
    optimizer    = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    scaler       = GradScaler()
    scheduler    = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.NUM_EPOCHS, eta_min=1e-6)
    cls_loss_fn  = FocalLoss(cfg.FOCAL_ALPHA, cfg.FOCAL_GAMMA, cfg.NUM_CLASSES)
    reg_loss_fn  = RegressionLoss()
    att_loss_fn  = AttentionLoss()

    print(">> Pre-computing anchors...", flush=True)
    anchors_cpu, anchors_gpu = precompute_anchors(cfg.IMAGE_SIZE, cfg)
    print(f'Anchors shape: {anchors_cpu.shape}', flush=True)

    # Resume from checkpoint if exists
    start_epoch = 0
    best_map    = 0.0
    history     = {'train_loss': [], 'cls_loss': [], 'reg_loss': [], 'att_loss': [],
                   'mAP50': [], 'mAP': [], 'Precision': [], 'Recall': [], 'F1': []}
    latest_ckpt = os.path.join(cfg.CHECKPOINT_DIR, 'latest.pth')
    if os.path.exists(latest_ckpt):
        start_epoch, best_map, saved_history = load_checkpoint(
            latest_ckpt, model, optimizer, scaler
        )
        if saved_history: history = saved_history
        start_epoch += 1
        for _ in range(start_epoch):
            scheduler.step()

    print(f'\nStarting from epoch {start_epoch+1}/{cfg.NUM_EPOCHS}', flush=True)
    print(f'Device: {device}  |  Batch: {cfg.BATCH_SIZE}  |  Image: {cfg.IMAGE_SIZE}', flush=True)
    print('─' * 70, flush=True)

    for epoch in range(start_epoch, cfg.NUM_EPOCHS):
        epoch_start = time.time()
        total_loss  = total_cls = total_reg = total_att = 0.0
        num_batches = 0

        for batch_idx, (images, targets) in enumerate(train_loader):
            try:
                loss, cls_l, reg_l, att_l = training_step(
                    images, targets, model, optimizer, scaler,
                    cls_loss_fn, reg_loss_fn, att_loss_fn,
                    anchors_cpu, anchors_gpu
                )
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    torch.cuda.empty_cache()
                    print(f'  [OOM] Skipping batch {batch_idx}', flush=True)
                    continue
                raise e

            total_loss += loss; total_cls += cls_l; total_reg += reg_l; total_att += att_l
            num_batches += 1

            if (batch_idx + 1) % cfg.PRINT_EVERY == 0:
                print(f'  Ep {epoch+1} | Batch {batch_idx+1}/{len(train_loader)} '
                      f'| Loss={loss:.4f} cls={cls_l:.4f} reg={reg_l:.4f} att={att_l:.4f}',
                      flush=True)

        scheduler.step()

        avg_loss = total_loss / max(num_batches, 1)
        avg_cls  = total_cls  / max(num_batches, 1)
        avg_reg  = total_reg  / max(num_batches, 1)
        avg_att  = total_att  / max(num_batches, 1)
        elapsed  = time.time() - epoch_start

        history['train_loss'].append(avg_loss)
        history['cls_loss'].append(avg_cls)
        history['reg_loss'].append(avg_reg)
        history['att_loss'].append(avg_att)

        print(f'\nEpoch {epoch+1}/{cfg.NUM_EPOCHS} | Avg Loss={avg_loss:.4f} '
              f'cls={avg_cls:.4f} reg={avg_reg:.4f} att={avg_att:.4f} '
              f'| LR={scheduler.get_last_lr()[0]:.2e} | Time={elapsed:.0f}s', flush=True)

        # Save checkpoint every epoch
        save_checkpoint(model, optimizer, scaler,
                        epoch, best_map, history, latest_ckpt)
        epoch_ckpt = os.path.join(cfg.CHECKPOINT_DIR, f'epoch_{epoch+1:03d}.pth')
        save_checkpoint(model, optimizer, scaler,
                        epoch, best_map, history, epoch_ckpt)

        # COCO eval every EVAL_EVERY epochs
        if (epoch + 1) % cfg.EVAL_EVERY == 0:
            torch.cuda.empty_cache()
            metrics = evaluate_coco(model, val_loader, val_dataset, cfg, anchors_gpu)
            map50   = metrics['mAP50']
            for k in ['mAP50', 'mAP', 'Precision', 'Recall', 'F1']:
                history[k].append(metrics[k])
            print(f'  mAP@0.50={map50:.4f}  best so far={best_map:.4f}', flush=True)
            if map50 > best_map:
                best_map  = map50
                best_ckpt = os.path.join(cfg.CHECKPOINT_DIR, 'best.pth')
                save_checkpoint(model, optimizer, scaler,
                                epoch, best_map, history, best_ckpt)
                print(f'  *** New best mAP: {best_map:.4f} ***', flush=True)

        print('─' * 70, flush=True)

    print(f'\nTraining complete. Best mAP@0.50: {best_map:.4f}', flush=True)

    # Save training curves
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    ax = axes[0]
    ax.plot(history['train_loss'], label='Total'); ax.plot(history['cls_loss'], '--', label='Focal')
    ax.plot(history['reg_loss'],   ':',  label='Reg'); ax.plot(history['att_loss'], '-.', label='Att')
    ax.set_title('Training Loss'); ax.legend(); ax.grid(True, alpha=0.3)
    eval_x = list(range(cfg.EVAL_EVERY, cfg.NUM_EPOCHS+1, cfg.EVAL_EVERY))[:len(history['mAP50'])]
    for ax, (key, label, color) in zip(axes[1:],
        [('mAP50','mAP@0.5','green'), ('mAP','mAP@0.5:0.95','blue'),
         ('Precision','Precision','orange'), ('Recall','Recall','red'), ('F1','F1','purple')]):
        ax.plot(eval_x, history[key][:len(eval_x)], marker='o', color=color)
        ax.set_title(label); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    curves_path = os.path.join(cfg.CHECKPOINT_DIR, 'training_curves.png')
    plt.savefig(curves_path, dpi=150)
    print(f'Training curves saved to {curves_path}', flush=True)

    if history['mAP50']:
        print('\n' + '='*50)
        print('Final Results (Paper Table 4 format)')
        print('='*50)
        for k, v in [('Precision', history['Precision'][-1]), ('Recall', history['Recall'][-1]),
                     ('F1', history['F1'][-1]), ('mAP@0.50', history['mAP50'][-1]),
                     ('mAP@0.5:0.95', history['mAP'][-1])]:
            print(f'{k:<15} {v:.4f}')
        print('='*50)


if __name__ == '__main__':
    main()