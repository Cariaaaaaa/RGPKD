"""
基于重建指导和提示模块的知识蒸馏方法（RGPKD）
"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import roc_auc_score
from scipy.ndimage import gaussian_filter

# 全局配置
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 256
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 800  # RGPKD训练轮次

class MVTecADDataset(Dataset):
    """MVTec AD数据集加载类"""
    def __init__(self, root_dir, category, train=True, transform=None):
        self.root_dir = root_dir
        self.category = category
        self.train = train
        self.transform = transform
        self.image_paths = []
        self.label_paths = []
        
        # 训练集（仅正常样本）
        if train:
            img_dir = os.path.join(root_dir, category, "train", "good")
            for img_name in os.listdir(img_dir):
                self.image_paths.append(os.path.join(img_dir, img_name))
                self.label_paths.append(None)  # 训练集无标签
        # 测试集（正常+异常样本）
        else:
            img_dir = os.path.join(root_dir, category, "test")
            for sub_dir in os.listdir(img_dir):
                sub_img_dir = os.path.join(img_dir, sub_dir)
                if not os.path.isdir(sub_img_dir):
                    continue
                
                # 异常标签路径
                label_dir = os.path.join(root_dir, category, "ground_truth", sub_dir) if sub_dir != "good" else None
                for img_name in os.listdir(sub_img_dir):
                    self.image_paths.append(os.path.join(sub_img_dir, img_name))
                    if sub_dir == "good":
                        self.label_paths.append(None)
                    else:
                        label_name = img_name.replace(".png", "_mask.png")
                        self.label_paths.append(os.path.join(label_dir, label_name))
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = None
        
        if self.label_paths[idx] is not None and os.path.exists(self.label_paths[idx]):
            label = Image.open(self.label_paths[idx]).convert("L")  # 灰度标签图
        
        if self.transform:
            image = self.transform(image)
            if label is not None:
                label = transforms.ToTensor()(label).squeeze(0)  # 转为张量并去除通道维
        
        # 标签：0=正常，1=异常
        img_label = 0 if self.label_paths[idx] is None else 1
        return image, label, img_label, img_path

def get_mvtec_dataloader(root_dir, category, train=True, batch_size=BATCH_SIZE):
    """获取数据集加载器"""
    if train:
        transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    dataset = MVTecADDataset(root_dir, category, train, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=4)
    return dataloader

def calculate_metrics(pred_scores, gt_labels, pred_masks=None, gt_masks=None):
    """计算评价指标：I-AUROC、P-AUROC、PRO"""
    # 图像级AUROC（I-AUROC）
    img_scores = [score.max() for score in pred_scores]  # 每张图的最大异常得分
    i_auroc = roc_auc_score(gt_labels, img_scores)
    
    # 像素级AUROC（P-AUROC）和PRO
    p_auroc = 0.0
    pro = 0.0
    
    if pred_masks is not None and gt_masks is not None:
        all_pred_pixels = []
        all_gt_pixels = []
        overlap_sum = 0.0
        count = 0
        
        for pred_mask, gt_mask in zip(pred_masks, gt_masks):
            if gt_mask is None:  # 正常样本，跳过像素级评估
                continue
            
            # 归一化到0-1
            pred_mask = (pred_mask - pred_mask.min()) / (pred_mask.max() - pred_mask.min() + 1e-8)
            gt_mask = (gt_mask > 0).float()  # 二值化标签
            
            # 像素级AUROC
            all_pred_pixels.extend(pred_mask.flatten().cpu().numpy())
            all_gt_pixels.extend(gt_mask.flatten().cpu().numpy())
            
            # PRO计算（预测与真实异常区域重叠率）
            intersection = (pred_mask > 0.5).float() * gt_mask
            union = (pred_mask > 0.5).float() + gt_mask - intersection
            overlap = (intersection.sum() / (union.sum() + 1e-8)).item()
            overlap_sum += overlap
            count += 1
        
        if len(all_pred_pixels) > 0:
            p_auroc = roc_auc_score(all_gt_pixels, all_pred_pixels)
        if count > 0:
            pro = overlap_sum / count
    
    return i_auroc, p_auroc, pro

def visualize_result(image, gt_mask, pred_mask, save_path=None):
    """可视化异常检测结果：原图、真实标签、预测热力图"""
    # 反归一化图像
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(DEVICE)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(DEVICE)
    image = image * std + mean
    image = image.permute(1, 2, 0).cpu().detach().numpy()
    image = (image * 255).astype(np.uint8)
    
    # 处理预测掩码
    pred_mask = (pred_mask - pred_mask.min()) / (pred_mask.max() - pred_mask.min() + 1e-8)
    pred_mask = pred_mask.cpu().detach().numpy()
    pred_heatmap = cv2.applyColorMap((pred_mask * 255).astype(np.uint8), cv2.COLORMAP_JET)
    pred_heatmap = cv2.cvtColor(pred_heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(image, 0.6, pred_heatmap, 0.4, 0)
    
    # 处理真实掩码
    gt_vis = image.copy()
    if gt_mask is not None:
        gt_mask = (gt_mask > 0).cpu().numpy()
        gt_vis[gt_mask] = [255, 0, 0]  # 红色标记真实异常区域
    
    # 绘图
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    
    axes[1].imshow(gt_vis)
    axes[1].set_title("Ground Truth Anomaly")
    axes[1].axis("off")
    
    axes[2].imshow(overlay)
    axes[2].set_title("Predicted Anomaly Heatmap")
    axes[2].axis("off")
    
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close()

class SSIMLoss(nn.Module):
    """结构相似度损失函数"""
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self._create_window(window_size, self.channel)
    
    def _create_window(self, window_size, channel):
        window = torch.ones(window_size, window_size) / (window_size * window_size)
        window = window.unsqueeze(0).unsqueeze(0).repeat(channel, 1, 1, 1)
        return window.to(DEVICE)
    
    def forward(self, img1, img2):
        mu1 = F.conv2d(img1, self.window, padding=self.window_size//2, groups=self.channel)
        mu2 = F.conv2d(img2, self.window, padding=self.window_size//2, groups=self.channel)
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(img1*img1, self.window, padding=self.window_size//2, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, self.window, padding=self.window_size//2, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, self.window, padding=self.window_size//2, groups=self.channel) - mu1_mu2
        
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        if self.size_average:
            return 1 - ssim_map.mean()
        else:
            return 1 - ssim_map.mean(1).mean(1).mean(1)

class PromptModule(nn.Module):
    """提示模块：多尺度特征匹配与融合"""
    def __init__(self, in_channels=512, mid_channels=256):
        super(PromptModule, self).__init__()
        self.theta = nn.Conv2d(in_channels, mid_channels, kernel_size=1, padding=0)
        self.phi = nn.Conv2d(in_channels, mid_channels, kernel_size=1, padding=0)
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(mid_channels*3, mid_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x, prompt_feature):
        """
        x: 输入特征 (B, C, H, W)
        prompt_feature: 提示库特征 (B, C, H_p, W_p)
        """
        # 特征投影
        theta_x = self.theta(x)  # (B, mid_C, H, W)
        phi_p = self.phi(prompt_feature)  # (B, mid_C, H_p, W_p)
        
        # 双向亲和度计算
        B, C, H, W = theta_x.shape
        _, _, Hp, Wp = phi_p.shape
        
        theta_x_flat = theta_x.view(B, C, -1).permute(0, 2, 1)  # (B, H*W, C)
        phi_p_flat = phi_p.view(B, C, -1)  # (B, C, Hp*Wp)
        
        aff_x2p = torch.bmm(theta_x_flat, phi_p_flat)  # (B, H*W, Hp*Wp)
        aff_p2x = torch.bmm(phi_p_flat.permute(0, 2, 1), theta_x_flat.permute(0, 2, 1).transpose(1,2))  # (B, Hp*Wp, H*W)
        
        # 关系特征构建
        aff_x2p = aff_x2p.view(B, 1, H, W, Hp*Wp).permute(0, 4, 2, 3, 1).view(B, Hp*Wp, H, W)
        aff_p2x = aff_p2x.view(B, 1, Hp, Wp, H*W).permute(0, 4, 2, 3, 1).view(B, H*W, Hp, Wp)
        aff_p2x = F.interpolate(aff_p2x, size=(H, W), mode='bilinear', align_corners=False)
        
        # 特征融合
        relation_feature = torch.cat([theta_x, 
                                     aff_x2p.mean(dim=1, keepdim=True).repeat(1, C//2, 1, 1),
                                     aff_p2x.mean(dim=1, keepdim=True).repeat(1, C//2, 1, 1)], dim=1)
        fused_feature = self.fusion_conv(relation_feature)
        return fused_feature

class UNetWithPrompt(nn.Module):
    """带提示模块的Unet学生网络"""
    def __init__(self, in_channels=3, out_channels=3, prompt_channels=256):
        super(UNetWithPrompt, self).__init__()
        
        # 编码器（ResNet18基础结构）
        resnet = models.resnet18(pretrained=True)
        self.encoder1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)  # (B, 64, 128, 128)
        self.encoder2 = nn.Sequential(resnet.maxpool, resnet.layer1)  # (B, 64, 64, 64)
        self.encoder3 = resnet.layer2  # (B, 128, 32, 32)
        self.encoder4 = resnet.layer3  # (B, 256, 16, 16)
        self.encoder5 = resnet.layer4  # (B, 512, 8, 8)
        
        # 提示模块
        self.prompt_module1 = PromptModule(in_channels=512, mid_channels=256)
        self.prompt_module2 = PromptModule(in_channels=256, mid_channels=128)
        self.prompt_module3 = PromptModule(in_channels=128, mid_channels=64)
        
        # 解码器
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )  # (B, 256, 16, 16)
        
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )  # (B, 128, 32, 32)
        
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )  # (B, 64, 64, 64)
        
        self.decoder4 = nn.Sequential(
            nn.ConvTranspose2d(64, 3, kernel_size=2, stride=2),
            nn.Sigmoid()
        )  # (B, 3, 128, 128)
        
        self.final_up = nn.Upsample(size=(IMAGE_SIZE, IMAGE_SIZE), mode='bilinear', align_corners=False)
        
        # 卷积融合层
        self.conv_fuse1 = nn.Conv2d(256+256, 256, kernel_size=3, padding=1)
        self.conv_fuse2 = nn.Conv2d(128+128, 128, kernel_size=3, padding=1)
        self.conv_fuse3 = nn.Conv2d(64+64, 64, kernel_size=3, padding=1)
    
    def forward(self, x, prompt_features=None):
        """
        x: 输入图像 (B, 3, 256, 256)
        prompt_features: 提示库特征 (B, 3, C, H, W) -> [F1, F2, F3]
        """
        # 编码过程
        e1 = self.encoder1(x)  # (B, 64, 128, 128)
        e2 = self.encoder2(e1)  # (B, 64, 64, 64)
        e3 = self.encoder3(e2)  # (B, 128, 32, 32)
        e4 = self.encoder4(e3)  # (B, 256, 16, 16)
        e5 = self.encoder5(e4)  # (B, 512, 8, 8)
        
        # 解码+提示模块融合
        d1 = self.decoder1(e5)  # (B, 256, 16, 16)
        if prompt_features is not None:
            d1 = self.prompt_module1(d1, prompt_features[2])  # F3对应16x16尺度
        d1 = torch.cat([d1, e4], dim=1)
        d1 = self.conv_fuse1(d1)  # (B, 256, 16, 16)
        
        d2 = self.decoder2(d1)  # (B, 128, 32, 32)
        if prompt_features is not None:
            d2 = self.prompt_module2(d2, prompt_features[1])  # F2对应32x32尺度
        d2 = torch.cat([d2, e3], dim=1)
        d2 = self.conv_fuse2(d2)  # (B, 128, 32, 32)
        
        d3 = self.decoder3(d2)  # (B, 64, 64, 64)
        if prompt_features is not None:
            d3 = self.prompt_module3(d3, prompt_features[0])  # F1对应64x64尺度
        d3 = torch.cat([d3, e2], dim=1)
        d3 = self.conv_fuse3(d3)  # (B, 64, 64, 64)
        
        d4 = self.decoder4(d3)  # (B, 3, 128, 128)
        rec_img = self.final_up(d4)  # (B, 3, 256, 256)
        
        # 返回多尺度解码器特征和重建图
        return [d3, d2, d1], rec_img

class RGPKDModel(nn.Module):
    """RGPKD整体模型：教师网络+学生网络"""
    def __init__(self):
        super(RGPKDModel, self).__init__()
        
        # 教师网络（预训练ResNet18，固定参数）
        self.teacher = models.resnet18(pretrained=True)
        for param in self.teacher.parameters():
            param.requires_grad = False
        
        # 学生网络（带提示模块的Unet）
        self.student = UNetWithPrompt()
        
        # 提示库（训练过程中构建）
        self.prompt_lib = None
    
    def extract_teacher_features(self, x):
        """提取教师网络多尺度特征"""
        features = []
        
        x = self.teacher.conv1(x)
        x = self.teacher.bn1(x)
        x = self.teacher.relu(x)
        x = self.teacher.maxpool(x)
        
        x = self.teacher.layer1(x)  # F1: (B, 64, 64, 64)
        features.append(x)
        
        x = self.teacher.layer2(x)  # F2: (B, 128, 32, 32)
        features.append(x)
        
        x = self.teacher.layer3(x)  # F3: (B, 256, 16, 16)
        features.append(x)
        
        return features  # [F1, F2, F3]
    
    def build_prompt_lib(self, dataloader):
        """构建提示库：教师网络提取的正常样本特征"""
        self.teacher.eval()
        prompt_lib = []
        
        with torch.no_grad():
            for imgs, _, _, _ in tqdm(dataloader, desc="Building Prompt Library"):
                imgs = imgs.to(DEVICE)
                feats = self.extract_teacher_features(imgs)
                prompt_lib.append(feats)
        
        # 按尺度拼接，形成提示库 (N, 3, C, H, W)
        prompt_lib = [torch.cat([feats[i] for feats in prompt_lib], dim=0) for i in range(3)]
        self.prompt_lib = prompt_lib
    
    def get_nearest_prompt(self, query_feat):
        """从提示库中查找最近邻特征（余弦相似度）"""
        query_feat = query_feat.view(query_feat.shape[0], query_feat.shape[1], -1).permute(0, 2, 1)
        prompt_feat = self.prompt_lib[2].view(self.prompt_lib[2].shape[0], self.prompt_lib[2].shape[1], -1).permute(0, 2, 1)
        
        similarities = []
        for q in query_feat:
            sim = torch.matmul(q, prompt_feat.permute(0, 2, 1)).mean(dim=1)
            similarities.append(sim)
        
        similarities = torch.stack(similarities, dim=0)
        nearest_idx = similarities.argmax(dim=1)
        
        # 返回最近邻的多尺度特征
        nearest_feats = [self.prompt_lib[i][nearest_idx].to(DEVICE) for i in range(3)]
        return nearest_feats
    
    def forward(self, x, train_mode=True):
        """
        train_mode: True-训练（使用提示库），False-推理（不使用提示库）
        """
        # 教师网络特征
        teacher_feats = self.extract_teacher_features(x)  # [F1, F2, F3]
        
        # 学生网络前向传播
        if train_mode and self.prompt_lib is not None:
            # 以学生网络e4作为查询（对应教师F3尺度）
            e4 = self.student.encoder4(self.student.encoder3(self.student.encoder2(self.student.encoder1(x))))
            nearest_feats = self.get_nearest_prompt(e4)
            student_feats, rec_img = self.student(x, nearest_feats)
        else:
            student_feats, rec_img = self.student(x)
        
        # 学生特征按尺度对应教师特征（反转顺序：学生[d3,d2,d1]对应教师[F1,F2,F3]）
        student_feats = student_feats[::-1]
        
        return teacher_feats, student_feats, rec_img

def train_rgpkd(root_dir, category):
    """训练RGPKD模型"""
    # 数据加载
    train_loader = get_mvtec_dataloader(root_dir, category, train=True)
    test_loader = get_mvtec_dataloader(root_dir, category, train=False)
    
    # 模型初始化
    model = RGPKDModel().to(DEVICE)
    model.build_prompt_lib(train_loader)  # 构建提示库
    
    # 损失函数与优化器
    mse_loss = nn.MSELoss()
    ssim_loss = SSIMLoss()
    optimizer = optim.SGD(model.student.parameters(), lr=LEARNING_RATE, momentum=0.9)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    # 训练过程
    best_i_auroc = 0.0
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        
        for imgs, _, _, _ in tqdm(train_loader, desc=f"RGPKD Epoch {epoch+1}/{EPOCHS}"):
            imgs = imgs.to(DEVICE)
            optimizer.zero_grad()
            
            # 前向传播
            teacher_feats, student_feats, rec_img = model(imgs, train_mode=True)
            
            # 基础损失（教师-学生特征MSE）
            loss_basic = 0.0
            for t_feat, s_feat in zip(teacher_feats, student_feats):
                loss_basic += mse_loss(s_feat, t_feat)
            
            # 重建损失（MSE + SSIM）
            loss_rm = mse_loss(rec_img, imgs)
            loss_rs = ssim_loss(rec_img, imgs)
            loss_rec = loss_rm + loss_rs
            
            # 总损失（λ1=λ2=λ3=1）
            loss = loss_basic + loss_rm + loss_rs
            
            # 反向传播与优化
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)
        
        avg_loss = total_loss / len(train_loader.dataset)
        scheduler.step()
        print(f"Epoch {epoch+1}, Avg Loss: {avg_loss:.4f}")
        
        # 验证（每20轮验证一次）
        if (epoch+1) % 20 == 0 or epoch == EPOCHS-1:
            i_auroc, p_auroc, pro = validate_rgpkd(model, test_loader)
            print(f"Validation - I-AUROC: {i_auroc:.4f}, P-AUROC: {p_auroc:.4f}, PRO: {pro:.4f}")
            
            # 保存最优模型
            if i_auroc > best_i_auroc:
                best_i_auroc = i_auroc
                torch.save(model.state_dict(), f"rgpkd_{category}_best.pth")
                print(f"Best model saved (I-AUROC: {best_i_auroc:.4f})")
    
    return model

def validate_rgpkd(model, test_loader):
    """验证RGPKD模型"""
    model.eval()
    pred_scores = []
    pred_masks = []
    gt_labels = []
    gt_masks = []
    
    with torch.no_grad():
        for imgs, labels, img_labels, img_paths in tqdm(test_loader, desc="Validating RGPKD"):
            imgs = imgs.to(DEVICE)
            gt_labels.extend(img_labels.numpy())
            gt_masks.extend([label.to(DEVICE) if label is not None else None for label in labels])
            
            # 前向传播
            teacher_feats, student_feats, _ = model(imgs, train_mode=False)
            
            # 生成异常评分图
            anomaly_maps = []
            for t_feat, s_feat in zip(teacher_feats, student_feats):
                # L2范数归一化
                t_feat_norm = F.normalize(t_feat, p=2, dim=1)
                s_feat_norm = F.normalize(s_feat, p=2, dim=1)
                
                # 计算特征差异
                diff = torch.norm(t_feat_norm - s_feat_norm, dim=1, keepdim=True)
                
                # 上采样到256x256
                diff_up = F.interpolate(diff, size=(IMAGE_SIZE, IMAGE_SIZE), mode='bilinear', align_corners=False)
                anomaly_maps.append(diff_up)
            
            # 多尺度融合
            anomaly_map = torch.sum(torch.stack(anomaly_maps, dim=0), dim=0).squeeze(1)
            
            # 高斯平滑
            for i in range(anomaly_map.shape[0]):
                anomaly_map[i] = torch.tensor(gaussian_filter(anomaly_map[i].cpu().numpy(), sigma=2)).to(DEVICE)
            
            # 归一化
            anomaly_map = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-8)
            
            pred_scores.extend(anomaly_map.max(dim=1)[0].max(dim=1)[0].cpu().numpy())
            pred_masks.extend(anomaly_map)
            
            # 可视化部分结果（每类随机选1张）
            if len(pred_masks) <= 1:
                visualize_result(imgs[0], gt_masks[0], anomaly_map[0], 
                               save_path=f"rgpkd_{img_paths[0].split('/')[-2]}_{img_paths[0].split('/')[-1]}")
    
    # 计算指标
    i_auroc, p_auroc, pro = calculate_metrics(pred_scores, gt_labels, pred_masks, gt_masks)
    return i_auroc, p_auroc, pro

if __name__ == "__main__":
    # 示例用法
    root_dir = "path/to/mvtec_ad"  # 替换为实际路径
    category = "bottle"  # 选择类别
    
    print("开始训练RGPKD模型...")
    model = train_rgpkd(root_dir, category)
    print("RGPKD训练完成！")
