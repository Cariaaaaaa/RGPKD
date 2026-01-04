import torch
import torch.nn as nn  # 添加这行代码
from dataset import get_data_transforms, load_data
from torchvision.datasets import ImageFolder
import numpy as np
from torch.utils.data import DataLoader
from resnet import resnet18, resnet34, resnet50, wide_resnet50_2
from de_resnet import de_resnet18, de_resnet50, de_wide_resnet50_2
from dataset import MVTecDataset, MVTecDataset_test
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from skimage import measure
import pandas as pd
from numpy import ndarray
from statistics import mean
from scipy.ndimage import gaussian_filter
from sklearn import manifold
from matplotlib.ticker import NullFormatter
from scipy.spatial.distance import pdist
import matplotlib
import pickle
import csv


# 定义辅助分类器
class AuxiliaryClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(AuxiliaryClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


# def cal_anomaly_map(fs_list, ft_list, out_size=224, amap_mode='mul'):
#
#     min_len = min(len(fs_list), len(ft_list))
#     fs_list = fs_list[:min_len]
#     ft_list = ft_list[:min_len]
#
#     if amap_mode == 'mul':
#         anomaly_map = np.ones([out_size, out_size])
#     else:
#         anomaly_map = np.zeros([out_size, out_size])
#
#     a_map_list = []
#     for i in range(len(ft_list)):
#         fs = fs_list[i]
#         ft = ft_list[i]
#
#     #     a_map = 1 - F.cosine_similarity(fs, ft)
#     #     a_map = torch.unsqueeze(a_map, dim=1)
#     #     a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True)
#     #     a_map = a_map[0, 0, :, :].to('cpu').detach().numpy()
#     #     a_map_list.append(a_map)
#     #     if amap_mode == 'mul':
#     #         anomaly_map *= a_map
#     #     else:
#     #         anomaly_map += a_map
#     #
#     # return anomaly_map, a_map_list
#     # 使用多尺度特征差异
#     diff = (fs - ft) ** 2  # 计算平方差
#     diff = torch.mean(diff, dim=1, keepdim=True)  # 沿通道维度平均
#
#     # 多尺度融合
#     a_map = F.interpolate(diff, size=out_size, mode='bilinear', align_corners=True)
#     a_map = a_map[0, 0, :, :].cpu().numpy()
#
#     # 归一化
#     a_map = (a_map - a_map.min()) / (a_map.max() - a_map.min() + 1e-8)
#
#     if amap_mode == 'mul':
#         anomaly_map *= a_map
#     else:
#             anomaly_map += a_map
#
#     # 后处理
#     if amap_mode == 'mul':
#         anomaly_map = np.power(anomaly_map, 1 / min_len)  # 几何平均
#     else:
#         anomaly_map /= min_len  # 算术平均
#
#     return anomaly_map, []
def cal_anomaly_map(fs_list, ft_list, out_size=224, amap_mode='mul'):
    anomaly_map = np.zeros([out_size, out_size])

    for fs, ft in zip(fs_list, ft_list):
        # 多尺度特征差异
        diff = torch.abs(fs - ft)

        # 通道注意力权重
        channel_weights = torch.mean(diff, dim=[2, 3], keepdim=True)
        weighted_diff = diff * channel_weights

        # 空间注意力
        spatial_weights = torch.mean(weighted_diff, dim=1, keepdim=True)

        # 插值到输出尺寸
        a_map = F.interpolate(spatial_weights, size=out_size, mode='bilinear', align_corners=True)
        a_map = a_map.squeeze().cpu().numpy()

        # 归一化
        a_map = (a_map - a_map.min()) / (a_map.max() - a_map.min() + 1e-8)

        # 融合方式
        if amap_mode == 'mul':
            anomaly_map = np.maximum(anomaly_map, a_map)  # 取最大值而不是相乘
        else:
            anomaly_map += a_map

    if amap_mode != 'mul':
        anomaly_map /= len(fs_list)

    return anomaly_map, []

def show_cam_on_image(img, anomaly_map):
    cam = np.float32(anomaly_map) / 255 + np.float32(img) / 255
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


def min_max_norm(image):
    a_min = np.nanmin(image)
    a_max = np.nanmax(image)
    if a_max == a_min:
        return np.zeros_like(image, dtype=np.float32)
    normalized = (image - a_min) / (a_max - a_min)
    normalized = np.clip(normalized, 0, 1)
    return normalized


def cvt2heatmap(gray):
    heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
    return heatmap


def evaluation(encoder, bn, decoder, decoder2, dataloader, device):
    print('Evaluating...')
    # if len(dataloader) == 0:
    #     raise ValueError("测试数据集中没有样本，请检查数据集路径和内容")
    bn.eval()
    decoder.eval()
    decoder2.eval()
    # if auxiliary_classifier:
    #     auxiliary_classifier.eval()

    gt_list_px = []
    pr_list_px = []
    gt_list_sp = []
    pr_list_sp = []
    aupro_list = []
    correct_test = 0
    total_test = 0

    with torch.no_grad():
        for img, gt, labels, types in dataloader:
            # # img classifer result
            # print('This img is a '+ str(types[0][0]) + ' ' +str(types[1][0])) # return tuble



            img = img.to(device)
            gt = gt.to(device)
            labels[0] = labels[0].to(device)
            labels[1] = labels[1].to(device)

            inputs = encoder(img)
            apha = 0.9
            beta = 1-apha
            outputs = [output * apha for output in decoder(bn(inputs))]
            outputs2 = [output2 * beta for output2 in decoder2(img)]
            output_sum = outputs2 +outputs
            # output_sum_e =[tensor1 /2.0 for tensor1 in output_sum]
            # print(f"Inputs 类型: {type(inputs)}, 长度: {len(inputs) if isinstance(inputs, list) else 'N/A'}")
            # print(
            #     f"Output_sum 类型: {type(output_sum)}, 长度: {len(output_sum) if isinstance(output_sum, list) else 'N/A'}")
            # for i, inp in enumerate(inputs):
            #     print(f"Input {i} 形状: {inp.shape}")
            # for i, out in enumerate(output_sum):
            #     print(f"Output {i} 形状: {out.shape}")

            anomaly_map, _ = cal_anomaly_map(inputs, output_sum, img.shape[-1], amap_mode='a')
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)
            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0

            if labels[0].item() != 0:
                aupro_list.append(compute_pro(gt.squeeze(0).cpu().numpy().astype(int),
                                              anomaly_map[np.newaxis, :, :]))

            gt_list_px.extend(gt.cpu().numpy().astype(int).ravel())
            pr_list_px.extend(anomaly_map.ravel())
            gt_list_sp.append(np.max(gt.cpu().numpy().astype(int)))
            pr_list_sp.append(np.max(anomaly_map))

            # 计算辅助分类器的分类准确率
            # if auxiliary_classifier:
            #     auxiliary_output = auxiliary_classifier(outputs)
            #     value_p, predicted = torch.max(auxiliary_output.data, 1)
            #     total_test += labels[1].size(0)
            #     correct_test += (predicted == labels[1]).sum().item()  # count the right classification number

        auroc_px = round(roc_auc_score(gt_list_px, pr_list_px), 3)
        auroc_sp = round(roc_auc_score(gt_list_sp, pr_list_sp), 3)
        aupro_px = round(np.mean(aupro_list), 3)

        # 计算辅助分类器的分类准确率
        # if auxiliary_classifier:
        #     test_accuracy = 100 * correct_test / total_test
        #     print(f'Auxiliary Classifier Test Accuracy: {test_accuracy:.4f}%')

    # return auroc_px, auroc_sp, aupro_px, test_accuracy
    return auroc_px, auroc_sp, aupro_px


def evaluation_test(encoder, bn, decoder, dataloader, device, _class_=None):
    bn.eval()
    decoder.eval()
    # if auxiliary_classifier:
    #     auxiliary_classifier.eval()

    gt_list_px = []
    pr_list_px = []
    gt_list_sp = []
    pr_list_sp = []
    aupro_list = []
    correct_test = 0
    total_test = 0

    with torch.no_grad():
        for img, gt, label, _ in dataloader:
            img = img.to(device)
            inputs = encoder(img)
            outputs = decoder(bn(inputs))
            anomaly_map, _ = cal_anomaly_map(inputs, outputs, img.shape[-1], amap_mode='a')
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)
            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0

            if label.item() != 0:
                aupro_list.append(compute_pro(gt.squeeze(0).cpu().numpy().astype(int),
                                              anomaly_map[np.newaxis, :, :]))

            gt_list_px.extend(gt.cpu().numpy().astype(int).ravel())
            pr_list_px.extend(anomaly_map.ravel())
            gt_list_sp.append(np.max(gt.cpu().numpy().astype(int)))
            pr_list_sp.append(np.max(anomaly_map))

            # 计算辅助分类器的分类准确率
            # if auxiliary_classifier:
            #     auxiliary_output = auxiliary_classifier(inputs[-1].view(inputs[-1].shape[0], -1))
            #     _, predicted = torch.max(auxiliary_output.data, 1)
            #     total_test += label.size(0)
            #     correct_test += (predicted == label).sum().item()

        auroc_px = round(roc_auc_score(gt_list_px, pr_list_px), 3)
        auroc_sp = round(roc_auc_score(gt_list_sp, pr_list_sp), 3)
        aupro_px = round(np.mean(aupro_list), 3)

    return auroc_px, auroc_sp, aupro_px


def test(test_class):
    # test_class = 'bottle'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    print(test_class)
    data_transform, gt_transform = get_data_transforms(256, 256)
    dataset_path_root = './dataset/test/' + test_class + '/'
    ckp_path = './checkpoints/' + 'Best' + '.pth'
    # test_data = MVTecDataset(root=test_path, transform=data_transform, gt_transform=gt_transform, phase="test")
    # 修改测试数据加载部分
    test_data = MVTecDataset_test(
        root=dataset_path_root,  # 确保路径正确
        transform=data_transform,
        gt_transform=gt_transform,
        phase="test"
    )
    print(f"测试样本数量: {len(test_data)}")  # 检查数据量
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

    encoder, bn = wide_resnet50_2(pretrained=True)
    encoder = encoder.to(device)
    bn = bn.to(device)
    encoder.eval()
    decoder = de_wide_resnet50_2(pretrained=False)
    decoder = decoder.to(device)

    # 加载辅助分类器
    # num_classes = len(test_data.classes)
    # train_dataloader, test_dataloader = load_data(dataset_name='mvtec')
    # num_classes = len(test_dataloader.dataset.classes)  # 获取类别数量
    # print("Number of classes:", num_classes)
    # auxiliary_classifier = AuxiliaryClassifier(input_dim=2048, num_classes=num_classes).to(device)
    ckp = torch.load(ckp_path)
    decoder.load_state_dict(ckp['decoder'])
    bn.load_state_dict(ckp['bn'])
    # if 'auxiliary_classifier' in ckp:
    #     auxiliary_classifier.load_state_dict(ckp['auxiliary_classifier'])

    auroc_px, auroc_sp, aupro_px = evaluation_test(encoder, bn, decoder, test_dataloader, device, test_class, )
    print("Class: {}, Pixel Auroc: {:.4f}, Sample Auroc： {:.4f}, Pixel Aupro： {:.4f}".format(
        test_class, auroc_px, auroc_sp, aupro_px))
    with open('./save_csv/Result_test.csv', 'a', newline='') as f1:
        writer = csv.writer(f1)
        row = [
            "Class: {}, Pixel Auroc: {:.4f}, Sample Auroc： {:.4f}, Pixel Aupro： {:.4f}".format(
                test_class, auroc_px, auroc_sp, aupro_px)]
        writer.writerow(row)
    return auroc_px


import os


def visualization(_class_):
    print(_class_)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    image_size = 256
    data_transform, gt_transform = get_data_transforms(image_size, image_size)
    dataset_path_root = './dataset/test/' + _class_ + '/'
    ckp_path = './checkpoints/' + 'Best' + '.pth'
    test_data = MVTecDataset_test(root=dataset_path_root, transform=data_transform, gt_transform=gt_transform,
                                  phase="test")
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

    encoder, bn = wide_resnet50_2(pretrained=True)
    encoder = encoder.to(device)
    bn = bn.to(device)

    encoder.eval()
    decoder = de_wide_resnet50_2(pretrained=False)
    decoder = decoder.to(device)
    ckp = torch.load(ckp_path)
    # for k, v in list(ckp['bn'].items()):
    #     if 'memory' in k:
    #         ckp['bn'].pop(k)
    decoder.load_state_dict(ckp['decoder'])
    bn.load_state_dict(ckp['bn'])
    idx = 1
    count = 0
    with torch.no_grad():
        for img, gt, label, _ in test_dataloader:
            if (label.item() == 0):
                continue
            # if count <= 10:
            #    count += 1
            #    continue

            decoder.eval()
            bn.eval()

            img = img.to(device)
            gt = gt.to(device)
            inputs = encoder(img)
            outputs = decoder(bn(inputs))

            # inputs.append(feature)
            # inputs.append(outputs)
            # t_sne(inputs)

            anomaly_map, amap_list = cal_anomaly_map([inputs[-1]], [outputs[-1]], img.shape[-1], amap_mode='a')
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)

            ano_map = min_max_norm(anomaly_map)
            ano_map = cvt2heatmap(ano_map * 255)

            anomaly_map[anomaly_map > 0.24] = 1  ##ROI阈值
            anomaly_map[anomaly_map <= 0.24] = 0
            anomaly_map = torch.Tensor(anomaly_map)
            anomaly_map = torch.unsqueeze(anomaly_map, 0)
            anomaly_map = torch.unsqueeze(anomaly_map, 0)
            anomaly_map = anomaly_map.to(device)
            img = cv2.cvtColor(img.permute(0, 2, 3, 1).cpu().numpy()[0] * 255, cv2.COLOR_BGR2RGB)
            img = np.uint8(min_max_norm(img) * 255)
            gt1 = gt.permute(0, 2, 3, 1).cpu()
            gt = cv2.cvtColor(gt1.numpy()[0] * 255, cv2.COLOR_BGR2RGB)
            gt = np.uint8(min_max_norm(gt) * 255)
            anomaly_map1 = anomaly_map.permute(0, 2, 3, 1).cpu()
            anomaly_map = cv2.cvtColor(anomaly_map1.numpy()[0] * 255, cv2.COLOR_BGR2RGB)
            anomaly_map = np.uint8(min_max_norm(anomaly_map) * 255)
            # if not os.path.exists('./results_all/'+_class_):
            #    os.makedirs('./results_all/'+_class_)
            # cv2.imwrite('./results_all/'+_class_+'/'+str(count)+'_'+'org.png',img)
            # plt.imshow(img)
            # plt.axis('off')
            # plt.savefig('org.png')
            # plt.show()
            ano_map = show_cam_on_image(img, ano_map)

            # cv2.imwrite('./results_all/'+_class_+'/'+str(count)+'_'+'ad.png', ano_map)
            save_file_name = 'Visualization_result'
            save_img_path = './' + save_file_name
            if not os.path.exists(save_img_path):
                os.makedirs(save_img_path)
            plt.subplot(2, 2, 1)
            plt.imshow(ano_map)
            plt.title('ano_map', y=-0.3)
            plt.subplot(2, 2, 2)
            # plt.axis('off')
            # plt.savefig(f'./save3_AANEWNEW/ad{idx}.png')
            plt.imshow(img)
            # plt.axis('off')
            # plt.savefig(f'./save3_AANEWNEW/ad{idx}or.png')
            plt.title('or_img', y=-0.3)
            plt.subplot(2, 2, 4)
            plt.imshow(gt)
            plt.title('ground_truth', y=-0.3)
            plt.subplot(2, 2, 3)
            plt.imshow(anomaly_map)
            plt.title('ROI', y=-0.3)
            plt.savefig('./Visualization_result/' + _class_ + f'{idx}.png')
            print(_class_ + str(idx) + '  saved')
            idx += 1
            # plt.show()

            # gt = gt.cpu().numpy().astype(int)[0][0]*255
            # cv2.imwrite('./results/'+_class_+'_'+str(count)+'_'+'gt.png', gt)

            # b, c, h, w = inputs[2].shape
            # t_feat = F.normalize(inputs[2], p=2).view(c, -1).permute(1, 0).cpu().numpy()
            # s_feat = F.normalize(outputs[2], p=2).view(c, -1).permute(1, 0).cpu().numpy()
            # c = 1-min_max_norm(cv2.resize(anomaly_map,(h,w))).flatten()
            # print(c.shape)
            # t_sne([t_feat, s_feat], c)
            # assert 1 == 2

            # name = 0
            # for anomaly_map in amap_list:
            #    anomaly_map = gaussian_filter(anomaly_map, sigma=4)
            #    ano_map = min_max_norm(anomaly_map)
            #    ano_map = cvt2heatmap(ano_map * 255)
            # ano_map = show_cam_on_image(img, ano_map)
            # cv2.imwrite(str(name) + '.png', ano_map)
            # plt.imshow(ano_map)
            # plt.axis('off')
            # plt.savefig(str(name) + '.png')
            # plt.show()
            #    name+=1
            count += 1
            # if count>20:
            #    return 0
            # assert 1==2


def vis_nd(name, _class_):
    print(name, ':', _class_)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    ckp_path = './checkpoints/' + name + '_' + str(_class_) + '.pth'
    train_dataloader, test_dataloader = load_data(name, _class_, batch_size=16)

    encoder, bn = resnet18(pretrained=True)
    encoder = encoder.to(device)
    bn = bn.to(device)
    encoder.eval()
    decoder = de_resnet18(pretrained=False)
    decoder = decoder.to(device)

    ckp = torch.load(ckp_path)

    decoder.load_state_dict(ckp['decoder'])
    bn.load_state_dict(ckp['bn'])
    decoder.eval()
    bn.eval()

    gt_list_sp = []
    prmax_list_sp = []
    prmean_list_sp = []

    count = 0
    with torch.no_grad():
        for img, label in test_dataloader:
            if img.shape[1] == 1:
                img = img.repeat(1, 3, 1, 1)
            # if count <= 10:
            #    count += 1
            #    continue
            img = img.to(device)
            inputs = encoder(img)
            # print(inputs[-1].shape)
            outputs = decoder(bn(inputs))

            anomaly_map, amap_list = cal_anomaly_map(inputs, outputs, img.shape[-1], amap_mode='a')
            # anomaly_map = gaussian_filter(anomaly_map, sigma=4)
            ano_map = min_max_norm(anomaly_map)
            ano_map = cvt2heatmap(ano_map * 255)
            img = cv2.cvtColor(img.permute(0, 2, 3, 1).cpu().numpy()[0] * 255, cv2.COLOR_BGR2RGB)
            img = np.uint8(min_max_norm(img) * 255)
            cv2.imwrite('./nd_results/' + name + '_' + str(_class_) + '_' + str(count) + '_' + 'org.png', img)
            # plt.imshow(img)
            # plt.axis('off')
            # plt.savefig('org.png')
            # plt.show()
            ano_map = show_cam_on_image(img, ano_map)
            cv2.imwrite('./nd_results/' + name + '_' + str(_class_) + '_' + str(count) + '_' + 'ad.png', ano_map)
            # plt.imshow(ano_map)
            # plt.axis('off')
            # plt.savefig('ad.png')
            # plt.show()

            # gt = gt.cpu().numpy().astype(int)[0][0]*255
            # cv2.imwrite('./results/'+_class_+'_'+str(count)+'_'+'gt.png', gt)

            # b, c, h, w = inputs[2].shape
            # t_feat = F.normalize(inputs[2], p=2).view(c, -1).permute(1, 0).cpu().numpy()
            # s_feat = F.normalize(outputs[2], p=2).view(c, -1).permute(1, 0).cpu().numpy()
            # c = 1-min_max_norm(cv2.resize(anomaly_map,(h,w))).flatten()
            # print(c.shape)
            # t_sne([t_feat, s_feat], c)
            # assert 1 == 2

            # name = 0
            # for anomaly_map in amap_list:
            #    anomaly_map = gaussian_filter(anomaly_map, sigma=4)
            #    ano_map = min_max_norm(anomaly_map)
            #    ano_map = cvt2heatmap(ano_map * 255)
            # ano_map = show_cam_on_image(img, ano_map)
            # cv2.imwrite(str(name) + '.png', ano_map)
            # plt.imshow(ano_map)
            # plt.axis('off')
            # plt.savefig(str(name) + '.png')
            # plt.show()
            #    name+=1
            # count += 1
            # if count>40:
            #    return 0
            # assert 1==2
            gt_list_sp.extend(label.cpu().data.numpy())
            prmax_list_sp.append(np.max(anomaly_map))
            prmean_list_sp.append(np.sum(anomaly_map))  # np.sum(anomaly_map.ravel().argsort()[-1:][::-1]))

        gt_list_sp = np.array(gt_list_sp)
        indx1 = gt_list_sp == _class_
        indx2 = gt_list_sp != _class_
        gt_list_sp[indx1] = 0
        gt_list_sp[indx2] = 1

        ano_score = (prmean_list_sp - np.min(prmean_list_sp)) / (np.max(prmean_list_sp) - np.min(prmean_list_sp))
        vis_data = {}
        vis_data['Anomaly Score'] = ano_score
        vis_data['Ground Truth'] = np.array(gt_list_sp)
        # print(type(vis_data))
        # np.save('vis.npy',vis_data)
        with open('vis.pkl', 'wb') as f:
            pickle.dump(vis_data, f, pickle.HIGHEST_PROTOCOL)


def compute_pro(masks: ndarray, amaps: ndarray, num_th: int = 200) -> None:
    """Compute the area under the curve of per-region overlaping (PRO) and 0 to 0.3 FPR
    Args:
        category (str): Category of product
        masks (ndarray): All binary masks in test. masks.shape -> (num_test_data, h, w)
        amaps (ndarray): All anomaly maps in test. amaps.shape -> (num_test_data, h, w)
        num_th (int, optional): Number of thresholds
    """

    assert isinstance(amaps, ndarray), "type(amaps) must be ndarray"
    assert isinstance(masks, ndarray), "type(masks) must be ndarray"
    assert amaps.ndim == 3, "amaps.ndim must be 3 (num_test_data, h, w)"
    assert masks.ndim == 3, "masks.ndim must be 3 (num_test_data, h, w)"
    assert amaps.shape == masks.shape, "amaps.shape and masks.shape must be same"
    assert set(masks.flatten()) == {0, 1}, "set(masks.flatten()) must be {0, 1}"
    assert isinstance(num_th, int), "type(num_th) must be int"

    df = pd.DataFrame([], columns=["pro", "fpr", "threshold"])
    binary_amaps = np.zeros_like(amaps, dtype=np.bool_)

    min_th = amaps.min()
    max_th = amaps.max()
    delta = (max_th - min_th) / num_th

    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th] = 0  # 从预测的map的最小值 一直到最大值
        binary_amaps[amaps > th] = 1

        pros = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                axes0_ids = region.coords[:, 0]  ## mask 的真值区域
                axes1_ids = region.coords[:, 1]
                tp_pixels = binary_amap[axes0_ids, axes1_ids].sum()
                pros.append(tp_pixels / region.area)  # 为 binarymap 与 mask交集所占mask区域的比例

        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()

        df = df._append({"pro": mean(pros), "fpr": fpr, "threshold": th}, ignore_index=True)

    # Normalize FPR from 0 ~ 1 to 0 ~ 0.3
    df = df[df["fpr"] < 0.3]
    df["fpr"] = df["fpr"] / df["fpr"].max()

    pro_auc = auc(df["fpr"], df["pro"])
    return pro_auc


def detection(encoder, bn, decoder, dataloader, device, _class_):
    # _, t_bn = resnet50(pretrained=True)
    bn.load_state_dict(bn.state_dict())
    bn.eval()
    # t_bn.to(device)
    # t_bn.load_state_dict(bn.state_dict())
    decoder.eval()
    gt_list_sp = []
    prmax_list_sp = []
    prmean_list_sp = []
    with torch.no_grad():
        for img, label in dataloader:

            img = img.to(device)
            if img.shape[1] == 1:
                img = img.repeat(1, 3, 1, 1)
            label = label.to(device)
            inputs = encoder(img)
            outputs = decoder(bn(inputs))
            anomaly_map, _ = cal_anomaly_map(inputs, outputs, img.shape[-1], 'acc')
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)

            gt_list_sp.extend(label.cpu().data.numpy())
            prmax_list_sp.append(np.max(anomaly_map))
            prmean_list_sp.append(np.sum(anomaly_map))  # np.sum(anomaly_map.ravel().argsort()[-1:][::-1]))

        gt_list_sp = np.array(gt_list_sp)
        indx1 = gt_list_sp == _class_
        indx2 = gt_list_sp != _class_
        gt_list_sp[indx1] = 0
        gt_list_sp[indx2] = 1

        auroc_sp_max = round(roc_auc_score(gt_list_sp, prmax_list_sp), 4)
        auroc_sp_mean = round(roc_auc_score(gt_list_sp, prmean_list_sp), 4)
    return auroc_sp_max, auroc_sp_mean


if __name__ == '__main__':
    item_list = ['carpet', 'bottle', 'hazelnut', 'leather', 'cable', 'capsule', 'grid', 'pill',
                 'transistor', 'metal_nut', 'screw', 'toothbrush', 'zipper', 'tile', 'wood']
    for i in item_list:
        test(i)
