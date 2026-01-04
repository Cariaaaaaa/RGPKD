from torchvision import transforms
from PIL import Image
import os
import torch
import glob
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST, ImageFolder
import numpy as np

def get_data_transforms(size, isize):
    mean_train = [0.485, 0.456, 0.406]
    std_train = [0.229, 0.224, 0.225]
    data_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.CenterCrop(isize),
        transforms.Normalize(mean=mean_train, std=std_train)
    ])
    gt_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.CenterCrop(isize),
        transforms.ToTensor()
    ])
    return data_transforms, gt_transforms

class MVTecDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, gt_transform, phase):
        # class_list =
        if phase == 'train':
            self.img_path_root = os.path.join(root, 'train')  # 直接使用 root/train
        else:
            self.img_path_root = os.path.join(root, 'test')  # 直接使用 root/test
            # self.gt_path = os.path.join(root, 'ground_truth')  # 直接使用 root/ground_truth

        self.transform = transform
        self.gt_transform = gt_transform
        # 加载数据集
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset()  # self.labels => good : 0, anomaly : 1
        self.classes = self.get_classes()  # 初始化 classes 属性
    def load_dataset(self):
        img_tot_paths = [] # the path of every img
        gt_tot_paths = [] # the correspond gt path
        tot_labels = [[],[]] # the label of every img
        tot_types = [[],[]] # the abnormally type of every img
        classifier_types = os.listdir(self.img_path_root)


        for i in range(len(classifier_types)):
            test_types_path = self.img_path_root+'/'+ classifier_types[i]+'/'
            test_types = os.listdir(test_types_path+'test/')
            for test_type in test_types:
                if test_type == 'good':
                    img_paths = glob.glob(os.path.join(test_types_path+'test/', test_type + '/*.png'))
                    img_tot_paths.extend(img_paths)
                    gt_tot_paths.extend([0] * len(img_paths))
                    tot_labels[0].extend([0]*len(img_paths)) # every 'good' image code to 0
                    tot_labels[1].extend([i]*len(img_paths)) # class code to the index i
                    tot_types[0].extend([test_type] * len(img_paths)) # explain of test abnormal type
                    tot_types[1].extend([classifier_types[i]] * len(img_paths)) # explain class of img
                else:
                    img_paths = glob.glob(os.path.join(test_types_path+'test/', test_type + '/*.png'))
                    gt_paths = glob.glob(os.path.join(test_types_path+'ground_truth/', test_type + "/*.png"))
                    img_paths.sort()
                    gt_paths.sort()
                    img_tot_paths.extend(img_paths)
                    gt_tot_paths.extend(gt_paths)
                    tot_labels[0].extend([1] * len(img_paths))  # every abnormal image code to 1
                    tot_labels[1].extend([i] * len(img_paths))  # class code to the index i
                    tot_types[0].extend([test_type] * len(img_paths))  # explain of test abnormal type
                    tot_types[1].extend([classifier_types[i]] * len(img_paths))  # explain class of img

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return img_tot_paths, gt_tot_paths, tot_labels, tot_types

    def get_classes(self):
        # 获取所有类别名称
        classifier_types = os.listdir(self.img_path_root)

        return classifier_types  # 返回类别列表，例如 ['bottle', 'cable', 'capsule', ...]

    def __len__(self):
        # return len(self.img_path_root)  # 返回数据集的长度
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], [self.labels[0][idx],self.labels[1][idx]], [self.types[0][idx],self.types[1][idx]]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        if gt == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)

        assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"

        return img, gt, label, img_type


class MVTecDataset_test(torch.utils.data.Dataset):
    def __init__(self, root, transform, gt_transform, phase):
        # class_list =
        if phase == 'train':
            self.img_path = os.path.join(root, 'train')  # 直接使用 root/train
        else:
            self.img_path = os.path.join(root, 'test')  # 直接使用 root/test
            self.gt_path = os.path.join(root, 'ground_truth')  # 直接使用 root/ground_truth
        self.transform = transform
        self.gt_transform = gt_transform
        # 加载数据集
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset()  # self.labels => good : 0, anomaly : 1
        self.classes = self.get_classes()  # 初始化 classes 属性

    def load_dataset(self):
        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = os.listdir(self.img_path)

        for defect_type in defect_types:
            if defect_type == 'good':
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0] * len(img_paths))
                tot_labels.extend([0] * len(img_paths))  # good 类别标签为 0
                tot_types.extend(['good'] * len(img_paths))
            else:
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
                gt_paths = glob.glob(os.path.join(self.gt_path, defect_type) + "/*.png")
                img_paths.sort()
                gt_paths.sort()
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(img_paths))  # 缺陷类别标签为 1
                tot_types.extend([defect_type] * len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return img_tot_paths, gt_tot_paths, tot_labels, tot_types

    def get_classes(self):
        # 获取所有类别名称
        defect_types = os.listdir(self.img_path)
        return defect_types  # 返回类别列表，例如 ['good', 'defect_type_1', 'defect_type_2', ...]

    def __len__(self):
        return len(self.img_paths)  # 返回数据集的长度

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        if gt == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)

        assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"

        return img, gt, label, img_type
def load_data(dataset_name='mnist', normal_class=0, batch_size=16):
    if dataset_name == 'cifar10':
        img_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

        os.makedirs("./Dataset/CIFAR10/train", exist_ok=True)
        dataset = CIFAR10('./Dataset/CIFAR10/train', train=True, download=True, transform=img_transform)
        print("Cifar10 DataLoader Called...")
        print("All Train Data: ", dataset.data.shape)
        dataset.data = dataset.data[np.array(dataset.targets) == normal_class]
        dataset.targets = [normal_class] * dataset.data.shape[0]
        print("Normal Train Data: ", dataset.data.shape)

        os.makedirs("./Dataset/CIFAR10/test", exist_ok=True)
        test_set = CIFAR10("./Dataset/CIFAR10/test", train=False, download=True, transform=img_transform)
        print("Test Train Data:", test_set.data.shape)

    elif dataset_name == 'mnist':
        img_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor()
        ])

        os.makedirs("./Dataset/MNIST/train", exist_ok=True)
        dataset = MNIST('./Dataset/MNIST/train', train=True, download=True, transform=img_transform)
        print("MNIST DataLoader Called...")
        print("All Train Data: ", dataset.data.shape)
        dataset.data = dataset.data[np.array(dataset.targets) == normal_class]
        dataset.targets = [normal_class] * dataset.data.shape[0]
        print("Normal Train Data: ", dataset.data.shape)

        os.makedirs("./Dataset/MNIST/test", exist_ok=True)
        test_set = MNIST("./Dataset/MNIST/test", train=False, download=True, transform=img_transform)
        print("Test Train Data:", test_set.data.shape)

    elif dataset_name == 'fashionmnist':
        img_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor()
        ])

        os.makedirs("./Dataset/FashionMNIST/train", exist_ok=True)
        dataset = FashionMNIST('./Dataset/FashionMNIST/train', train=True, download=True, transform=img_transform)
        print("FashionMNIST DataLoader Called...")
        print("All Train Data: ", dataset.data.shape)
        dataset.data = dataset.data[np.array(dataset.targets) == normal_class]
        dataset.targets = [normal_class] * dataset.data.shape[0]
        print("Normal Train Data: ", dataset.data.shape)

        os.makedirs("./Dataset/FashionMNIST/test", exist_ok=True)
        test_set = FashionMNIST("./Dataset/FashionMNIST/test", train=False, download=True, transform=img_transform)
        print("Test Train Data:", test_set.data.shape)

    elif dataset_name == 'retina':
        data_path = 'Dataset/OCT2017/train'
        orig_transform = transforms.Compose([
            transforms.Resize([128, 128]),
            transforms.ToTensor()
        ])
        dataset = ImageFolder(root=data_path, transform=orig_transform)
        test_data_path = 'Dataset/OCT2017/test'
        test_set = ImageFolder(root=test_data_path, transform=orig_transform)

    elif dataset_name == 'mvtec':
        img_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        gt_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
        train_path = './mvtec/train'  # 直接使用 train_path
        test_path = './mvtec/test'  # 直接使用 test_path
        dataset = MVTecDataset(root=train_path, transform=img_transform, gt_transform=gt_transform, phase="train")
        test_set = MVTecDataset(root=test_path, transform=img_transform, gt_transform=gt_transform, phase="test")

    else:
        raise Exception(
            "You enter {} as dataset, which is not a valid dataset for this repository!".format(dataset_name))

    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
    )

    return train_dataloader, test_dataloader