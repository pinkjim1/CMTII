import random
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader





class CustomCombineImageDataset(Dataset):
    def __init__(self, images, labels, tuple_list, weights=None):

        self.images = images
        self.labels = labels
        self.tuple_list = tuple_list
        self.data = []
        for label, image in zip(labels, images):
            self.data.append((label, image))
        if tuple_list is not None:
            self.data.extend(tuple_list)
            if weights is not None:
                tem_weights=[1.0]*len(self.labels)
                tem_weights +=[weights] * len(self.tuple_list)
                self.weights = torch.tensor(tem_weights)
            else:
                self.weights = torch.ones(len(self.data))
        else:
            self.weights = torch.ones(len(self.data))



    def __len__(self):
        # 返回数据集中图片的数量
        return len(self.data)

    def __getitem__(self, idx):
        # 根据索引返回图像和标签
        return self.data[idx], self.weights[idx]


class CustomCombineDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=4, shuffle=True, preprocess=None, total_labels=None, message_type=None, device='cpu', **kwargs):
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)
        self.preprocess = preprocess
        self.device = device
        self.total_labels = [f'a photo of a {label}' for label in total_labels]
        self.message_type = message_type

    def __iter__(self):
        # 自定义迭代器
        for batch in super().__iter__():
            (labels, images_path), weights = batch
            filtered_images = []
            filtered_labels = []
            for image_path, label in zip(images_path, labels):
                image = Image.open(image_path)
                filtered_images.append(image)
                filtered_labels.append(self.message_type[label])

            pil_images = [image.resize((512, 512)) for image in filtered_images]
            return_value=self.preprocess(text=self.total_labels, images=pil_images, return_tensors="pt", padding=True)
            return_value = return_value.to(self.device)
            label_tensor = torch.tensor(filtered_labels, dtype=torch.long).to(self.device)
            weights = weights.to(self.device)
            yield return_value, label_tensor, weights