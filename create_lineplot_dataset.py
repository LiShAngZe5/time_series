import copy
import os
import numpy as np
import pandas as pd
import glob
import re
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import sys
import argparse
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import stats
import json

from utils.timefeatures import time_features
from data_provider.m4 import M4Dataset, M4Meta
from data_provider.uea import (
    subsample,
    interpolate_missing,
    Normalizer,
)
from sktime.datasets import load_from_tsfile_to_dataframe
import warnings
import random
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from natsort import natsorted
from models import TimesNet
warnings.filterwarnings("ignore")


args = argparse.Namespace()
args.model_id = "Cricket"         #更换数据集的名称


class CANLoader(Dataset):
    def __init__(self, args, root_path, flag=None):
        self.root_path = root_path
        self.name = args.model_id
        self.data_path = os.path.join(root_path, f'{self.name}.npy')
        self.X, self.y = self.load_can_data(self.data_path, flag)
        self.max_seq_len = self.X.shape[1]

    def load_can_data(self, data_path, flag):

        Data = np.load(data_path, allow_pickle=True).item()

        if flag == 'train':
            return torch.tensor(Data['All_train_data'], dtype=torch.float32).permute(0, 2, 1), \
                torch.tensor(Data['All_train_label'], dtype=torch.int64)
        elif flag == 'val':
            return torch.tensor(Data['test_data'], dtype=torch.float32).permute(0, 2, 1), \
                torch.tensor(Data['test_label'], dtype=torch.int64)
        elif flag == 'test':
            return torch.tensor(Data['test_data'], dtype=torch.float32).permute(0, 2, 1), \
                torch.tensor(Data['test_label'], dtype=torch.int64)
        else:
            raise ValueError('flag must be TRAIN, VAL, or TEST')

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return len(self.y)
    
class Config:
    def __init__(self):
        self.seq_len =1197     #序列长度
        self.pred_len = 0      
        self.top_k = 2         # 选择top-k个周期
        self.d_model =6        #特征数量
        self.d_ff = 256         
        self.num_kernels = 6
        self.grid = (3, 3)    

configs = Config()
times_block = TimesNet.TimesBlock(configs)

feature_representations = []



def calculate_brightness(color):

    r, g, b = color

    brightness = 0.299 * r + 0.587 * g + 0.114 * b
    return brightness


def generate_distinct_colors(num_colors, seed=42, max_brightness=0.85, min_brightness=0.15):

    import colorsys
    
    feature_colors = [] 
    # 在色相上均匀分布，从0到1（不包括1，避免首尾颜色相同）
    for i in range(num_colors):
        hue = i / num_colors  
        saturation = 0.85     
        value = 0.85         
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
        color = (r, g, b)
        brightness = calculate_brightness(color)
        
        if brightness > max_brightness or brightness < min_brightness:
            if brightness > max_brightness:
                value = 0.65  
            else:
                value = 0.95  
            
            r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
            color = (r, g, b)
        feature_colors.append(color)
    
    return feature_colors
    
def create_dataset_with_permutations(args, output_base_dir="./lineplot_dataset", 
                                     num_permutations=5):

    sample_data = CANLoader(args, r"/data/iamlisz/time/UEA", flag='train')
    num_features = sample_data.X.shape[2]
    
    feature_colors = generate_distinct_colors(
        num_colors=num_features, 
        seed=42,
        max_brightness=0.85,  # 避免过亮（接近白色）
        min_brightness=0.15   # 避免过暗（不易看清）
    )
    fixed_color_mapping = {i: feature_colors[i] for i in range(num_features)}
    
    print("   特征颜色映射:")
    for feat_idx, color in fixed_color_mapping.items():
        brightness = calculate_brightness(color)
        print(f"     特征{feat_idx}: RGB{color} (亮度: {brightness:.3f})")
    
    # 保存全局颜色映射
    color_mapping_path = os.path.join(output_base_dir, "global_color_mapping.json")
    os.makedirs(output_base_dir, exist_ok=True)
    with open(color_mapping_path, 'w') as f:
        json_ready_mapping = {str(k): list(v) for k, v in fixed_color_mapping.items()}
        json.dump(json_ready_mapping, f, indent=2)
    print(f"   颜色映射已保存: {color_mapping_path}")
    
    # 生成排列
    selected_permutations = []
    
    # 第一个排列：原始顺序
    selected_permutations.append(list(range(num_features)))
    
    # 生成剩余的随机排列
    np.random.seed(42)
    for i in range(num_permutations - 1):
        perm = list(range(num_features))
        np.random.shuffle(perm)
        selected_permutations.append(perm)
    
    all_metadata = {}
    
    for split_name in ['train', 'val', 'test']:
        time_series_data = CANLoader(args, r"/data/iamlisz/time/UEA", flag=split_name)
        data = time_series_data.X
        labels = time_series_data.y.detach().cpu().numpy()
        num_classes = len(np.unique(labels))
        print(f"\n{split_name} 集类别数: {num_classes}")
        split_metadata = []
        for perm_idx, permutation in enumerate(selected_permutations):
            perm_output_dir = os.path.join(output_base_dir, split_name, f"perm_{perm_idx}")
            print(f"\n{split_name} 集，排列 {perm_idx}: {permutation}")
            
            metadata = plot_data(
                data, 
                labels, 
                save_dir=perm_output_dir,
                outlier_method=None,
                interpolation=True,
                grid_layout=configs.grid,
                feature_permutation=permutation,
                color_mapping=fixed_color_mapping
            )
            
            split_metadata.extend(metadata)
        
        all_metadata[split_name] = split_metadata
    
    # 保存数据集信息
    dataset_info = {
        'dataset_name': '_',
        'num_permutations': len(selected_permutations),
        'permutations': selected_permutations,
        'color_mapping_file': 'global_color_mapping.json',
        'total_samples': sum(len(meta) for meta in all_metadata.values()),
        'original_samples_per_split': {
            split: len(meta) // len(selected_permutations) 
            for split, meta in all_metadata.items()
        }
    }
    
    with open(os.path.join(output_base_dir, 'dataset_info.json'), 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    return all_metadata, dataset_info

def plot_data(tensor_data, labels, save_dir="./timeseries_images", 
              outlier_method=None, interpolation=True, 
              color_mapping=None, grid_layout=configs.grid,
              linestyle='-', linewidth=1.0, marker='*', markersize=1.5, feature_permutation=None):

    if isinstance(tensor_data, torch.Tensor):
        data = tensor_data.detach().cpu().numpy()
    else:
        data = np.array(tensor_data)
    
    num_samples, num_timesteps, num_features = data.shape
    print(f"数据维度: {data.shape}")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 特征颜色映射
    if color_mapping is None:
        feature_colors = generate_distinct_colors(
            num_colors=num_features,
            seed=42,
            max_brightness=0.85,
            min_brightness=0.15
        )
        color_mapping = {i: feature_colors[i] for i in range(num_features)}
    
    # 计算每个特征的数值范围（考虑异常值）
    feature_ranges = []
    
    for feat_idx in range(num_features):
        feat_data = data[:, :, feat_idx].flatten()
        valid_data = feat_data[~np.isnan(feat_data) & (feat_data != 0)]
        
        if len(valid_data) == 0:
            feature_ranges.append([0, 1])
            continue
            
        if outlier_method == "iqr":
            q1 = np.percentile(valid_data, 25)
            q3 = np.percentile(valid_data, 75)
            iqr = q3 - q1
            lower_bound = q1 - (1.5 * iqr)
            upper_bound = q3 + (1.5 * iqr)
        elif outlier_method == "sd":
            mean_val = np.mean(valid_data)
            std_val = np.std(valid_data)
            lower_bound = mean_val - (3 * std_val)
            upper_bound = mean_val + (3 * std_val)
        elif outlier_method == "mzs":
            med = np.median(valid_data)
            deviation_from_med = valid_data - med
            mad = np.median(np.abs(deviation_from_med))
            lower_bound = (-3.5/0.6745) * mad + med
            upper_bound = (3.5/0.6745) * mad + med
        else:
            lower_bound = np.min(valid_data)
            upper_bound = np.max(valid_data)
        
        padding = (upper_bound - lower_bound) * 0.05
        feature_ranges.append([lower_bound - padding, upper_bound + padding])

    metadata_list = []
    grid_height, grid_width = grid_layout
    
    for sample_idx in tqdm(range(num_samples), desc="生成图像"):
        plt.figure(figsize=(4.48, 4.48), dpi=100)  
        sample_data = data[sample_idx]  # (时间步长, 特征数)
        
        # 网格绘制每个特征
        for subplot_idx, original_feat_idx in enumerate(feature_permutation):
            plt.subplot(grid_height, grid_width, subplot_idx + 1)
            
            # 时间序列数据
            time_steps = np.arange(num_timesteps)
            values = sample_data[:, original_feat_idx]
            
            # 处理缺失值和异常值
            if interpolation:
                # 移除缺失值和异常值
                valid_mask = (~np.isnan(values)) & (values != 0)
                valid_mask = valid_mask & (values >= feature_ranges[original_feat_idx][0]) & (values <= feature_ranges[original_feat_idx][1])
                time_steps = time_steps[valid_mask]
                values = values[valid_mask]
            else:
                # 将缺失值和异常值设为NaN
                invalid_mask = (np.isnan(values)) | (values == 0) | (values < feature_ranges[original_feat_idx][0]) | (values > feature_ranges[original_feat_idx][1])
                values[invalid_mask] = np.nan
            # 绘制线图
            plt.plot(time_steps, values, 
                    color=color_mapping[original_feat_idx],
                    linestyle=linestyle,
                    linewidth=linewidth,
                    marker=marker,
                    markersize=markersize)
            # 设置坐标轴
            plt.xlim(0, num_timesteps-1)
            plt.ylim(feature_ranges[original_feat_idx])
            plt.xticks([])
            plt.yticks([])
        
        # 调整布局
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        
        img_path = os.path.join(save_dir, f"sample_{sample_idx:04d}.png")
        plt.savefig(img_path, pad_inches=0)
        plt.close()
        
        # 保存元数据
        label = labels[sample_idx] if labels is not None else 0
        metadata = {
            "id": sample_idx,
            "image_path": img_path,
            "label": int(label),
            "label_name": "no" if label == 0 else "yes",
            "param_num": num_features,
            "features": list(range(num_features)),
            "feature_permutation": feature_permutation,
            "color_mapping": color_mapping
        }
        metadata_list.append(metadata)

    # 保存元数据到文件
    metadata_path = os.path.join(save_dir, "metadata.npy")
    np.save(metadata_path, metadata_list)
    
    # 保存颜色映射到JSON文件
    color_mapping_path = os.path.join(save_dir, "color_mapping.json")
    with open(color_mapping_path, 'w') as f:
        json_ready_mapping = {k: list(v) for k, v in color_mapping.items()}
        json.dump(json_ready_mapping, f)
    
    return metadata_list




if __name__ == "__main__":
    create_dataset_with_permutations(args, output_base_dir="./lineplot_dataset_augmented_cri", 

                                    num_permutations=5)
