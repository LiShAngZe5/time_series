import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import warnings
import math
import random
from torchvision import transforms
warnings.filterwarnings('ignore')

class Config:
    data_dir = '/data/iamlisz/time/Cricket-train/lineplot_dataset'   #图像数据集路径
    num_classes = 12                                                 #类别数需要根据画图代码的输出修改
    model_name = "/data/iamlisz/time/Qwen2.5-VL-7B-Instruct/Qwen2.5-VL-7B-Instruct"  #模型位置

    seq_len =1197   #根据数据集填写时间步

    num_features = 6     #根据数据集填写特征数

    patch_len = 128     #patchtst参数

    stride = 64        #patchtst参数
    
 
    use_lora = True
    freeze_backbone = True
    use_quantization = False
    lora_r = 16 
    lora_alpha = 32 
    lora_dropout = 0.2 
    lora_target_modules = [
        "q_proj", "v_proj" ,"k_proj","o_proj"
    ]
    batch_size = 4
    gradient_accumulation_steps = 4
    learning_rate = 2e-5  
    classifier_lr = 1e-4  
    temporal_lr = 4e-5        
    fusion_lr = 4e-5        
    num_epochs = 80
    warmup_ratio = 0.15  
    weight_decay = 0.04
    max_grad_norm = 1.0  
    
    image_size = 448
    max_pixels = 448 * 448
    min_pixels = 224 * 224
    

    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    fp16 = True
    gradient_checkpointing = True
    
    output_dir = './qwen25_vl_checkpoint'
    save_steps = 500
    logging_steps = 50
    eval_steps = 50
    
    # 早停
    early_stop_patience = 7  
    min_improvement = 0.001
    
    @classmethod
    def print_config(cls):
        print("配置 ")
        print(f"LoRA: r={cls.lora_r}, alpha={cls.lora_alpha}, dropout={cls.lora_dropout}")
        print(f"LoRA模块: {', '.join(cls.lora_target_modules)}")
        print(f"学习率: base={cls.learning_rate}, classifier={cls.classifier_lr}")
        print(f"批量大小: {cls.batch_size} × {cls.gradient_accumulation_steps} = {cls.batch_size * cls.gradient_accumulation_steps}")



class PatchTSTEncoder(nn.Module):

    def __init__(self, seq_len = Config.seq_len,
                num_features = Config.num_features, 
                patch_len = Config.patch_len, 
                stride = Config.stride, 
                d_model = 128, 
                n_heads = 8, 
                num_layers = 3,
                dropout = 0.2):
        super().__init__()

        self.seq_len = seq_len
        self.num_features = num_features
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model

        self.num_patches = (seq_len - patch_len) // stride + 1

        self.patch_embedding = nn.Linear(patch_len, d_model)

        self.positional_encoding = nn.Parameter(
            torch.randn(1, self.num_patches, d_model)
        )
        
      
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
       
        self.feature_aggregation = nn.Sequential(
            nn.Linear(d_model * num_features, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, num_features, seq_len]
        Returns:
            temporal_features: [batch_size, num_patches, d_model] - 用于Cross-Attention
            pooled_features: [batch_size, d_model] - 用于最终分类
        """
        batch_size = x.shape[0]
        

        all_features = []
        
        for feat_idx in range(self.num_features):
            
            feat_data = x[:, feat_idx, :]
            
            patches = feat_data.unfold(dimension=1, size=self.patch_len, step=self.stride)
            
            patch_embeds = self.patch_embedding(patches)
  
            patch_embeds = patch_embeds + self.positional_encoding
            
            encoded = self.transformer(patch_embeds)  # [B, num_patches, d_model]
            
            all_features.append(encoded)
        

        concatenated = torch.cat(all_features, dim=-1)
        
     
        temporal_features = self.feature_aggregation(
            concatenated.reshape(batch_size, self.num_patches, -1)
        )
        
       
        pooled_features = temporal_features.mean(dim=1)
        
        return temporal_features, pooled_features


class CrossAttentionFusion(nn.Module):
    
    def __init__(self, visual_dim, temporal_dim, fusion_dim=512, n_heads=8, dropout=0.2):
        super().__init__()
        
        self.visual_dim = visual_dim
        self.temporal_dim = temporal_dim
        self.fusion_dim = fusion_dim
        
        # 投影到统一维度
        self.visual_proj = nn.Linear(visual_dim, fusion_dim)
        self.temporal_proj = nn.Linear(temporal_dim, fusion_dim)
        
        # 双向Cross-Attention
        
        self.v2t_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.t2v_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        
        self.ln_v = nn.LayerNorm(fusion_dim)
        self.ln_t = nn.LayerNorm(fusion_dim)
        
        
        self.ffn_v = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim * 4, fusion_dim),
            nn.Dropout(dropout)
        )
        
        self.ffn_t = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim * 4, fusion_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, visual_features, temporal_features):
        """
        Args:
            visual_features: [B, num_visual_patches, visual_dim]
            temporal_features: [B, num_temporal_patches, temporal_dim]
        Returns:
            fused_visual: [B, num_visual_patches, fusion_dim]
            fused_temporal: [B, num_temporal_patches, fusion_dim]
        """
      
        V = self.visual_proj(visual_features)      # [B, Nv, D]
        T = self.temporal_proj(temporal_features)  # [B, Nt, D]
        
       
        T_attn, _ = self.v2t_attention(
            query=self.ln_t(T),
            key=self.ln_v(V),
            value=self.ln_v(V)
        )
        T = T + T_attn
        T = T + self.ffn_t(self.ln_t(T))
        
       
        V_attn, _ = self.t2v_attention(
            query=self.ln_v(V),
            key=self.ln_t(T),
            value=self.ln_t(T)
        )
        V = V + V_attn
        V = V + self.ffn_v(self.ln_v(V))
        
        return V, T




class AdaptiveGatedFusion(nn.Module):

    def __init__(self, fusion_dim=512, dropout=0.1):
        super().__init__()
        self.fusion_dim = fusion_dim
        
        # 样本级别的门控网络
        # 输入: 两个模态的全局特征 [B, 2*D]
        # 输出: 两个模态的权重 [B, 2]
        self.sample_gate = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, 2),
            nn.Softmax(dim=-1)  
        )
        
        # 特征级别的门控网络
        # 输入: 两个模态的全局特征 [B, 2*D]
        # 输出: 逐元素的门控 [B, D]
        self.feature_gate = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
            nn.Sigmoid()  
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, visual_features, temporal_features):
        """
        Args:
            visual_features: [B, Nv, D] - 交叉注意力后的视觉特征
            temporal_features: [B, Nt, D] - 交叉注意力后的时序特征
        
        Returns:
            fused: [B, Nv+Nt, D] - 门控加权后的特征
            gate_info: dict - 包含各种权重信息（用于可视化）
        """
        B, Nv, D = visual_features.shape
        Nt = temporal_features.shape[1]
        

        V_global = visual_features.mean(dim=1)  # [B, D]
        T_global = temporal_features.mean(dim=1)  # [B, D]

        gate_input = torch.cat([V_global, T_global], dim=-1)  # [B, 2*D]
        
        
        sample_weights = self.sample_gate(gate_input)  # [B, 2]
        w_v = sample_weights[:, 0:1].unsqueeze(1)  # [B, 1, 1]
        w_t = sample_weights[:, 1:2].unsqueeze(1)  # [B, 1, 1]
        

        feature_mask = self.feature_gate(gate_input)  # [B, D]
        feature_mask = feature_mask.unsqueeze(1)  # [B, 1, D]
        

        V_weighted = visual_features * w_v * feature_mask
        
        
        T_weighted = temporal_features * w_t * feature_mask
        
       
        fused = torch.cat([V_weighted, T_weighted], dim=1)  # [B, Nv+Nt, D]
        

        gate_info = {
            'sample_weights': sample_weights,  # [B, 2]
            'visual_weight': w_v.squeeze(),    # [B]
            'temporal_weight': w_t.squeeze(),  # [B]
            'feature_mask': feature_mask.squeeze(1),  # [B, D]
        }
        
        return fused, gate_info
    

class SimpleFusionClassifier(nn.Module):
    
    def __init__(self, hidden_size, num_classes):
        super().__init__()
        
        self.attention_pool = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.Tanh(),
            nn.Linear(hidden_size // 4, 1)
        )
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Dropout(0.25),
            nn.Linear(256, num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, features, batch_size):
        num_tokens = features.shape[0] // batch_size
        features = features.view(batch_size, num_tokens, -1)
        
        attn_weights = self.attention_pool(features)
        attn_weights = torch.softmax(attn_weights, dim=1)
        pooled = (features * attn_weights).sum(dim=1)
        
        logits = self.classifier(pooled)
        return logits

class QwenVLDataset(Dataset):
    
    def __init__(self, data_dir, 
                split='train',
                processor=None, 
                use_permutation_in_train=True,
                timeseries_data_path=None):
        """
        Args:
            data_dir: 数据集根目录
            split: 'train', 'val', 或 'test'
            processor: 图像处理器
            use_permutation_in_train: 训练集是否使用排列增强
        """
        self.data_dir = data_dir
        self.split = split
        self.processor = processor
        self.use_permutation_in_train = use_permutation_in_train
        
        with open(os.path.join(data_dir, 'dataset_info.json'), 'r') as f:
            self.dataset_info = json.load(f)

        self.is_permuted = 'num_permutations' in self.dataset_info
        
        if self.is_permuted:
            if split == 'train' and use_permutation_in_train:
                # 训练集：加载所有排列
                self.metadata = self._load_all_permutations()
                print(f"   排列数: {self.dataset_info['num_permutations']}")
                print(f"   总图像数: {len(self.metadata)}")
                print(f"   原始样本数: {len(self.metadata) // self.dataset_info['num_permutations']}")
            else:
                # 验证/测试集：只用perm_0
                self.metadata = self._load_single_permutation(perm_idx=0)
                print(f"   总样本数: {len(self.metadata)}")
        else:
            
            metadata_path = os.path.join(data_dir, split, 'metadata.npy')
            self.metadata = np.load(metadata_path, allow_pickle=True)
        
        self.class_names = self.dataset_info.get('class_names', 
            [f'class_{i}' for i in range(Config.num_classes)])
        self.num_classes = len(self.class_names)
        
        if self.num_classes != Config.num_classes:
            print(f"类别数不匹配 ({self.num_classes} vs {Config.num_classes})")
            self.num_classes = Config.num_classes
        
        self.class_descriptions = self._build_class_descriptions()
        
        # 打印类别分布
        labels = [meta['label'] for meta in self.metadata]
        unique, counts = np.unique(labels, return_counts=True)
        print(f"   类别分布: min={min(counts)}, max={max(counts)}, 类别数={len(unique)}")
        


        self.timeseries_data_path = timeseries_data_path
        if timeseries_data_path and os.path.exists(timeseries_data_path):
            ts_data = np.load(timeseries_data_path, allow_pickle=True).item()
            
            if split == 'train':
                self.timeseries = torch.tensor(
                    ts_data['All_train_data'], dtype=torch.float32
                )  # [N, C, T]
            elif split == 'val':
                self.timeseries = torch.tensor(
                    ts_data['test_data'], dtype=torch.float32
                )
            else:  # test
                self.timeseries = torch.tensor(
                    ts_data['test_data'], dtype=torch.float32
                )
            
            print(f"加载时序数据: {self.timeseries.shape}")
        else:
            self.timeseries = None
            print("仅使用视觉特征")



    def __getitem__(self, idx):
        meta = self.metadata[idx]
        image_path = self._get_image_path(meta)
        label = meta['label']
        
        # 加载图像
        try:
            image = Image.open(image_path).convert('RGB')
            if Config.image_size != 224:
                image = image.resize((Config.image_size, Config.image_size), 
                                    Image.LANCZOS)
        except Exception as e:
            print(f"加载图片失败 {image_path}: {str(e)}")
            image = Image.new('RGB', (Config.image_size, Config.image_size), 
                            color='black')
        
        result = {
            'image': image,
            'label': label,
            'image_path': image_path
        }
        
  
        if self.timeseries is not None:
            #获取原始样本索引
            ts_idx = meta.get('original_sample_idx', idx)
            
            #边界检查
            if ts_idx >= self.timeseries.shape[0]:
                raise IndexError(
                    f"时序数据索引越界！ts_idx={ts_idx}, "
                    f"timeseries.shape={self.timeseries.shape[0]}, "
                    f"dataset_idx={idx}"
                )
            
            result['timeseries'] = self.timeseries[ts_idx]
        
        return result
    def _load_single_permutation(self, perm_idx=0):
        """只加载一个排列（用于验证/测试集）"""
        perm_dir = f'perm_{perm_idx}'
        metadata_path = os.path.join(self.data_dir, self.split, perm_dir, 'metadata.npy')
        
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"未找到metadata文件: {metadata_path}")
        
        metadata = np.load(metadata_path, allow_pickle=True)
        
        # 添加排列信息，方便追踪
        for original_idx, meta in enumerate(metadata):
            meta['permutation_dir'] = perm_dir
            meta['permutation_idx'] = perm_idx
            meta['original_sample_idx'] = original_idx
        
        print(f"   加载 {perm_dir}: {len(metadata)} 样本")
        
        return list(metadata)
    
    def _load_all_permutations(self):
        """加载所有排列的metadata（仅用于训练集）"""
        all_metadata = []
        
        split_dir = os.path.join(self.data_dir, self.split)
        
        # 获取所有perm_X目录
        perm_dirs = sorted([d for d in os.listdir(split_dir) 
                           if d.startswith('perm_') and 
                           os.path.isdir(os.path.join(split_dir, d))])
        
        if not perm_dirs:
            raise ValueError(f"未找到排列目录 (perm_*) 在 {split_dir}")
        
        for perm_dir in perm_dirs:
            metadata_path = os.path.join(split_dir, perm_dir, 'metadata.npy')
            
            if os.path.exists(metadata_path):
                perm_metadata = np.load(metadata_path, allow_pickle=True)
                
                perm_idx = int(perm_dir.split('_')[1])
                             
                for original_idx, meta in enumerate(perm_metadata):
                    meta_copy = dict(meta)
                    meta_copy['permutation_dir'] = perm_dir
                    meta_copy['permutation_idx'] = perm_idx
                    meta_copy['original_sample_idx'] = original_idx       # original_idx在这里而不是生成数据集时加入
                    all_metadata.append(meta_copy)
                
                print(f"   {perm_dir}: {len(perm_metadata)} 样本")
            else:
                print(f"  未找到: {metadata_path}")
        
        return all_metadata
    
    def _build_class_descriptions(self):
        """构建类别描述"""
        descriptions = {}
        for i, class_name in enumerate(self.class_names):
            if isinstance(class_name, str) and len(class_name) > 0:
                descriptions[i] = class_name
            else:
                descriptions[i] = f"pattern_{i}"
        return descriptions
    
    def __len__(self):
        return len(self.metadata)
    
    def _get_image_path(self, meta):
        original_path = meta['image_path']
        cleaned_path = original_path.replace('\\', '/')
        
        # 移除可能的前缀
        prefixes = [
            './lineplot_dataset/',
            'lineplot_dataset/',
            './lineplot_dataset_augmented/',
            'lineplot_dataset_augmented/',
            './lineplot_dataset_augmented_UWaveGestureLibrary/',
            'lineplot_dataset_augmented_UWaveGestureLibrary/'
        ]
        
        for prefix in prefixes:
            if cleaned_path.startswith(prefix):
                cleaned_path = cleaned_path[len(prefix):]
                break
        if self.is_permuted and 'permutation_dir' in meta:
            # 路径格式: split/perm_X/sample_XXXX.png
            perm_dir = meta['permutation_dir']
            filename = os.path.basename(cleaned_path)
            final_path = os.path.join(self.data_dir, self.split, perm_dir, filename)
        else:
            # 原始路径处理
            if cleaned_path.startswith(self.split + '/'):
                final_path = os.path.join(self.data_dir, cleaned_path)
            else:
                final_path = os.path.join(self.data_dir, self.split, 
                                         os.path.basename(cleaned_path))
        
        return final_path
    
class Qwen2VLClassifier(nn.Module):
    
    def __init__(self, 
                model_name=Config.model_name, 
                num_classes=Config.num_classes,
                use_timeseries = True,
                timeseries_config =None):
        super().__init__()
        
        
        self.processor1 = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
            min_pixels=Config.min_pixels,
            max_pixels=Config.max_pixels
        )
        self.processor = self.processor1.image_processor
        
        model_kwargs = {"trust_remote_code": True}
        if Config.use_quantization:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4'
            )
            model_kwargs["quantization_config"] = bnb_config
        
        use_bf16 = Config.fp16 and torch.cuda.is_bf16_supported()
        model_dtype = torch.bfloat16 if use_bf16 else torch.float32
        print(f" 加载模型: {model_name} (dtype: {model_dtype})")
        
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=model_dtype,
            device_map="auto",
            **model_kwargs
        )
        
        self.model.config.use_cache = False
        
        if Config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        
        if Config.use_lora:
            self.setup_lora()
            if Config.freeze_backbone:
                for name, param in self.model.named_parameters():
                    if 'lora' not in name.lower():
                        param.requires_grad = False
        elif Config.freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

        self.use_timeseries = use_timeseries
        if use_timeseries and timeseries_config:
            self.temporal_encoder = PatchTSTEncoder(
                seq_len=timeseries_config['seq_len'],
                num_features=timeseries_config['num_features'],
                patch_len=timeseries_config.get('patch_len', 16),
                stride=timeseries_config.get('stride', 8),
                d_model=timeseries_config.get('d_model', 128),
                n_heads=timeseries_config.get('n_heads', 8),
                num_layers=timeseries_config.get('num_layers', 3)
            ).to(Config.device)
            
            visual_hidden_size = self.model.config.hidden_size
            temporal_hidden_size = timeseries_config.get('d_model', 128)
            
            self.fusion = CrossAttentionFusion(
                visual_dim=visual_hidden_size,
                temporal_dim=temporal_hidden_size,
                fusion_dim=512,
                n_heads=8
            ).to(Config.device)

            self.adaptive_gate = AdaptiveGatedFusion(
                fusion_dim=512,
                dropout=0.1
            ).to(Config.device)
           
            self.classifier = SimpleFusionClassifier(
                hidden_size=512,
                num_classes=num_classes
            ).to(Config.device)
            
            print(f"     序列长度: {timeseries_config['seq_len']}")
            print(f"     特征数: {timeseries_config['num_features']}")
        
        self.num_classes = num_classes
        self._print_params_info()
    
    def setup_lora(self):
        """配置LoRA"""
        if Config.use_quantization:
            self.model = prepare_model_for_kbit_training(self.model)
        
        lora_config = LoraConfig(
            r=Config.lora_r,
            lora_alpha=Config.lora_alpha,
            target_modules=Config.lora_target_modules,
            lora_dropout=Config.lora_dropout,
            bias="none",
            task_type="FEATURE_EXTRACTION",
        )
        
        self.model = get_peft_model(self.model, lora_config)
        print(f"     LoRA模块: {len(Config.lora_target_modules)}个")
    
    def _print_params_info(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        lora_params = sum(p.numel() for n, p in self.named_parameters() 
                         if 'lora' in n.lower() and p.requires_grad)
        classifier_params = sum(p.numel() for p in self.classifier.parameters())
        
        print(f"\n参数统计:")
        print(f"   总参数: {total_params:,}")
        print(f"   可训练: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
        if Config.use_lora:
            print(f"   LoRA: {lora_params:,}")
        print(f"   分类头: {classifier_params:,}")
    
    def forward(self, batch_images, batch_timeseries=None):
        """
        Args:
            batch_images: list of PIL Images
            batch_timeseries: [B, C, T] or None
        Returns:
            logits: [B, num_classes]
        """
        batch_size = len(batch_images)
        
        inputs = self.processor(
            images=batch_images,
            return_tensors="pt"
        ).to(Config.device)
        
        pixel_values = inputs['pixel_values']
        image_grid_thw = inputs.get('image_grid_thw', None)
        
        with torch.cuda.amp.autocast(
            enabled=Config.fp16, 
            dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        ):
            image_embeds = self.model.visual(
                pixel_values, 
                grid_thw=image_grid_thw
            )
        
        patches_per_image = image_embeds.shape[0] // batch_size
        visual_features = image_embeds.view(batch_size, patches_per_image, -1)
        # [B, Nv, Dv]
        
        if self.use_timeseries and batch_timeseries is not None:
            batch_timeseries = batch_timeseries.to(Config.device)
            
            # PatchTST编码: [B, C, T] -> [B, Nt, Dt], [B, Dt]
            temporal_features, _ = self.temporal_encoder(batch_timeseries)
            # temporal_features: [B, Nt, Dt]
            
            fused_visual, fused_temporal = self.fusion(
                visual_features, temporal_features
            )
            # fused_visual: [B, Nv, D_fusion]
            # fused_temporal: [B, Nt, D_fusion]

            gated_features, gate_info = self.adaptive_gate(
                fused_visual, fused_temporal
            )
            
            combined_features = gated_features
            # [B, Nv+Nt, D_fusion]
            combined_features = combined_features.reshape(
                batch_size * (patches_per_image + temporal_features.shape[1]),
                -1
            )
            
            logits = self.classifier(
                combined_features,
                batch_size
            )
        else:
            # 仅使用视觉特征
            logits = self.classifier(image_embeds, batch_size)
        
        return logits




class Qwen2VLTrainer:
    def __init__(self, model, train_dataset, val_dataset, test_dataset):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=Config.batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=self.collate_fn,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=Config.batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=self.collate_fn,
            pin_memory=True
        )
        
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=Config.batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=self.collate_fn,
            pin_memory=True
        )
        

        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.05) 
        self.setup_optimizer()
        use_scaler = Config.fp16 and not torch.cuda.is_bf16_supported()
        self.scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)
        self.train_losses = []
        self.train_accs = []
        self.val_losses = []
        self.val_accs = []
        print(f"   训练样本: {len(train_dataset)}")
        print(f"   验证样本: {len(val_dataset)}")
        print(f"   每epoch步数: {len(self.train_loader)}")
    
    def collate_fn(self, batch):
        images = [item['image'] for item in batch]
        labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)
        
        result = {
            'images': images,
            'labels': labels
        }
        
        if 'timeseries' in batch[0]:
            timeseries = torch.stack([item['timeseries'] for item in batch])
            result['timeseries'] = timeseries
        
        return result
    
    def setup_optimizer(self):
       
        param_groups = []
        all_param_names = set()
        lora_params = []
        for name, param in self.model.named_parameters():
            if 'lora' in name.lower() and param.requires_grad:
                lora_params.append(param)
                all_param_names.add(name)
        
        if lora_params:
            param_groups.append({
                'params': lora_params,
                'lr': Config.learning_rate,
                'name': 'lora',
                'weight_decay': Config.weight_decay
            })
        if hasattr(self.model, 'temporal_encoder') and self.model.temporal_encoder is not None:
            temporal_params = []
            for name, param in self.model.temporal_encoder.named_parameters():
                if param.requires_grad:
                    temporal_params.append(param)
                    full_name = f"temporal_encoder.{name}"
                    all_param_names.add(full_name)
            if temporal_params:
                param_groups.append({
                    'params': temporal_params,
                    'lr': Config.temporal_lr,
                    'name': 'temporal_encoder',
                    'weight_decay': Config.weight_decay
                })
        if hasattr(self.model, 'adaptive_gate') and self.model.adaptive_gate is not None:
            gate_params = []
            for name, param in self.model.adaptive_gate.named_parameters():
                if param.requires_grad:
                    gate_params.append(param)
                    full_name = f"adaptive_gate.{name}"
                    all_param_names.add(full_name)
            
            if gate_params:
                param_groups.append({
                    'params': gate_params,
                    'lr': Config.fusion_lr,  # 使用与fusion相同的学习率
                    'name': 'adaptive_gate',
                    'weight_decay': Config.weight_decay
                })
        if hasattr(self.model, 'fusion') and self.model.fusion is not None:
            fusion_params = []
            for name, param in self.model.fusion.named_parameters():
                if param.requires_grad:
                    fusion_params.append(param)
                    full_name = f"fusion.{name}"
                    all_param_names.add(full_name)
            
            if fusion_params:
                param_groups.append({
                    'params': fusion_params,
                    'lr': Config.fusion_lr,
                    'name': 'fusion',
                    'weight_decay': Config.weight_decay
                })
                

        classifier_params = []
        for name, param in self.model.classifier.named_parameters():
            if param.requires_grad:
                classifier_params.append(param)
                full_name = f"classifier.{name}"
                all_param_names.add(full_name)
        
        if classifier_params:
            param_groups.append({
                'params': classifier_params,
                'lr': Config.classifier_lr,
                'name': 'classifier',
                'weight_decay': Config.weight_decay * 0.5
            })

        
        trainable_param_names = [n for n, p in self.model.named_parameters() if p.requires_grad]
        missing_params = set(trainable_param_names) - all_param_names


        self.optimizer = optim.AdamW(
            param_groups,
            betas=(0.9, 0.999),
            eps=1e-8
        )

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=Config.num_epochs,  # 整个训练周期
            eta_min=1e-7
        )

        total_steps = Config.num_epochs * len(self.train_loader)
        
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc='训练中')
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(pbar):
            batch_data = batch['images']
            labels = batch['labels'].to(Config.device)
            batch_timeseries = batch.get('timeseries', None)
            
            with torch.cuda.amp.autocast(
                enabled=Config.fp16, 
                dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            ):
                logits = self.model(batch_data, batch_timeseries)
                loss = self.criterion(logits, labels)
                scaled_loss = loss / Config.gradient_accumulation_steps
            
            if self.scaler.is_enabled():
                self.scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            
            total_loss += loss.item()
            with torch.no_grad():
                _, predicted = logits.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            
            if (batch_idx + 1) % Config.gradient_accumulation_steps == 0:
                if self.scaler.is_enabled():
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=Config.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=Config.max_grad_norm)
                    self.optimizer.step()
                
                
                self.optimizer.zero_grad()
            
            current_acc = 100. * correct / total if total > 0 else 0
            current_lr = self.optimizer.param_groups[0]["lr"]
            
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{current_acc:.2f}%',
                'LR': f'{current_lr:.2e}'
            })

        if (len(self.train_loader) % Config.gradient_accumulation_steps) != 0:
            if self.scaler.is_enabled():
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=Config.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=Config.max_grad_norm)
                self.optimizer.step()
            self.optimizer.zero_grad()
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total if total > 0 else 0
        
        return avg_loss, accuracy
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='验证中')
            for batch in pbar:
                batch_data = batch['images']
                labels = batch['labels'].to(Config.device)
                batch_timeseries = batch.get('timeseries', None)
                
                with torch.cuda.amp.autocast(
                    enabled=Config.fp16, 
                    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                ):
                    logits = self.model(batch_data, batch_timeseries)
                    loss = self.criterion(logits, labels)
                
                total_loss += loss.item()
                _, predicted = logits.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                current_acc = 100. * correct / total if total > 0 else 0
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{current_acc:.2f}%'
                })
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total if total > 0 else 0
        
        return avg_loss, accuracy, all_preds, all_labels
    
    def train(self):
        os.makedirs(Config.output_dir, exist_ok=True)
        best_val_acc = 0
        patience_counter = 0
        

        for epoch in range(Config.num_epochs):
            print(f"\nEpoch {epoch+1}/{Config.num_epochs}")
            
            train_loss, train_acc = self.train_epoch()
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            
            current_lr = self.optimizer.param_groups[0]['lr']
            
            val_loss, val_acc, _, _ = self.validate()
            self.scheduler.step()  
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)
            
            print(f" Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%, LR: {current_lr:.2e}")
            print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
            
  
            if val_acc > best_val_acc + Config.min_improvement:
                best_val_acc = val_acc
                patience_counter = 0
                
                save_path = os.path.join(Config.output_dir, 'best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                }, save_path)
                
                print(f"保存最佳模型 (Val Acc: {val_acc:.2f}%)")
            else:
                patience_counter += 1
                print(f"未提升 ({patience_counter}/{Config.early_stop_patience})")
            
            if patience_counter >= Config.early_stop_patience:
                print(f"\n早停，最佳Val Acc: {best_val_acc:.2f}%")
                break
        
        print(f"\n完成。。最佳Val Acc: {best_val_acc:.2f}%")
        return best_val_acc
    
    def test(self):
        torch.cuda.empty_cache()
        
        checkpoint = torch.load(
            os.path.join(Config.output_dir, 'best_model.pth'),
            map_location='cpu'
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(Config.device)
        torch.cuda.empty_cache()
        
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc='测试中'):
                batch_data = batch['images']
                labels = batch['labels'].to(Config.device)
                batch_timeseries = batch.get('timeseries', None)
                
                with torch.cuda.amp.autocast(
                    enabled=Config.fp16, 
                    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                ):
                    logits = self.model(batch_data, batch_timeseries)
                _, predicted = logits.max(1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = accuracy_score(all_labels, all_preds)
        report = classification_report(
            all_labels, all_preds,
            target_names=self.test_dataset.class_names,
            digits=4
        )
        
        print(f"\n测试准确率: {accuracy*100:.2f}%")
        print("\n分类报告:")
        print(report)
        
        self.plot_confusion_matrix(all_labels, all_preds)
        
        return accuracy, report
    
    def plot_confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.test_dataset.class_names,
                   yticklabels=self.test_dataset.class_names)
        plt.title('Confusion Matrix (Improved V2)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        save_path = os.path.join(Config.output_dir, 'confusion_matrix.png')
        plt.savefig(save_path, dpi=300)
        print(f"混淆矩阵已保存: {save_path}")
        plt.close()
    
    def plot_training_history(self):
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        epochs = range(1, len(self.train_losses) + 1)
        
        axes[0].plot(epochs, self.train_losses, 'b-', label='Train', linewidth=2)
        axes[0].plot(epochs, self.val_losses, 'r-', label='Val', linewidth=2)
        axes[0].set_title('Loss Curve (Improved V2)')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(epochs, self.train_accs, 'b-', label='Train', linewidth=2)
        axes[1].plot(epochs, self.val_accs, 'r-', label='Val', linewidth=2)
        axes[1].set_title('Accuracy Curve (Improved V2)')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(Config.output_dir, 'training_history.png')
        plt.savefig(save_path, dpi=300)
        print(f"训练历史已保存: {save_path}")
        plt.close()


def main():

    
    Config.print_config()
    

    timeseries_config = {
        'seq_len':1197,        # 根据你的数据调整
        'num_features':6,    # 根据你的数据调整
        'patch_len': 128,
        'stride':64,
        'd_model': 128,
        'n_heads': 8,
        'num_layers': 3
    }
    
    timeseries_data_path = '/data/iamlisz/time/UEA/Cricket.npy'  # 修改为UEA数据集路径
    
    try:

        model = Qwen2VLClassifier(
            model_name=Config.model_name,
            num_classes=Config.num_classes,
            use_timeseries=True,  
            timeseries_config=timeseries_config
        )
        
        train_dataset = QwenVLDataset(
            Config.data_dir, 
            split='train', 
            processor=model.processor,
            use_permutation_in_train=True,
            timeseries_data_path=timeseries_data_path  
        )
        
        val_dataset = QwenVLDataset(
            Config.data_dir, 
            split='val', 
            processor=model.processor,
            use_permutation_in_train=False,
            timeseries_data_path=timeseries_data_path  
        )
        
        test_dataset = QwenVLDataset(
            Config.data_dir, 
            split='test', 
            processor=model.processor,
            use_permutation_in_train=False,
            timeseries_data_path=timeseries_data_path  
        )
        

        print("数据集统计:")

        print(f"训练集: {len(train_dataset)} 图像")
        print(f"验证集: {len(val_dataset)} 图像 ")
        print(f"测试集: {len(test_dataset)} 图像 ")

        

        trainer = Qwen2VLTrainer(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset
        )
        
        best_val_acc = trainer.train()
        torch.cuda.empty_cache()
        
        test_acc, test_report = trainer.test()
        trainer.plot_training_history()
        
        results = {
            'version': 'Qwen2.5-VL',
            'dataset_info': {
                'train_images': len(train_dataset),
                'val_images': len(val_dataset),
                'test_images': len(test_dataset),
                'train_uses_permutation': True,
                'val_uses_permutation': False,
                'test_uses_permutation': False,
            },
            'best_val_accuracy': float(best_val_acc),
            'test_accuracy': float(test_acc * 100),
            'config': {
                'lora_r': Config.lora_r,
                'learning_rate': Config.learning_rate,
                'classifier_lr': Config.classifier_lr,
                'batch_size': Config.batch_size * Config.gradient_accumulation_steps,
            }
        }
        
        results_path = os.path.join(Config.output_dir, 'results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"   最佳验证准确率: {best_val_acc:.2f}%")
        print(f"   测试准确率: {test_acc*100:.2f}%")
        print(f"   结果已保存至: {results_path}")
        
    except Exception as e:
        print(f"\n训练出错: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()




    


    
