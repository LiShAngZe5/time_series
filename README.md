# time_series
画图代码里需要根据实际数据集修改的部分：
    1.args.model_id = "Cricket" 数据集名称更换
    2.class Config:
      def __init__(self):
          self.seq_len =1197     时间步
          self.pred_len = 0      
          self.top_k = 2         
          self.d_model =6        特征数量
          self.d_ff = 256         
          self.num_kernels = 6
          self.grid = (3, 3)    网格数
    3.create_dataset_with_permutations函数里CANloader中UEA数据集地址
    4.main函数里排列增强数量


训练代码里需要调整的
    1.class Config:
      data_dir = '/data/iamlisz/time/Cricket-train/lineplot_dataset'   图像数据集路径
      num_classes = 12                                                 类别数需要根据画图代码的输出修改
      model_name = "/data/iamlisz/time/Qwen2.5-VL-7B-Instruct/Qwen2.5-VL-7B-Instruct"  模型位置
  
      seq_len =1197   根据数据集填写时间步
  
      num_features = 6    根据数据集填写特征数

    2.main函数里
      timeseries_config = {
          'seq_len':1197,        根据你的数据调整
          'num_features':6,     根据你的数据调整
          'patch_len': 128,
          'stride':64,
          'd_model': 128,
          'n_heads': 8,
          'num_layers': 3
      }
      
      timeseries_data_path = '/data/iamlisz/time/UEA/Cricket.npy'  修改为UEA数据集路径
    
