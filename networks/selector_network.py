import torch
import torch.nn as nn
import torch.nn.functional as F


class DFPSelector(nn.Module):
    """DFP选择器网络
    
    用于预测输入区域特征应该使用哪个动态特征池(DFP)
    """
    
    def __init__(self, feature_dim: int, num_dfp: int, hidden_dim: int = 128, dropout: float = 0.1):
        """
        Args:
            feature_dim: 输入特征维度
            num_dfp: DFP的数量
            hidden_dim: 隐藏层维度
            dropout: dropout比率
        """
        super(DFPSelector, self).__init__()
        
        self.feature_dim = feature_dim
        self.num_dfp = num_dfp
        
        # 特征处理层
        self.feature_processor = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        
        # 分类头，输出每个DFP的概率
        self.classifier = nn.Linear(hidden_dim, num_dfp)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            features: 输入特征，形状 [batch_size, feature_dim] 或 [batch_size * H * W * D, feature_dim]
            
        Returns:
            logits: DFP选择的logits，形状 [batch_size, num_dfp] 或对应的展平形状
        """
        original_shape = features.shape
        
        # 确保是2D张量
        if features.dim() > 2:
            features = features.view(-1, features.shape[-1])
        
        # 特征处理
        processed_features = self.feature_processor(features)
        
        # 分类
        logits = self.classifier(processed_features)
        
        # 恢复原始形状（除了最后一维）
        if len(original_shape) > 2:
            new_shape = original_shape[:-1] + (self.num_dfp,)
            logits = logits.view(new_shape)
        
        return logits
    
    def predict_dfp(self, features: torch.Tensor) -> torch.Tensor:
        """预测最佳DFP索引
        
        Args:
            features: 输入特征
            
        Returns:
            dfp_indices: 预测的DFP索引，形状与输入的前n-1维相同
        """
        with torch.no_grad():
            logits = self.forward(features)
            dfp_indices = torch.argmax(logits, dim=-1)
        return dfp_indices


class AdaptivePoolingSelector(nn.Module):
    """带自适应池化的DFP选择器
    
    适用于不同大小的输入特征图
    """
    
    def __init__(self, feature_dim: int, num_dfp: int, hidden_dim: int = 128, 
                 output_size: int = 8, dropout: float = 0.1):
        """
        Args:
            feature_dim: 输入特征维度
            num_dfp: DFP的数量  
            hidden_dim: 隐藏层维度
            output_size: 自适应池化后的空间尺寸
            dropout: dropout比率
        """
        super(AdaptivePoolingSelector, self).__init__()
        
        self.feature_dim = feature_dim
        self.num_dfp = num_dfp
        self.output_size = output_size
        
        # 3D自适应平均池化
        self.adaptive_pool = nn.AdaptiveAvgPool3d(output_size)
        
        # 卷积特征提取器
        self.conv_extractor = nn.Sequential(
            nn.Conv3d(feature_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm3d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout),
            
            nn.Conv3d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm3d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout),
        )
        
        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, num_dfp)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            features: 输入特征图，形状 [batch_size, feature_dim, H, W, D]
            
        Returns:
            logits: DFP选择的logits，形状 [batch_size, num_dfp]
        """
        # 自适应池化到固定尺寸
        x = self.adaptive_pool(features)  # [batch_size, feature_dim, output_size, output_size, output_size]
        
        # 卷积特征提取
        x = self.conv_extractor(x)  # [batch_size, hidden_dim//2, output_size, output_size, output_size]
        
        # 全局平均池化
        x = self.global_pool(x)  # [batch_size, hidden_dim//2, 1, 1, 1]
        x = x.view(x.size(0), -1)  # [batch_size, hidden_dim//2]
        
        # 分类
        logits = self.classifier(x)  # [batch_size, num_dfp]
        
        return logits
    
    def predict_dfp(self, features: torch.Tensor) -> torch.Tensor:
        """预测最佳DFP索引"""
        with torch.no_grad():
            logits = self.forward(features)
            dfp_indices = torch.argmax(logits, dim=-1)
        return dfp_indices


def create_selector(selector_type: str = "simple", **kwargs) -> nn.Module:
    """创建选择器网络的工厂函数
    
    Args:
        selector_type: 选择器类型 ("simple" 或 "adaptive")
        **kwargs: 其他参数
        
    Returns:
        选择器网络实例
    """
    if selector_type == "simple":
        return DFPSelector(**kwargs)
    elif selector_type == "adaptive":
        return AdaptivePoolingSelector(**kwargs)
    else:
        raise ValueError(f"Unknown selector type: {selector_type}") 