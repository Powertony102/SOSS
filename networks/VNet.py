import torch
from torch import nn
from .selector_network import create_selector
from myutils.second_order_feature import SecondOrderFeatureModule


class SecondOrderFeatureModule2D(nn.Module):
    """
    2D Adapter for SecondOrderFeatureModule
    Converts 2D features to 3D, applies SecondOrderFeatureModule, then converts back to 2D
    """
    def __init__(self, K, mlp_hidden, eps=1e-5):
        super(SecondOrderFeatureModule2D, self).__init__()
        self.second_order_3d = SecondOrderFeatureModule(K=K, mlp_hidden=mlp_hidden, eps=eps)
        
    def forward(self, x):
        # x: (B, C, H, W) -> (B, C, H, W, 1)
        B, C, H, W = x.shape
        x_3d = x.unsqueeze(-1)  # Add depth dimension
        
        # Apply 3D SecondOrderFeatureModule
        enhanced_3d = self.second_order_3d(x_3d)  # (B, C+K, H, W, 1)
        
        # Remove depth dimension: (B, C+K, H, W, 1) -> (B, C+K, H, W)
        enhanced_2d = enhanced_3d.squeeze(-1)
        
        return enhanced_2d


class ConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ConvBlock, self).__init__()
        ops = []
        for i in range(n_stages):
            input_channel = n_filters_in if i == 0 else n_filters_out
            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False
            ops.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*ops)
    def forward(self, x):
        return self.conv(x)


class ResidualConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ResidualConvBlock, self).__init__()
        ops = []
        for i in range(n_stages):
            input_channel = n_filters_in if i == 0 else n_filters_out
            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False
            if i != n_stages - 1:
                ops.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*ops)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.relu(self.conv(x) + x)


class DownsamplingConvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(DownsamplingConvBlock, self).__init__()
        ops = []
        ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
        if normalization == 'batchnorm':
            ops.append(nn.BatchNorm3d(n_filters_out))
        elif normalization == 'groupnorm':
            ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
        elif normalization == 'instancenorm':
            ops.append(nn.InstanceNorm3d(n_filters_out))
        elif normalization != 'none':
            assert False
        ops.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*ops)
    def forward(self, x):
        return self.conv(x)


class Upsampling_function(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none', mode_upsampling=1):
        super(Upsampling_function, self).__init__()
        ops = []
        if mode_upsampling == 0:
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
        elif mode_upsampling == 1:
            ops.append(nn.Upsample(scale_factor=stride, mode="trilinear", align_corners=True))
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, kernel_size=3, padding=1))
        elif mode_upsampling == 2:
            ops.append(nn.Upsample(scale_factor=stride, mode="nearest"))
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, kernel_size=3, padding=1))
        if normalization == 'batchnorm':
            ops.append(nn.BatchNorm3d(n_filters_out))
        elif normalization == 'groupnorm':
            ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
        elif normalization == 'instancenorm':
            ops.append(nn.InstanceNorm3d(n_filters_out))
        elif normalization != 'none':
            assert False
        ops.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*ops)
    def forward(self, x):
        return self.conv(x)


class Encoder(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False, has_residual=False):
        super(Encoder, self).__init__()
        self.has_dropout = has_dropout
        convBlock = ConvBlock if not has_residual else ResidualConvBlock
        self.block_one = convBlock(1, n_channels, n_filters, normalization=normalization)
        self.block_one_dw = DownsamplingConvBlock(n_filters, 2 * n_filters, normalization=normalization)
        self.block_two = convBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization)
        self.block_three = convBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization)
        self.block_four = convBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16, normalization=normalization)
        self.block_five = convBlock(3, n_filters * 16, n_filters * 16, normalization=normalization)
        self.second_order_module = SecondOrderFeatureModule(
            K=8,
            mlp_hidden=n_filters * 8,
            eps=1e-5
        )
        self.channel_adjust = nn.Conv3d(n_filters * 16 + 8, n_filters * 16, 1, padding=0)
        self.dropout = nn.Dropout3d(p=0.5, inplace=False)
    def forward(self, input):
        x1 = self.block_one(input)
        x1_dw = self.block_one_dw(x1)
        x2 = self.block_two(x1_dw)
        x2_dw = self.block_two_dw(x2)
        x3 = self.block_three(x2_dw)
        x3_dw = self.block_three_dw(x3)
        x4 = self.block_four(x3_dw)
        x4_dw = self.block_four_dw(x4)
        x5 = self.block_five(x4_dw)
        x5_enhanced = self.second_order_module(x5)
        x5 = self.channel_adjust(x5_enhanced)
        if self.has_dropout:
            x5 = self.dropout(x5)
        res = [x1, x2, x3, x4, x5]
        return res


class Decoder(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False, has_residual=False, up_type=0):
        super(Decoder, self).__init__()
        self.has_dropout = has_dropout
        convBlock = ConvBlock if not has_residual else ResidualConvBlock
        self.block_five_up = Upsampling_function(n_filters * 16, n_filters * 8, normalization=normalization, mode_upsampling=up_type)
        self.block_six = convBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_six_up = Upsampling_function(n_filters * 8, n_filters * 4, normalization=normalization, mode_upsampling=up_type)
        self.block_seven = convBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_seven_up = Upsampling_function(n_filters * 4, n_filters * 2, normalization=normalization, mode_upsampling=up_type)
        self.block_eight = convBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_eight_up = Upsampling_function(n_filters * 2, n_filters, normalization=normalization, mode_upsampling=up_type)
        self.block_nine = convBlock(1, n_filters, n_filters, normalization=normalization)
        self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)
        self.dropout = nn.Dropout3d(p=0.5, inplace=False)
    def forward(self, features, with_feature=False):
        x1, x2, x3, x4, x5 = features
        x5_up = self.block_five_up(x5) + x4
        x6 = self.block_six(x5_up)
        x6_up = self.block_six_up(x6) + x3
        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7) + x2
        x8 = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8) + x1
        x9 = self.block_nine(x8_up)
        if self.has_dropout:
            x9 = self.dropout(x9)
        out_seg = self.out_conv(x9)
        if with_feature:
            return out_seg, x9
        else:
            return out_seg


class VNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False, has_residual=False):
        super(VNet, self).__init__()
        self.encoder = Encoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual)
        self.decoder1 = Decoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, 0)
    def forward(self, input):
        features = self.encoder(input)
        out_seg1 = self.decoder1(features)
        return out_seg1


class corf(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False, has_residual=False, feat_dim=32, num_dfp=8, use_selector=True):
        super(corf, self).__init__()
        self.n_filters = n_filters
        self.feat_dim = feat_dim
        self.num_dfp = num_dfp
        self.use_selector = use_selector
        self.encoder1 = Encoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual)
        self.encoder2 = Encoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual)
        self.decoder1 = Decoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, 0)
        self.decoder2 = Decoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, 1)
        self.projection_head1 = nn.Sequential(
            nn.Linear(n_filters, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim)
        )
        self.prediction_head1 = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim)
        )
        self.projection_head2 = nn.Sequential(
            nn.Linear(n_filters, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim)
        )
        self.prediction_head2 = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim)
        )
        if self.use_selector:
            self.dfp_selector = create_selector(
                selector_type="simple",
                feature_dim=feat_dim,
                num_dfp=num_dfp,
                hidden_dim=128,
                dropout=0.1
            )
    def forward(self, input, with_hcc=False, with_selector=False, region_features=None):
        features1 = self.encoder1(input)
        features2 = self.encoder2(input)
        out_seg1, embedding1 = self.decoder1(features1, with_feature=True)
        out_seg2, embedding2 = self.decoder2(features2, with_feature=True)
        output_dict = {
            'seg1': out_seg1,
            'seg2': out_seg2,
            'embedding1': embedding1,
            'embedding2': embedding2
        }
        if with_hcc:
            output_dict['features1'] = features1
            output_dict['features2'] = features2
        if with_selector and self.use_selector and region_features is not None:
            selector_logits = self.dfp_selector(region_features)
            dfp_predictions = torch.argmax(selector_logits, dim=-1)
            output_dict['selector_logits'] = selector_logits
            output_dict['dfp_predictions'] = dfp_predictions
        if not with_hcc and not with_selector:
            return out_seg1, out_seg2, embedding1, embedding2
        elif with_hcc and not with_selector:
            return out_seg1, out_seg2, embedding1, embedding2, features1, features2
        else:
            return output_dict
    
