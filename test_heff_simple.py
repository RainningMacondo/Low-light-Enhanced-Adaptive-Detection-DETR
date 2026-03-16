"""
HEFF模块简单测试脚本

只测试HEFF模块本身，不依赖HybridEncoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# 复制HEFF模块的核心代码
class BasicBlock(nn.Module):
    """基础残差块"""
    def __init__(self, in_channels, out_channels, stride=1, act='silu'):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.SiLU(inplace=True) if act == 'silu' else nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.SiLU(inplace=True) if act == 'silu' else nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.act2(out)


class BottleneckBlock(nn.Module):
    """瓶颈残差块"""
    def __init__(self, in_channels, out_channels, stride=1, expansion=4, act='silu'):
        super().__init__()
        hidden_channels = out_channels // expansion

        self.conv1 = nn.Conv2d(in_channels, hidden_channels, 1, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.act1 = nn.SiLU(inplace=True) if act == 'silu' else nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_channels)
        self.act2 = nn.SiLU(inplace=True) if act == 'silu' else nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(hidden_channels, out_channels, 1, 1, 0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        self.act3 = nn.SiLU(inplace=True) if act == 'silu' else nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.act2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        return self.act3(out)


class HEFFLayer(nn.Module):
    """HEFF 单层融合模块"""
    def __init__(self, in_channels, out_channels, block_type='bottleneck',
                 num_blocks=2, expansion=4, act='silu'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # 特征投影
        self.proj_high = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)
        self.proj_low = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)

        # 选择block类型
        BlockClass = BottleneckBlock if block_type == 'bottleneck' else BasicBlock
        if block_type == 'bottleneck':
            self.blocks = nn.Sequential(*[
                BlockClass(out_channels, out_channels, stride=1, expansion=expansion, act=act)
                for _ in range(num_blocks)
            ])
        else:
            # BasicBlock不需要expansion参数
            self.blocks = nn.Sequential(*[
                BlockClass(out_channels, out_channels, stride=1, act=act)
                for _ in range(num_blocks)
            ])

        # 输出融合
        self.fusion_conv = nn.Conv2d(out_channels * 2, out_channels, 1, 1, 0, bias=False)
        self.fusion_bn = nn.BatchNorm2d(out_channels)
        self.fusion_act = nn.SiLU(inplace=True) if act == 'silu' else nn.ReLU(inplace=True)

    def forward(self, feat_low, feat_high):
        # 投影到统一维度
        proj_low = self.proj_low(feat_low)
        proj_high = F.interpolate(self.proj_high(feat_high),
                                  size=feat_low.shape[2:],
                                  mode='bilinear',
                                  align_corners=True)

        # 拼接并应用残差块
        fused = torch.cat([proj_low, proj_high], dim=1)
        x = self.fusion_act(self.fusion_bn(self.fusion_conv(fused)))
        x = self.blocks(x)

        return x


class HEFF(nn.Module):
    """完整的HEFF模块"""
    def __init__(self, in_channels=[512, 1024, 2048], hidden_dim=256,
                 num_fusion_layers=2, block_type='bottleneck', num_blocks=2,
                 expansion=4, act='silu', share_weights=False):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.num_fusion_layers = num_fusion_layers
        self.share_weights = share_weights

        # 输入投影
        self.input_proj = nn.ModuleList()
        for in_channel in in_channels:
            self.input_proj.append(
                nn.Sequential(
                    nn.Conv2d(in_channel, hidden_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(hidden_dim)
                )
            )

        # HEFF融合层
        if share_weights:
            self.heff_layer = HEFFLayer(
                in_channels=hidden_dim, out_channels=hidden_dim,
                block_type=block_type, num_blocks=num_blocks,
                expansion=expansion, act=act
            )
            self.heff_layers = [self.heff_layer] * (len(in_channels) - 1)
        else:
            self.heff_layers = nn.ModuleList([
                HEFFLayer(hidden_dim, hidden_dim, block_type, num_blocks, expansion, act)
                for _ in range(len(in_channels) - 1)
            ])

        # 输出卷积
        self.output_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(inplace=True) if act == 'silu' else nn.ReLU(inplace=True)
            )
            for _ in range(len(in_channels))
        ])

    def forward(self, feats):
        # 投影到统一维度
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]

        # Top-down融合
        inner_outs = [proj_feats[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_high = inner_outs[0]
            feat_low = proj_feats[idx - 1]
            fused_feat = self.heff_layers[len(self.in_channels) - 1 - idx](feat_low, feat_high)
            inner_outs.insert(0, fused_feat)

        # 输出卷积
        outs = [self.output_convs[i](feat) for i, feat in enumerate(inner_outs)]
        return outs


def test_heff():
    """测试HEFF模块"""
    print("="*80)
    print("HEFF模块测试")
    print("="*80)

    batch_size = 2
    feats = [
        torch.randn(batch_size, 512, 64, 64),
        torch.randn(batch_size, 1024, 32, 32),
        torch.randn(batch_size, 2048, 16, 16),
    ]

    print("\n1. 测试标准HEFF (bottleneck blocks)")
    heff_std = HEFF(
        in_channels=[512, 1024, 2048],
        hidden_dim=256,
        num_fusion_layers=2,
        block_type='bottleneck',
        num_blocks=2,
        expansion=4,
        act='silu',
        share_weights=False
    )

    outs_std = heff_std(feats)
    params_std = sum(p.numel() for p in heff_std.parameters())

    print(f"  输入shapes: {[f.shape for f in feats]}")
    print(f"  输出shapes: {[o.shape for o in outs_std]}")
    print(f"  参数量: {params_std:,}")

    assert len(outs_std) == 3
    assert outs_std[0].shape == torch.Size([batch_size, 256, 64, 64])
    assert outs_std[1].shape == torch.Size([batch_size, 256, 32, 32])
    assert outs_std[2].shape == torch.Size([batch_size, 256, 16, 16])
    print("  [PASS]")

    print("\n2. 测试轻量级HEFF (basic blocks + 权重共享)")
    heff_lite = HEFF(
        in_channels=[512, 1024, 2048],
        hidden_dim=256,
        num_fusion_layers=1,
        block_type='basic',
        num_blocks=1,
        expansion=1,
        act='silu',
        share_weights=True
    )

    outs_lite = heff_lite(feats)
    params_lite = sum(p.numel() for p in heff_lite.parameters())

    print(f"  输出shapes: {[o.shape for o in outs_lite]}")
    print(f"  参数量: {params_lite:,}")

    assert len(outs_lite) == 3
    assert outs_lite[0].shape == torch.Size([batch_size, 256, 64, 64])
    print("  [PASS]")

    print("\n3. 参数量对比")
    print(f"  标准HEFF: {params_std:,} 参数")
    print(f"  轻量级HEFF: {params_lite:,} 参数")
    print(f"  参数减少: {(1 - params_lite/params_std) * 100:.1f}%")

    print("\n" + "="*80)
    print("[SUCCESS] All tests passed!")
    print("="*80)

    print("\n下一步:")
    print("1. 启动训练: conda activate coalCnn")
    print("2. 运行HEFF配置: python tools/train.py -c configs/rtdetr/rtdetr_r50vd_meikuang_heff_denoising.yml")


if __name__ == '__main__':
    test_heff()
