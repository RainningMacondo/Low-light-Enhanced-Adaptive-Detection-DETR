# HEFF + Denoising Training 快速上手指南

## 🚀 3分钟快速开始

### 步骤1: 环境准备 (1分钟)

```bash
# 激活conda环境
conda activate coalCnn

# 进入项目目录
cd F:\dzr-cnn\rtdetr_pytorch
```

### 步骤2: 测试HEFF模块 (1分钟)

```bash
# 返回项目根目录并运行测试
cd F:\dzr-cnn
python test_heff_simple.py

# 看到以下输出表示成功：
# [SUCCESS] All tests passed!
```

### 步骤3: 开始训练 (1分钟)

```bash
# 进入RT-DETR目录
cd F:\dzr-cnn\rtdetr_pytorch

# 启动HEFF + Denoising训练 (推荐配置)
python tools/train.py -c configs/rtdetr/rtdetr_r50vd_meikuang_heff_denoising.yml

# 训练启动时会打印架构配置，确认看到：
# Mode: Encoder-Free (HEFF enabled)
# Denoising Training: ✓ ENABLED
```

---

## 📋 实验配置对比

### 推荐配置 (按优先级排序)

#### 1️⃣ HEFF + Denoising (最佳性能)

```bash
python tools/train.py -c configs/rtdetr/rtdetr_r50vd_meikuang_heff_denoising.yml
```

**特点**:
- ✅ HEFF替代Transformer Encoder (bottleneck blocks)
- ✅ Denoising training (100 queries)
- ✅ 保留PAN路径
- ✅ 6层decoder
- ⭐ **推荐用于最终实验**

**预期**: 性能最优，训练时间中等

---

#### 2️⃣ HEFF-Lite (快速验证)

```bash
python tools/train.py -c configs/rtdetr/rtdetr_r50vd_meikuang_heff_lite.yml
```

**特点**:
- ✅ HEFF (basic blocks + 权重共享)
- ✅ Denoising training
- ✅ 仅3层decoder
- ✅ 禁用PAN (最快速度)
- ⭐ **推荐用于快速验证**

**预期**: 训练速度最快，参数量最少

---

#### 3️⃣ 标准RT-DETR (Baseline)

```bash
python tools/train.py -c configs/rtdetr/rtdetr_r50vd_meikuang.yml
```

**特点**:
- ❌ 原始Transformer Encoder (1层)
- ✅ Denoising training
- ✅ 保留PAN路径
- ✅ 6层decoder
- ⭐ **用于对比baseline**

**预期**: 标准性能，用于对比

---

### 消融实验配置 (可选)

#### 4️⃣ 禁用Denoising

```bash
python tools/train.py -c configs/rtdetr/rtdetr_r50vd_meikuang_ablation_no_denoising.yml
```

**目的**: 验证denoising training的效果

---

#### 5️⃣ 禁用Encoder

```bash
python tools/train.py -c configs/rtdetr/rtdetr_r50vd_meikuang_ablation_no_encoder.yml
```

**目的**: 验证transformer encoder的作用

---

#### 6️⃣ 减少Decoder层数

```bash
python tools/train.py -c configs/rtdetr/rtdetr_r50vd_meikuang_ablation_decoder3.yml
```

**目的**: 验证decoder深度的影响

---

#### 7️⃣ 禁用PAN路径

```bash
python tools/train.py -c configs/rtdetr/rtdetr_r50vd_meikuang_ablation_no_pan.yml
```

**目的**: 验证bottom-up fusion的作用

---

## 📊 实验记录表

| 配置 | 训练命令 | mAP@0.5 | mAP@0.5:0.95 | 训练时间 | 推理速度 |
|------|---------|---------|-------------|---------|---------|
| Baseline | `rtdetr_r50vd_meikuang.yml` | - | - | - | - |
| HEFF | `rtdetr_r50vd_meikuang_heff_denoising.yml` | - | - | - | - |
| HEFF-Lite | `rtdetr_r50vd_meikuang_heff_lite.yml` | - | - | - | - |
| No Denoising | `ablation_no_denoising.yml` | - | - | - | - |
| No Encoder | `ablation_no_encoder.yml` | - | - | - | - |
| Decoder-3 | `ablation_decoder3.yml` | - | - | - | - |
| No PAN | `ablation_no_pan.yml` | - | - | - | - |

**填写说明**:
- 训练完成后，从 `output/<config_name>/log.txt` 中提取最终mAP
- 训练时间: 从训练日志中记录总时长
- 推理速度: 运行评估时查看FPS

---

## 🔍 训练监控

### 启动时检查清单

训练启动后，检查以下输出：

```
================================================================================
RT-DETR Architecture Configuration
================================================================================

📦 Encoder Configuration:
  - Mode: Encoder-Free (HEFF enabled)  ← 确认看到这个
  - HEFF enabled: True                   ← 确认看到这个
  ...

🔧 Decoder Configuration:
  - Denoising Training: ✓ ENABLED        ← 确认看到这个
  - num_decoder_layers: 6                ← 检查层数
  ...

📉 Loss Configuration:
  - Losses: ['vfl', 'boxes']
  - Weights: {...}
  - num_classes: 2                       ← 确认类别数正确
```

### 训练过程监控

关键指标:
- **loss_vfl**: Varifocal Loss (分类损失)
- **loss_bbox**: L1 Loss (bbox回归损失)
- **loss_giou**: GIoU Loss ( IoU损失)
- **loss_gcd**: GCD Loss (小目标损失)

正常训练趋势:
- 所有loss应逐步下降
- 前10个epoch下降最快
- 50-100 epoch后趋于稳定

---

## 🛠️ 常见问题

### Q1: 训练报错 "CUDA out of memory"

**解决方案**:
```yaml
# 修改配置文件，减小batch size
# 找到 dataloader 配置项
train_dataloader:
  batch_size: 2  # 从4或8改为2
```

---

### Q2: HEFF参数无法从预训练权重迁移

**说明**: 这是正常的，HEFF是新模块，没有对应的预训练权重

**解决方案**:
```yaml
# 使用backbone预训练权重
tuning: /path/to/rtdetr_r50vd_6x_coco_from_paddle.pth

# HEFF参数会随机初始化，从头训练
```

---

### Q3: 如何验证HEFF是否启用？

**方法1**: 查看启动日志，确认看到：
```
Mode: Encoder-Free (HEFF enabled)
HEFF enabled: True
```

**方法2**: 运行测试脚本
```bash
python test_heff_simple.py
```

---

### Q4: Denoising训练如何禁用？

**方法1**: 使用已提供的配置
```bash
python tools/train.py -c configs/rtdetr/rtdetr_r50vd_meikuang_ablation_no_denoising.yml
```

**方法2**: 修改配置文件
```yaml
RTDETRTransformer:
  num_denoising: 0  # 改为0
```

---

### Q5: 如何调整HEFF的复杂度？

**轻量级** (快速训练):
```yaml
HybridEncoder:
  encoder_free: true
  heff_num_fusion_layers: 1     # 减少融合层数
  heff_block_type: 'basic'      # 使用基础块
  heff_num_blocks: 1            # 减少残差块数量
  heff_share_weights: true      # 权重共享
```

**标准** (平衡性能):
```yaml
HybridEncoder:
  encoder_free: true
  heff_num_fusion_layers: 2
  heff_block_type: 'bottleneck'
  heff_num_blocks: 2
  heff_share_weights: false
```

**重量级** (最佳性能):
```yaml
HybridEncoder:
  encoder_free: true
  heff_num_fusion_layers: 3     # 增加融合层数
  heff_block_type: 'bottleneck'
  heff_num_blocks: 3            # 增加残差块数量
  heff_share_weights: false
```

---

## 📈 性能基准

### 预期性能对比

| 配置 | 参数量 | 训练速度 | 收敛速度 | 最终mAP |
|------|--------|---------|---------|---------|
| Baseline | 基准 (~40M) | 基准 | 基准 | 基准 |
| HEFF | -50% | +30-50% | +20-40% | +1-2% |
| HEFF-Lite | -70% | +50-70% | +10-20% | 持平或略降 |

*注: 以上为理论预期，实际结果以实验为准*

---

## 📞 获取帮助

### 文档资源

1. **详细配置文档**: `HEFF消融实验配置说明.md`
2. **完成总结**: `HEFF改造完成总结.md`
3. **快速上手**: 本文档

### 代码位置

- HEFF模块: `rtdetr_pytorch/src/zoo/rtdetr/heff.py`
- HybridEncoder: `rtdetr_pytorch/src/zoo/rtdetr/hybrid_encoder.py`
- 配置文件: `rtdetr_pytorch/configs/rtdetr/`
- 测试脚本: `test_heff_simple.py`

---

## 🎯 推荐实验流程

### 阶段1: 快速验证 (1-2天)

1. 运行 `heff_lite.yml` (快速训练)
2. 验证HEFF功能正常
3. 记录初步结果

### 阶段2: 完整实验 (1周)

1. 运行 `heff_denoising.yml` (最佳配置)
2. 运行 baseline 对比
3. 运行消融实验

### 阶段3: 结果分析 (2-3天)

1. 整理所有实验结果
2. 绘制对比图表
3. 撰写实验报告

---

## ✅ 成功标准

### 最小目标

- [ ] HEFF模块测试通过 (`test_heff_simple.py` 成功)
- [ ] 至少完成1个完整训练 (收敛至稳定)
- [ ] 记录训练日志和mAP

### 理想目标

- [ ] 完成7个配置的完整训练
- [ ] 对比分析HEFF vs Transformer Encoder
- [ ] 证明HEFF的优势 (速度/精度/参数量)
- [ ] 撰写完整的实验报告

---

**开始时间**: 2025-01-18
**预计完成**: 2周内完成所有实验

祝实验顺利！🎉
