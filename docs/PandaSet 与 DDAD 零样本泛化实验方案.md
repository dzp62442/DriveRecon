# PandaSet 与 DDAD 零样本泛化实验方案

> 状态：方案已获用户确认并完成代码实现；CPU 回归、真实资产接口、源权重加载和真实 VGG 掩码指标对照已通过。尚未运行正式零样本评估。日期：2026-10-01。验证范围见第 12 节。
> 核对版本：DriveRecon `comp_svfgs@c650851184c758ec4fe15e822997f52d20841b95`；SVF-GS `9a93216a11af6277dd3bd9c7d97abe89ff3e97fa`。
> 用户已确认：先评估 112×200；同时准备 224×400 配置，待同分辨率 OmniScene 预训练权重就绪后再运行，不默认进行跨分辨率测试。
> 用户已确认：DDAD 独立测试和训练中 mini 指标评估均默认使用自车掩码；训练及验证损失不受影响。

## 1. 目标、输入与实验边界

加载本项目在 OmniScene 风格 nuScenes 上训练完成的 DriveRecon 静态六相机权重，在 PandaSet、DDAD 的既有预处理数据上直接测试，不微调、不做测试时优化、不更新参数或归一化统计。网络继续使用 `T=1`，一次读取中央时刻的 6 路 RGB 和相机内外参，生成一份高斯场景，渲染 18 个目标视角。

SVF-GS 当前的两个数据集均使用 `svfgs_temporal18_v1`：每个相机的记录为 `[center, before, after]`。本方案遵循当前的真实 18 视角协议，不采用早期只输出输入 6 视角的版本：

- 输入：中央时刻 6 路环视相机。
- 目标索引 `0:12`：按相机顺序排列的 before/after，两张一组，共 12 个新视角。
- 目标索引 `12:18`：中央 6 路输入视角。
- before/after 的 RGB 只用于训练监督或测试指标，其相机参数用于渲染；不送入重建网络，不引入历史状态或运动位移。

报告 `all_18`、`novel_12`、`input_6` 三组 PSNR、SSIM、LPIPS；PCC 保留为诊断字段，参考深度改为已有 Metric3D-v2 米制深度，暂不作为论文正式指标。继续记录参数量与完整高斯重建耗时。

两个新数据集均按用户要求视作静态：不生成、不读取动态物体掩码。DDAD 自车掩码是独立的测试有效像素掩码，只改变 12 个新视角的指标计算范围。

**原有数据集路径不改动。** 不修改 `train.py`、`train_omniscene.py`、`comp_svfgs/`、`scene/`、原生光栅器及 `configs/omniscene/` 的现有实现；只在新增入口中导入可直接复用的模型、渲染、训练与恢复工具。原 OmniScene 的两组指标、DA2 深度读取、训练掩码、续训与运行命令全部保持原样。

**不审查数据集。** 不移植 SVF-GS 加载器中的 manifest 完成标志门禁、文件哈希核验、选帧身份比对、位姿/内参合法性判断、深度范围/有限值检查、图像质量检查或有效像素数量门禁；不重选帧、不筛选、不丢弃、不重复补齐样本。只按已发布索引逐项正常读取，缺失文件等正常 I/O 错误不通过伪造数据或切换 split 掩盖。配置和模型权重的兼容性检查与数据质量审查分开处理。

## 2. 已核对的资源与限制

### 2.1 数据路径

规划阶段只读取 SVF-GS 源码、目录及少量记录结构，未对数据内容进行扫描审查。实施时已在 DriveRecon 的 `datasets/` 下建立以下软链接，指向 SVF-GS 使用的共享数据：

| 本项目路径 | 现有共享数据根目录 | 预处理目录 |
| --- | --- | --- |
| `datasets/PandaSet` | `/home/B_UserData/dongzhipeng/Datasets/PandaSet` | `datasets/PandaSet/processed` |
| `datasets/DDAD` | `/home/B_UserData/dongzhipeng/Datasets/DDAD` | `datasets/DDAD/processed` |

共享目录目前均包含 `bins_test.json`、`bin_infos/`、`images_small/`、`params_small/` 和已有 Metric3D-v2 产物。PandaSet 深度目录为 `dptm/`，DDAD 为 `dptm_small/`；实际读取以每个 sensor 记录中的路径为准，不凭文件名拼接猜测。

读取现有索引得到 PandaSet test 为 **264 bins**，DDAD test 为 **324 bins**。这些是当前资产清单的规模，不硬编码为加载器的通过条件。

两者当前仅有 `bins_test.json`，没有 `bins_train.json`。新增训练配置和训练接口会实现，但实际训练需由 SVF-GS 提供对应的 train 预处理产物。本项目不运行预处理，不复制 test 清单作为 train，也不接入 `processed_only_input` 等旧产物作为兜底。

### 2.2 源权重

112×200 零样本配置默认使用：

```text
work_dirs/omniscene/driverecon_static_t1_112x200/checkpoints/step-00100001/
```

此目录为完整的 `state.pt` 检查点，来自已经完成的 `FP32 参数存储 + BF16 前向 + AdamW + 正确 AABB 反向` 实验。不能把它标注为原版 BF16 参数/Adam 配方。

224×400 配置保留独立源目录：

```text
work_dirs/omniscene/driverecon_static_t1_224x400/checkpoints/step-00100001/
```

当前尚无该分辨率正式权重；只提供配置，不自动退回 112×200 权重，不据此宣称 224×400 实验可立即运行。命令行可显式指定另一个同分辨率检查点目录。

## 3. 新增代码与配置组织

已新增：

```text
train_cross_dataset.py
cross_dataset/
  __init__.py
  config.py              # 新入口参数与配置加载
  dataset.py             # PandaSet/DDAD 共用的 6→18 接口与 split 映射
  io.py                  # 读取现有 RGB、K、位姿、Metric3D
  ego_mask.py            # DDAD 已处理掩码的查表、缩放与缓存
  metrics.py             # 三组汇总及 DDAD masked 指标
  evaluation.py          # 独立评估循环、来源记录、进度、耗时
  checkpoint.py          # 零样本权重加载与目标域续训身份区分
  trainer.py             # 仅必要的目标域训练编排适配
configs/
  pandaset/{112x200,224x400}.py
  ddad/{112x200,224x400}.py
  zero_shot/
    omniscene_to_pandaset_112x200.py
    omniscene_to_pandaset_224x400.py
    omniscene_to_ddad_112x200.py
    omniscene_to_ddad_224x400.py
tests/
  test_cross_dataset.py
  test_cross_dataset_metrics.py
```

每份目标域训练配置直接以对应 `configs/omniscene/<resolution>.py` 为 `_base_`，覆盖 dataset（以 `_delete_=True` 替换 OmniScene 专用字段）、工作目录及新数据集评估选项；不复制并修改已有模型、优化器或损失配置。零样本配置再继承对应目标域配置，设置 `mode='test'`、源 checkpoint、测试输出目录及 DDAD 掩码默认值。

新配置的 `evaluation.eval_use_ego_mask` 在 DDAD 中默认 True、PandaSet 中默认 False；该字段只传给指标评估使用的 test/mini loader 和 evaluator，不传给训练或验证 loss 的 loader。

仍使用 `mmcv.Config.fromfile()` 和 `--cfg-options`。配置优先级为继承配置 < 数据集/零样本配置 < 命令行。`--print-config` 只解析配置，不读数据、不加载模型、不访问 CUDA，因此没有 train 资产或 224×400 权重时也能检查配置。

新入口提供 `--mode train|test`、`--work-dir`、`--data-root`、`--checkpoint`、`--test-split total|mini`、`--cfg-options` 和 `--print-config`。内部数据集 split 只取 `train/val/test`；`total/mini` 是 test 的范围选择，不新增一套数据划分。

实施时仅需在 `.gitignore` 新增 `/datasets/PandaSet`、`/datasets/DDAD` 两条本地链接忽略规则，并在 README 新增中文运行说明。原 README 内容和原实验文档不作伴随重写。

## 4. 数据加载与几何适配

### 4.1 数据集划分

严格对应 SVF-GS 当前 `TemporalDataset` 与 `SVFGSDataModule`：

| DriveRecon 新入口用途 | SVF-GS 对应用途 | 索引和选择方式 |
| --- | --- | --- |
| train | `split='train'` | `bins_train.json`，原顺序交给已有训练 sampler |
| val | `split='val'` | 从 `bins_test.json` 按 `np.linspace(0,N-1,min(10,N),dtype=int)` 取 10 个或更少 |
| test / total | `split='test'` | `bins_test.json` 的全部项目，保持列表顺序 |
| test / mini | `split='mini-test'` | 从 test 按 `np.linspace(0,N-1,min(100,N),dtype=int)` 取 100 个或更少 |

mini 规模沿用新数据集在 SVF-GS 的 100-bin 协议，不套用 OmniScene 的间隔 14、最多 2,048 bins。训练节奏仍保持 OmniScene 设置。val/mini 是既有 test 清单的子集，不声称为额外独立留出集。PandaSet 的 test 也不改称官方独立测试划分；DDAD 的项目 test 对应其官方 validation 预处理产物。

### 4.2 相机与视角排列

| 逻辑相机顺序 | PandaSet 物理相机 | DDAD 物理相机 | novel 索引 | input 目标索引 |
| --- | --- | --- | --- | --- |
| `CAM_FRONT` | `front_camera` | `CAMERA_01` | 0、1 | 12 |
| `CAM_FRONT_RIGHT` | `front_right_camera` | `CAMERA_06` | 2、3 | 13 |
| `CAM_FRONT_LEFT` | `front_left_camera` | `CAMERA_05` | 4、5 | 14 |
| `CAM_BACK` | `back_camera` | `CAMERA_09` | 6、7 | 15 |
| `CAM_BACK_LEFT` | `left_camera` | `CAMERA_07` | 8、9 | 16 |
| `CAM_BACK_RIGHT` | `right_camera` | `CAMERA_08` | 10、11 | 17 |

直接读取 `bin_infos/<token>.pkl` 中 `sensor_info[logical_camera]` 的三条记录：第一条组装 context，后两条组装 novel，最后将中央六条追加到 target。重复的中央目标可直接复用已读张量，不重复读盘。

### 4.3 RGB、内参与外参

从 `processed_root` 解析 `data_path`、`intrinsic_path`、`depth_path`。不加载 `source_path` 指向的原始图像，不访问原始点云，不重新根据行驶距离选帧，不重新做主点居中裁剪。

RGB 使用已有 `images_small` 的 224×400 图像，转换为 `[0,1]` RGB。112×200 以 PIL bilinear 缩放，224×400 直接使用；同时按实际缩放比例缩放 `params_small` 中的 `camera_intrinsic` 前两行。保留当前分辨率的像素 K，以及分别除以 W、H 的归一化 K。

`sensor2lidar_transform` 将相机局部坐标映射到该 bin 的中央参考坐标系；字段名含 lidar 不表示需要读取 LiDAR 数据。相机局部均保留 **OpenCV 坐标约定**，但公共参考系需要区分数据集：PandaSet 沿用现有外参；DDAD 预处理使用 **X 向前、Y 向左、Z 向上**，须在加载边界转换到 OmniScene / nuScenes 训练使用的 **X 向右、Y 向前、Z 向上**。不能由“相机局部约定相同”推断“公共坐标也相同”。

DDAD 的全部输入和目标外参统一左乘下式，即 `(x, y, z) → (-y, x, z)`，同时转换 `R` 和 `t`：

```text
B = [[ 0, -1,  0,  0],
     [ 1,  0,  0,  0],
     [ 0,  0,  1,  0],
     [ 0,  0,  0,  1]]
c2w_model = B @ sensor2lidar_transform
R_model = B[:3, :3] @ R_prepared
t_model = B[:3, :3] @ t_prepared
```

转换在 `CrossDataset.__getitem__()` 中、组装 context/target 之前完成；中央六路只转换一次，再复用为目标最后六路。DDAD 的 train/val/test、mini/total、两档分辨率和自车掩码开关均走同一路径。相对位姿、相机局部射线、像素投影和米制深度保持不变；公共坐标下的射线方向、相机中心、模型反投影位置，以及渲染所需的逆外参和组合投影矩阵，均由转换后的外参在既有路径中计算，没有另一份需要手工更新的几何缓存。网络的几何位置融合会使用这些公共坐标，不能假定其对公共坐标旋转不敏感。

SVF-GS 的 `camera_tensors()` 另乘 `diag(1,-1,-1,1)` 是为其 OpenGL 接口服务，不能照搬到 DriveRecon，也不复用其 half-pixel 射线构造。DDAD 公共坐标对齐不改变相机局部 OpenCV 约定。

继续调用既有 `comp_svfgs.camera.target_cameras`、模型内反投影和原生 surfel renderer；不修改原相机投影公式、不移动场景原点、不归一化位移尺度。

既有预处理资产保留 DDAD 原始参考系，无需重新预处理。DDAD 评估元数据新增 `camera_frame=nuscenes_axes_x_right_y_forward_z_up` 与 `reference_to_model` 记录加载后的坐标约定；协议、启动指令、实验名和输出路径保持原样。旧 DDAD 结果由用户清理后按原命令重新评估，本次修复不删除或改写实验结果，不涉及模型原有 bug。

### 4.4 深度、静态标签和 batch

Metric3D-v2 的 `depth_path` 已是米制深度。读取后按 SVF-GS 的 PIL float bilinear 缩放；不反转为 disparity、不做逐图 min-max、不重复乘焦距或尺度、不新增截断。DriveRecon 不需要深度置信图，因此不读取 `confidence_path`，也不加载 Metric3D 模型重新预测。

沿用现有模型与训练函数需要的 batch 契约：

| 字段 | 单样本形状/用途 |
| --- | --- |
| `context.image` | `[6,3,H,W]`，唯一 RGB 网络输入 |
| `context.extrinsics` | `[6,4,4]`，OpenCV 相机到中央参考系；DDAD 已统一为 nuScenes 公共轴 |
| `context.intrinsics_pixel` / `intrinsics` | `[6,3,3]`，像素/归一化 K |
| `context.metric_depth` | train/val 的 `[6,H,W]` 米制辅助监督；零样本重建不需要 |
| `context.segmentation_label` | train/val 中创建全 0 的 `[6,H,W]` 静态类别标签，不读取掩码文件 |
| `target.image` / 相机参数 | `[18,3,H,W]` 及对应的 K、位姿 |
| `target.loss_mask` | train/val 中创建全 1 的 `[18,H,W]`，18 路 RGB 损失均使用全图 |
| `target.metric_depth` | test 中的 `[18,H,W]`，仅用于 PCC 参考 |
| `target.eval_mask` | DDAD 开启自车掩码时的 `[18,H,W]` bool，独立于 `loss_mask` |
| `bin_token` / `scene` | 保留原 token、scene_id，用于输出与掩码查表 |

DataLoader 增加 batch 维，batch size 始终为 1。测试时的全部目标 RGB、Metric3D 深度及自车掩码仅供 evaluator 使用，不传给 `model(context)`；不能因 SVF-GS 也使用 Metric3D 作为输入而改变 DriveRecon 的网络输入。

## 5. 训练配置与源模型保持一致

本次不启动目标域训练，但新接口必须能在相应 train 资产就绪后按下表运行：

| 项目 | 固定配置 |
| --- | --- |
| 两档图像分辨率 | 112×200、224×400，输入与目标同尺寸 |
| 网络 | `scene.PointNet.UNet + Guassian_Adaptor`，复用 `StaticDriveRecon` |
| UNet 通道 | in=3、out=128；down=(64,128,256,256)，up=(256,256,128) |
| 注意力 | down=(False,False,False,False)，mid=True，up=(True,True,False) |
| 其他 UNet 参数 | layers_per_block=1，skip_scale=√0.5；view_num=6、num_frames=1 |
| 高斯输出 | 保留半分辨率网格：56×100 或 112×200；不增减网络 head |
| 深度与高斯范围 | 200 个类别，0.1–400 米；scale=0.001–4.0；seg_num=3；max_shift=5 |
| 时间相关模块 | 结构保留，T=1，不应用运动位移或跨时间损失 |
| Renderer | 原生 2D Gaussian；znear=0.01、zfar=1e8、黑背景；AABB backward version=1 |
| 参数/前向精度 | FP32 参数、梯度和优化器状态；BF16 autocast 前向 |
| 优化器 | AdamW；lr=4e-4、weight_decay=0.05、betas=(0.9,0.999)、eps=1e-15 |
| 学习率调度 | 恒定，不添加 warmup、余弦衰减或额外参数组 |
| RGB 损失 | 每视角 L1，全图均值，18 路求和，权重 1 |
| 分割损失 | 保留 3 类 head 与交叉熵，真值全静态类别 0，权重 1 |
| 深度分类 | 按原 200 个采样点映射 Metric3D 标签，交叉熵权重 2 |
| 深度回归 / 辅助几何 | 复用当前 L2 的单位、归一化与像素范围，权重分别为 2 / 2 |
| 动态掩码 | `use_dynamic_mask=False`，不读文件；与 DDAD 自车评估掩码分离 |
| batch / 进程 / 梯度累计 | train、val、test 均为 1；单 GPU、单进程、累计 1 |
| 总更新数 | 100,001 次实际 optimizer 更新 |
| 验证 | 每 1,000 次更新，使用上文 val 子集 |
| 保存 | 每 5,000 次更新及最终步保存完整状态 |
| 训练中 mini | 每 10 次验证，即每 10,000 步；100,001 步后必须再执行一次 |
| 续训 | 只从当前目标域工作目录自动恢复完整状态；不把零样本源权重当续训点 |
| 日志 | 每 100 步训练日志；评估每 100 bins 及末尾显示进度、速度和 ETA |
| 随机性 | seed=0，确定性设置与当前 OmniScene 配置一致 |
| DataLoader | num_workers=0、pin_memory=True，与当前 OmniScene 一致 |
| W&B / 通知 | W&B 默认关闭，开启也强制 offline；复用 `auto_monitor.send_feishu` |

全静态标签只替换新数据集的监督内容，不关闭分割 head 或擅自修改分割损失权重。深度监督仍用当前 `auxiliary_losses()`：分类的 `target>0.1`、回归与辅助几何的像素范围属于原损失定义，不作为数据集审查或整样本跳过条件。

训练侧尽量直接复用现有 `Trainer`、`train_update()`、`validate()`、sampler 与原子 checkpoint 保存。新数据 batch 提供全静态标签和全 1 loss mask，即可复用原更新逻辑；训练中评估注入新的三组 Evaluator。仅在新 `CrossDatasetTrainer` 中补充目标域续训身份检查及三组 mini 通知，不修改已有 Trainer，也不复制整个优化循环。

目标域工作目录独立为 `work_dirs/pandaset/driverecon_static_t1_<resolution>/` 与 `work_dirs/ddad/driverecon_static_t1_<resolution>/`。训练开始和 mini 完成继续通过已有 `Notifier` 发送关键日志，不使用 feishu-cli；代码开发或配置检查本身不发送通知。

按用户确认，DDAD 独立测试、训练中周期 mini 和最终 mini 均默认开启自车掩码，仅作用于 12 个新视角的指标；训练更新和验证 loss 仍使用全图，后 6 个输入视角的指标也始终使用全图。SVF-GS 当前把开关限制在全局 `mode=test`，本项目只复用其掩码读取和指标定义，不移植该模式限制；改由新入口按 loader/evaluator 的用途传递开关。显式关闭时所有指标评估恢复全图口径。

## 6. 零样本加载与结果隔离

新入口在 `mode=test` 时只执行严格的模型状态加载、`eval()` 和 `no_grad()` 推理，不建立 optimizer，不恢复源训练 sampler/RNG，不运行任何目标域训练更新。

当前 `load_checkpoint()` 会比较训练 loss、optimizer 等配置，不能把跨数据集适配变成修改原 checkpoint 兼容规则。新增加载函数在 CPU 读取完整源状态，检查模型结构、参数精度、renderer 版本及图像分辨率等模型兼容项，并以 `strict=True` 加载完整模型；dataset、输出目录、分组与评估掩码允许不同。模型不兼容属于配置错误，不属于数据集审查。

保存源检查点绝对路径、global_step、源配置、源分辨率、目标数据集、目标分辨率、split 与像素协议。源 checkpoint 文件不被重写。参数量仍按训练时 `requires_grad` 口径统计，不能因为推理无梯度就把可训练参数全部报告成冻结参数。

零样本输出与训练目录分离，例如：

```text
work_dirs/zero_shot/omniscene_112x200_to_pandaset_112x200/
  step-00100001/test/total/novel18_s10_d1p6_min0p1/full_image/
work_dirs/zero_shot/omniscene_112x200_to_ddad_112x200/
  step-00100001/test/total/novel18_s10_d1p6_min0p1/ddad_ego_novel12_v1/
  step-00100001/test/total/novel18_s10_d1p6_min0p1/full_image/
```

两个 DDAD 像素协议不得覆盖彼此；训练中 mini 的输出目录也需包含该像素协议，自定义源实验使用不同 `--work-dir`。224×400 输出同样包含源/目标分辨率，不写入现有 OmniScene 正式结果目录。

## 7. 三组指标、PCC 和 DDAD 自车掩码

### 7.1 分组与汇总

逐视角计算 PSNR、SSIM、LPIPS，再对每个 bin 的 `all_18=0:18`、`novel_12=0:12`、`input_6=12:18` 分别等权平均；最后对 split 内全部 bin 等权平均。PCC 在每个 bin 的每个视角组内展平深度计算，再对 bin 平均。

开启 DDAD 自车掩码时，`all_18` 的 RGB 指标是 12 个 masked 新视角与 6 个全图输入视角分数的等权平均，即 `(12×novel_12 + 6×input_6)/18`。不能改成按各视角有效像素数加权；上述等式不适用于组内展平的 PCC。

原 `comp_svfgs.metrics` 与 Evaluator 只汇总两组，不直接扩展它们；新模块复用其中的全图指标基础函数和 VGG 权重加载逻辑，新增自己的三组统计。原 OmniScene 输出格式不变。

### 7.2 Metric3D PCC

参考为 `target.metric_depth`，保持米制深度，不转 disparity，不加载 DA2。使用原 `renderer.render_pcc_depth()` 的 `accumulated_center_z`；不改成 alpha 归一化深度、surfel 交点深度，也不对 PCC 深度做 RGB clamp。

无掩码时用本项目现有无状态 Pearson 函数；有掩码时在组内选择有效位置后展平，再计算同一 Pearson 公式。不引入 SVF-GS 的有状态 `PearsonCorrCoef` 实例或其数据拒绝检查。结果记录 `pcc_reference='metric3d_v2'`、`pcc_role='diagnostic_only'`，避免与 OmniScene 的 DA2-PCC 混为同一参考口径。

### 7.3 DDAD 自车掩码读取

默认路径：`datasets/DDAD/processed/ego_masks/vidar_v1/manifest.json`。

只复用已经处理好的 PNG，查表路径为：

```text
scene_id + sensor.camera
→ manifest.scene_camera_to_variant[scene_id][camera]
→ manifest.variants[variant_id].mask_id
→ manifest.masks[mask_id].path
→ 相对于 ego_masks/vidar_v1/ 的已处理 PNG
```

PNG 中 255 表示有效像素、0 表示自车遮挡；以 PIL nearest 缩放到实际评估尺寸，按 mask_id 和分辨率缓存。前 12 路使用各自目标记录对应的物理相机掩码，后 6 路始终创建全 1 掩码。

不重新下载模板、不做连通域清理、不重新 warp/crop、不推断未知场景的替代掩码；这些均由 SVF-GS 预处理完成。新实现不调用带全量审查的 `DDADEgoMasks` 类，不检查资产哈希、原始内参或裁剪一致性。可将小型 mask manifest 的哈希写入结果作为来源记录，但不以哈希进行拒绝加载的判断。关闭开关时不读任何自车掩码文件。

### 7.4 masked 指标的精确定义

| 指标 | 与 SVF-GS 一致的计算 |
| --- | --- |
| PSNR | 将 RGB 裁剪到 `[0,1]`；`MSE=sum(M*(pred-gt)^2)/(3*sum(M))`，再计算 `-10*log10(MSE)` |
| SSIM | `win_size=11`、Gaussian weights、sigma=1.5、sample covariance=True、data_range=1；获取完整 SSIM map，只平均 11×11 窗口全在有效区的中心（方形 binary erosion，图像边界无效） |
| LPIPS | 同一 VGG 权重和 `normalize=True`；指标内部用 `pred_eval=where(M,pred,gt)`，`spatial=True` 获得距离图，再在 M 的有效像素处归一化平均 |
| PCC | 只选择组内 M 有效位置的参考/渲染深度计算 Pearson，不把无效区置零后计入样本 |

整张掩码全 1 的视角直接走原全图函数，保证 `input_6` 数值口径不变。空间 LPIPS 使用独立实例、同一套本地预训练权重，不修改全图实例的 `spatial` 属性，不触发在线下载；实例仅作评估，不进入模型参数统计。

不将预测和 GT 同时涂黑后直接计算全图均值。掩码不进入网络、不改变渲染结果或保存的图像；开启后的分数变化仅代表评价像素范围变化，不能表述为模型渲染能力提升。

用户明确禁止数据审查，因此不会照搬 SVF-GS 中“没有有效 SSIM 窗口”“深度常值”等主动中断检查。数学上未定义的计算保留 NaN，不伪造成 0 或好分数，不跳过该 bin，也不使用 `nanmean` 隐藏它；正常有限输入时数值定义与参考实现一致。PSNR 的零误差对应 `+Inf`，不额外裁剪分数。

## 8. 输出、耗时与进度

每次评估保存：

- `resolved_config.py`：含源 checkpoint、目标数据集、分辨率、split 与掩码开关。
- `evaluation_summary.json`：`complete`、processed_bins、三组指标与各组样本数、源权重信息、像素协议、PCC 参考、精度、设备和总评估时间。
- `per_bin_metrics.csv`：每 bin 三行，包含 token、scene_id、group、PSNR/SSIM/LPIPS/PCC、像素协议及掩码来源标识。
- `parameter_counts.json`：可训练、冻结、总参数量；排除评估专用 LPIPS。
- `reconstruction_timing.json`：逐 bin 耗时及 mean/median/P95，前 5 个 bin 仅排除测速、不排除质量指标。
- 运行日志和原生 renderer 的版本/来源记录；若选择保存渲染图，保持未经指标掩码修改的原图。

默认模型参数量预期仍为可训练/总参数 **58,249,264**、冻结 **0**，最终以实际模型遍历结果报告。

耗时边界沿用 OmniScene：数据已送入 GPU、同步后开始，覆盖完整 `model(context)` 到高斯场景组装结束，再同步计时；不包括文件读取、CPU→GPU 传输、18 路渲染与指标计算。每个 bin 只重建一次，不按三个指标分组重复推理。正式测速应在 GPU 无其他计算任务时进行。

每 100 bins 以及最后一个 bin 输出进度、耗时、速度、ETA；只有循环正常结束并写完结果才设 `complete=True`。不因 PCC 的非有限诊断值主动中断，但汇总应如实显示该数值状态。

## 9. 实施后的运行命令设计

以下接口已实现。112×200 使用现有 OmniScene 权重；224×400 仍需先准备对应源权重。

首次接入已有数据链接（本机已完成，其他机器按实际路径建立）：

```bash
mkdir -p datasets
ln -s /home/B_UserData/dongzhipeng/Datasets/PandaSet datasets/PandaSet
ln -s /home/B_UserData/dongzhipeng/Datasets/DDAD datasets/DDAD
```

112×200 正式零样本评估，配置内已指定 OmniScene 的 100001 步源权重：

```bash
conda activate drivingrecon

CUDA_VISIBLE_DEVICES=0 python train_cross_dataset.py \
  --config configs/zero_shot/omniscene_to_pandaset_112x200.py --mode test --test-split total

CUDA_VISIBLE_DEVICES=0 python train_cross_dataset.py \
  --config configs/zero_shot/omniscene_to_ddad_112x200.py --mode test --test-split total
```

DDAD 默认开启自车掩码；显式关闭，输出自动落在独立的 `full_image` 子目录：

```bash
CUDA_VISIBLE_DEVICES=0 python train_cross_dataset.py \
  --config configs/zero_shot/omniscene_to_ddad_112x200.py --mode test --test-split total \
  --cfg-options evaluation.eval_use_ego_mask=False
```

`--cfg-options evaluation.eval_use_ego_mask=True` 可显式开启。该选项也控制未来 DDAD 训练中的周期/最终 mini 指标评估，不影响训练或验证 loss；对其他数据集开启属于配置错误。`--checkpoint <完整检查点目录>` 可替换配置中的源权重。

224×400 同理使用后缀 `_224x400.py` 的两份零样本配置，但需先准备同分辨率 OmniScene 权重。

目标域训练配置示例，仅在对应 train 预处理数据可用、且用户另行决定训练后使用：

```bash
CUDA_VISIBLE_DEVICES=0 python train_cross_dataset.py --config configs/pandaset/112x200.py --mode train
CUDA_VISIBLE_DEVICES=0 python train_cross_dataset.py --config configs/ddad/112x200.py --mode train
```

两者也提供 `224x400.py`。此训练入口从目标域工作目录自动续训或从头训练，不隐式把零样本源 checkpoint 当作目标域预训练初始化。

## 10. 实施顺序与验收

1. 新增配置、CLI 和数据 adapter，确认两档配置继承的模型、损失、优化器、训练节奏与当前 OmniScene 相同，差异仅为本方案列出的数据和评估字段。
2. 新增三组 evaluator、Metric3D-PCC、DDAD mask loader 和 masked 指标；使用单独测试验证指标数值和作用范围。
3. 新增严格的零样本模型加载路径、目标域训练适配、输出隔离及 README 中文说明；补建本地数据链接，不生成预处理资产。
4. 运行必要的 CPU 回归；GPU 空闲且用户授权范围允许时，才做小规模实际功能验证。正式全量实验单独启动，不在配置测试中自动发起。

验收重点如下，均用于检查实现，不建立生产数据质量审查机制：

- 用合成 fixture 验证 train/val/test 和 mini 选择公式、6→18 排列、两档 resize/K 缩放及中央目标复用。
- 用合成几何验证 OpenCV 投影往返；避免错误复用 SVF-GS 的 OpenGL 翻转。
- 修改目标 RGB/Metric3D/自车 mask 不应改变高斯重建；测试不得把目标信息送入模型。
- 不存在动态掩码文件也可读取新数据；测试模式不需要 `bins_train.json`；训练不回退到 test。
- 对相同有限张量对照 SVF-GS：全图、全 1 mask、部分 mask 的 PSNR/SSIM/LPIPS/PCC 与三组汇总一致；仅扰动遮挡区预测时 masked 指标不变，仅扰动有效区时指标改变。
- DDAD mask 开关不改变高斯、渲染、`input_6` 指标或源权重；全图和 masked 输出分目录记录。
- DDAD 训练配置默认让周期/最终 mini 使用自车掩码，同时验证 train/val loader 不读取自车掩码、训练更新和验证 loss 不受开关影响。
- 零样本加载保持完整严格权重匹配，无优化步、无参数和 BatchNorm 状态变化；224×400 不隐式使用 112×200 源权重。
- 复用现有 CPU 回归，重点确认训练更新、恢复事件、原生 VJP 与旧两组指标行为不变；核对受保护旧文件相对实施前版本无差异。
- 将共享小型样本用于按需集成检查时，只核对适配代码的输出，不调用 SVF-GS 的全量校验、重选帧或预处理脚本。

本次交付包含新入口、数据适配、两档配置、指标、训练恢复适配、测试及中文运行说明。目标域 train 资产与 OmniScene 224×400 权重仍是未来对应实验的外部前置资源，不在本项目内制造替代品。

## 11. 代码依据

- DriveRecon：`train_omniscene.py`；`configs/omniscene/{112x200,224x400,model,runtime,dataset}.py`。
- DriveRecon：`comp_svfgs/{dataset_omniscene,omniscene_io,model,camera,losses,metrics,evaluation,trainer,checkpoint,notifications}.py`。
- SVF-GS：`configs/build_config.py::build_zero_shot_config/build_eval_mask_config`、`configs/main.yaml`。
- SVF-GS：`data_module.py`、`data/temporal_dataset.py`、`data/transforms/temporal_loading.py`。
- SVF-GS：`data/transforms/ego_mask.py`、`tools/metrics.py::compute_image_metrics/compute_eval_pcc`、`tools/ablation_metrics.py::render_records`。
- SVF-GS：`docs/零样本泛化实验/PandaSet 数据集适配方案.md` 与 `DDAD 数据集适配方案.md`。其中数据审查和中断规则不移植，以本次用户要求为准。

## 12. 实现与验证记录（2026-10-01）

代码已经按本方案实现，原训练循环通过 `CrossDatasetTrainer` 继承复用，原模型、相机、渲染器、损失与 OmniScene 入口未修改。README 仅插入新的中文跨数据集说明，原有内容逐字节保留；数据软链接已加入忽略规则。用户原有暂存区没有被改动。

开发检查输出统一位于 `/tmp/driverecon-cross-dataset-20261001-TYZVRi/`，不写入正式实验结果目录：

| 验证 | 实际结果 |
| --- | --- |
| 全部 CPU 回归 | 45 项通过，含原有 32 项和新增 13 项；覆盖数据接口、三组指标、mask 不变性、未定义指标保留、5k 保存及中断恢复 |
| 真实预处理数据接口 | PandaSet/DDAD 各读取第一个 test/val 样本，两档分辨率均通过；6 输入、18 目标、输入目标复用、静态监督与 DDAD mask 路由正确；未扫描审查数据内容 |
| 真实 VGG masked 指标 | CPU 上对照 SVF-GS 的 `compute_image_metrics` 函数，PSNR/SSIM/LPIPS 最大绝对差均为 0；修改被遮挡区预测不影响 masked 指标 |
| 正式源权重加载 | 在 CPU 上严格加载 112×200 的 100001 步 OmniScene 完整模型；总/可训练参数 58,249,264，冻结 0，源训练优化器不进入测试流程 |
| 原路径隔离 | 对原入口、`comp_svfgs`、`scene`、OmniScene 配置及 surfel 子目录共 1,594 个已跟踪文件逐项比较，内容未变 |
| GPU 功能检查 | 尚未运行：检查时 GPU 总占用 1,150 MiB，超过用户此前约定的低于 1 GB 调试阈值；没有干扰其他进程 |

已经提供 `tests/smoke_cross_dataset_cuda.py`，只允许输出到 `/tmp`，启动前检查 GPU 显存占用。空闲后可运行：

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 python tests/smoke_cross_dataset_cuda.py \
  --output /tmp/driverecon-cross-dataset-smoke
```

该脚本在两数据集、两档分辨率各做一个 bin 的功能检查，包括三组评估、DDAD mask 开关、一次诊断更新、验证及完整状态保存/恢复，不发送通知。112×200 使用真实 OmniScene 权重；224×400 仅使用随机初始化检查接口，不能视为零样本泛化结果。训练接口的真实数据检查使用明确标记的 val 样本，不生成或冒充目标域 train 索引。

PCC 或其他数学上未定义的数值在 CSV 中保留，在 JSON 中沿用原 `atomic_json` 的显式字符串表示（如 `"nan"`），不丢弃该 bin、不转成好分数。正式报告仍需运行第 9 节的全量命令；上述 CPU 检查不是实际 CUDA 渲染结果或测速结果。

## 13. DDAD 公共坐标对齐修复（2026-10-01）

核对 VolSplat 的 DDAD 修复、SVF-GS 预处理代码以及本项目加载路径后，确认此前也遗漏了第 4.3 节的公共坐标对齐。真实样本的前相机光轴在 DDAD 预处理参考系中约为 `(0.99768, 0.06738, -0.00935)`，转换后为 `(-0.06738, 0.99768, -0.00935)`；作为方向约定参照，读取的 OmniScene 训练样本约为 `(0.00954, 0.99978, 0.01854)`。各数据集的实际相机安装角不同，无需让这些数值完全相等。

运行逻辑仅修改 `cross_dataset/dataset.py`：DDAD 的所有输入、目标相机同步转换旋转和平移，并记录加载坐标元数据。模型几何分支、高斯中心反投影和渲染相机继续使用现有代码，从转换后的外参计算派生量；不修改模型、渲染器、其他数据集或指标实现，也不处理原模型其他 bug。

| 检查 | 结果 |
| --- | --- |
| CPU 回归 | 48 项全部通过，新增 3 项覆盖旋转/平移、中央视角复用、各加载分支、派生几何和坐标元数据 |
| 合成几何 | 两档分辨率，train/val/test、mini/total、自车掩码开关均覆盖；6 输入及 18 目标外参只转换一次；相对位姿与像素投影保持一致 |
| 真实资产加载 | PandaSet/DDAD 各取一个 bin，覆盖两档分辨率与可用的 val/test、mini/total、掩码开关，共 16 组；与修改前加载器对照，RGB、内参、深度、标签、掩码和划分不变，PandaSet 的全部输出不变；没有扫描或审查全量数据 |
| 真实派生几何 | 相对位姿最大绝对差 `3.58e-7`；反投影位置与预期公共坐标转换的最大差为 `0`；相同三维点的齐次渲染投影最大差 `2.86e-6`，均在 float32 误差范围内 |
| 修改范围 | 原入口、模型、相机、渲染器、配置和 README 等 1,611 个已跟踪文件相对本次修改前内容未变 |

本次检查全部使用 CPU，输出位于 `/tmp/driverecon-ddad-frame-sBB55p2i/`，没有运行训练或正式推理，也未删除、改写已有结果。上述检查证明加载与几何的一致性，不代表修复后的指标；用户清理旧 DDAD 零样本结果后，继续使用原命令、原实验名重新评估即可。DDAD 自车掩码仍默认开启，关闭方式不变。
