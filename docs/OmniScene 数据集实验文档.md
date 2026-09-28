# OmniScene 数据集实验文档

> 状态（2026-09-28）：修复光栅器反向后的 FP32+AdamW 实验已完成 100,001 步及 30,080-bin total。本次审查恢复原版 Adam 的候选配置，4k 已出现辅助几何深度退化；用户据此确认保留 FP32 参数/动量和 AdamW 两项稳定性例外。默认配置仍与已完成实验一致，其余模型、损失和学习率设置按发布代码及既定静态协议。
> 第 1–11 节为当前协议；第 12–15 节保留各轮排障历史；第 16 节记录发布代码一致性审查、Adam 回退诊断及用户最终决定。用户已授权 GPU 显存占用低于 1 GB 时进行有界调试。
> 代码核对日期：2026-09-25。DriveRecon：`comp_svfgs@17e25e6`；SVF-GS：`af39b31`；depthsplat：`405b9a5`。

## 1. 实验目标与已确认选择

在 OmniScene 上训练 DriveRecon 的静态六相机版本：每个样本仅以同一中心时刻的 6 路 RGB 和相机参数进行一次前馈重建，得到一份场景高斯，再渲染 18 路目标视角。Metric3D 深度和动态掩码用于训练监督。分别报告 `all_18`、`novel_12` 的 PSNR、SSIM、LPIPS、PCC，以及模型参数量、完整高斯重建耗时。

实验应标注为 **DriveRecon / OmniScene / static-T1**。这是基于公开实现的静态适配实验，不能当作作者原始多帧 4D 实验或其论文 nuScenes 结果的直接复现。

### 1.1 信息边界

| 信息 | 用途与限制 |
| --- | --- |
| 中心 6 路 RGB | 唯一的图像网络输入；不拼接其他时刻，不建立跨样本缓存或历史状态 |
| 相机内外参 | 六路输入的几何计算、18 路目标的渲染；保留每张图各自的标定 |
| 已有 Metric3D 尺度深度 | 使用中心 6 路的米制深度，替代原项目稀疏 LiDAR 深度监督；不增加离线深度生成流程 |
| 已有动态物体掩码 | 新视角训练损失屏蔽；中心输入图的分割监督 |
| 目标 RGB | 仅用于训练/验证监督和测试指标；不得进入重建编码器 |
| 已有 DepthAnything V2 相对深度 | 仅用于 mini/total 测试的 PCC，不作训练输入或深度监督 |
| 网络自行预测的深度、分割和几何特征 | 允许，属于网络中间结果 |
| LiDAR 点云/投影深度、天空掩码、额外语义标签、光流等 | 不读取、不生成、不引入离线预处理依赖 |

OmniScene 的新视角图像可能采自区间两端，但这里仅作为带位姿的监督/评估目标。它们不构成多帧输入，渲染也不使用时间戳或运动位移。

**不审查数据集。** 直接信任已有 split、RGB、位姿、深度和掩码；不做启动前全量扫描，不增加质量阈值、位姿审查、数据指纹门禁、坏样本列表、跳过/替换 bin 或重新划分数据。文件按需正常读取；不以零图、重复样本或丢弃样本掩盖实际 I/O 错误。网络内部的张量适配、损失定义中的像素范围和数值运算，不用于判定整个样本不可用。

### 1.2 用户已确认的实现选择

1. 保留时间注意力、运动预测相关模块，`T=1`；运动位移不作用于高斯，跨时间静态/动态损失关闭。
2. 保留分割头及原交叉熵损失，用已有中心输入动态掩码监督。
3. 两档实验的实际网络输入和目标图像均为指定分辨率；保持原网络半分辨率高斯输出，不把网络输入放大两倍。
4. 正式结果使用 `total`；`mini` 只用于训练中评估，不用 `center150` 替代 total。
5. 深度读取和米制单位按 SVF-GS；保留 DriveRecon 的深度分类、L2 及原权重，将分类标签正确映射到原有 200 个深度采样点。

当前源码没有 RE10K 配置，也没有可选的 Base/Large 模型系列。本方案选用实际训练入口使用的 `scene.PointNet.UNet + Guassian_Adaptor` 默认架构，不从 depthsplat 借用其 RE10K 编码器、预训练权重或损失配置。

## 2. 当前实现与适配范围

### 2.1 原 Waymo 入口的执行路径与适配原因

原 Waymo 路径为 `train.py → WaymoDataset → Scene/Camera → Gaussian_LRM.forward()`。本次 OmniScene 适配处理的问题包括：

- 数据路径硬编码为 Waymo，输入默认 3 个相机、多个时刻；`PointNet.py`、`PD_Block.py`、注意力和 reshape 中还存在相机数/时间长度常量。
- 网络实际使用 `scene/PointNet.py` 的 UNet，不是仓库中的另一份 `scene/unet.py`。
- 原训练循环对一个加载出的 batch 连续更新 500 次，外层循环和优化步数混用。不能仅修改 `iterations` 就认为已经限制为 100,001 次更新。
- `Gaussian_LRM.forward()` 混合了高斯预测、几何组装、渲染、损失计算；原目标索引还被用来推导时间和三相机高斯切片。
- 原 `capture/restore`、`--start_checkpoint` 路径不足以提供本实验所需的完整自动续训。
- 当前主训练循环没有完整的定期验证、mini 测试和最终测试编排，也没有 WandB 在线汇报路径。

已新增独立的 OmniScene 入口和训练器，复用并参数化实际生效的网络/渲染代码。原 Waymo 入口仍可按原默认参数使用。共享模块的新参数提供原值作为默认值，OmniScene 显式传入六相机和单时刻。

### 2.2 与 depthsplat 的复用边界

| 部分 | depthsplat 当前实现 | 本项目方案 |
| --- | --- | --- |
| 配置与入口 | Hydra、`src.main`、Lightning | 保留 DriveRecon 已有 mmcv 配置生态，使用新入口和 Accelerate 训练器 |
| 样本组织 | `context/target` 字典 | 复用这种组织；增加 Metric3D 深度和输入分割标签 |
| 网络输入 | 编码器读取 `[B,V,...]` context | 适配为 DriveRecon 的 `[B,1,6,...]`；T 维只作接口兼容 |
| 内参 | 以图像宽高归一化 | 同时持有归一化 K 与像素 K，DriveRecon 几何层使用当前特征尺度的像素 K |
| 位姿 | OpenCV c2w | 复用；转换为原生光栅器所需的转置 w2c/投影矩阵 |
| 深度监督读取 | 当前 OmniScene loader 不读 Metric3D | 从 SVF-GS 移植米制深度读取，不导入其置信图依赖 |
| 训练掩码 | 加载后转 bool，RGB 损失还会选择有效像素 | 按 SVF-GS 保留 float 掩码，预测和真值同时乘掩码后在整张图上求均值 |
| 分辨率 shim | 默认 `4×4=16` 对齐，112×200 会中心裁剪为 112×192 | 不复用该 shim；保留完整图像，必要时仅对模型内部特征补齐并还原 |
| 高斯渲染 | 3D Gaussian CUDA decoder | 保留 DriveRecon 的 `diff_surfel_rasterization` 2D Gaussian 光栅器 |
| PCC | 相对深度读取、按视角组展平后求 Pearson | 可移植数值计算；需单独适配渲染深度及 all_18/novel_12 汇总 |
| 测试范围 | OmniScene `test` 当前固定截取 mini | 新增显式 mini/total，total 不继承 2,048 个样本上限 |

可以复用数据协议和小型工具函数，但不能把 depthsplat 的 Dataset、Lightning 主程序、decoder 或整个测试函数原封不动接上。后续将相关小型实现放入本仓库，避免运行时通过 `sys.path` 导入整个兄弟项目。

## 3. 配置设计与两档实验

### 3.1 配置文件与加载顺序

已新增以下配置文件：

~~~text
configs/omniscene/
  dataset.py          # 数据根目录、视角顺序、split、数据字段
  model.py            # DriveRecon 默认网络、原生 renderer、静态开关
  runtime.py          # 优化器、迭代节奏、续训、日志、通知、指标
  112x200.py          # _base_ 继承上述三份，仅设置分辨率与工作目录
  224x400.py          # 同上
~~~

新入口用 `mmcv.Config.fromfile()` 加载 `_base_`，再按命令行显式覆盖；优先级为公共配置 < 分辨率实验配置 < 命令行。保存最终 `resolved_config.py`，日志中输出实际模型、图像/高斯分辨率、数据 split、损失权重和恢复步数。

原 `merge_hparams()` 只覆盖 argparse 已有的少数配置组，新的 dataset/evaluation/notification 字段会被忽略，因此新入口直接使用完整 Config，不把新配置塞入旧合并器。

工作目录稳定且按分辨率隔离：

~~~text
work_dirs/omniscene/driverecon_static_t1_112x200/
work_dirs/omniscene/driverecon_static_t1_224x400/
~~~

同一条训练命令重复执行即尝试恢复本目录。新实验通过显式另设 work_dir 开始，不用时间戳自动创建空目录而绕过续训。

### 3.2 固定运行配置

| 项目 | 设置 |
| --- | --- |
| image_shape | 两份配置分别 `[112,200]`、`[224,400]`，顺序 H×W |
| context / target | 6 / 18；所有训练样本使用完整 12 个新视角加 6 个输入视角作为目标 |
| num_frames / num_views | 1 / 6 |
| input_scale | 1；取消旧 loader 的两倍输入放大 |
| batch_size | train、val、mini、total 全部为 1 |
| devices / gradient_accumulation_steps | 默认单 GPU / 1，有效训练 batch size 也为 1；不继承原 2/8/24 进程启动配置 |
| max_steps | 100_001，指实际完成的 optimizer 更新次数 |
| validate_every_steps | 1_000；使用 step 控制，不再叠加 0.01 epoch 触发器 |
| mini_every_n_validations | 10，即正常训练在 10k、20k、…、100k 后评估 mini |
| final_mini_test | True，在完成第 100_001 次更新之后单独执行 |
| checkpoint_every_steps | 5_000；最后一步 100_001 另存；验证本身不触发权重保存，默认每次 mini 前仍有对应步的完整状态 |
| use_dynamic_mask | True，只影响训练/验证损失，不屏蔽正式图像指标 |
| shuffle | train=True；val/mini/total=False |
| precision / parameter_dtype | `precision=bf16`，`model.parameter_dtype=float32`：FP32 参数、梯度累积与 AdamW 动量，前向使用 BF16 autocast；深度采样点、几何变换、softmax、光栅接口及指标保留 FP32 |
| diagnostics.enabled | True；第 1 步、每次验证对应的更新步、最终更新步记录参数更新；每次验证记录全部验证 bin 的深度分布 |
| optimizer | `type='AdamW'`，作为用户确认的稳定性例外保留；UNet 和 adapter 两组均 lr=4e-4、weight_decay=0.05、betas=(0.9,0.999)、eps=1e-15；保留理由是修正反向后恢复 Adam 仍发生辅助深度退化，而非仅因论文如此，见第 16 节 |
| lr_scheduler | constant；原代码虽创建衰减函数，实际循环没有调用，不引入 SVF-GS 的 warmup |
| gradient_clip | 不新增裁剪策略，沿用当前有效训练路径 |
| initialization | 无恢复状态时随机初始化；不加载其他方法的预训练权重 |
| seed / deterministic | seed=0，对齐原入口最后生效的 seed；cuDNN deterministic=True 与原 setup_seed 一致，benchmark=False |
| report_to | 默认本地日志/JSON；如开启 WandB，固定 offline，不执行自动 sync |
| feishu.enabled | True，通过已有 `send_feishu` 模块发送启动和 mini 完成通知 |
| resume | auto，恢复当前 work_dir 中最新完整训练状态 |

设备由入口统一管理，不依赖构造函数中的裸 `.cuda()`。保留第 15 节已经数值验证的原生反向修复，以及用户在第 16 节审查后确认的 FP32 参数/动量和 AdamW 两项稳定性例外。不把论文与代码不一致本身视为 bug，也不因此更改其他模型、损失或学习率调度。

### 3.3 模型与 renderer 配置

以下参数来自当前实际生效的 DriveRecon 构造函数：

| 模块 | 参数与处理 |
| --- | --- |
| UNet | in_channels=3，out_channels=128，layers_per_block=1，skip_scale=sqrt(0.5) |
| 下采样 | channels=(64,128,256,256)，attention=(False,False,False,False)，三次下采样 |
| 中间层 | attention=True；保留带预测几何的 PD_Block 和 TCAttention；num_frames=1、view_num=6 |
| 上采样 | channels=(256,256,128)，前两级启用 PD 注意力，第三级关闭；两次上采样 |
| PD/聚类 | 保留实际调用的 proposal/head/fold 设置；统一传入六相机，不用另一份未调用网络的默认值替代 |
| Gaussian adapter | 输入特征 128；保留 RGB、属性、深度分布、深度残差、位置偏移、分割各头 |
| 深度分类 | 200 类，线性采样点覆盖 0.1–400 米；不是 depthsplat 的 128 类或 0.5–100 范围 |
| 深度残差 | 原 tanh 输出乘 `(400-0.1)/200`，加到分类期望深度上 |
| 尺度 | gaussian_scale_min=0.001、gaussian_scale_max=4；保留原 softplus 变换，4 是变换参数，不声称是硬截断上界 |
| 颜色 | 原 3 通道 RGB 输出，shs_pre=False；不因 argparse 中 sh_degree=3 就误启用 48 通道 SH |
| 分割 | seg_num=3，保留头形状；仅监督静态/动态两种已有类别，不生成天空标签 |
| 位置偏移 | max_shift=5；保留用于静态重建的 uv_shift；means_shift 的运动部分不作用于坐标 |
| 渲染 | 原 2D Gaussian/surfel 光栅器，coarse 路径、黑色背景、RGB 颜色预计算、compute_cov3D_python=False |
| 光栅器反向版本 | `renderer.aabb_backward_version=1`；修正投影中心 VJP，与原前向的 cutoff=3 保持一致，见第 15 节；启动时核对实际加载的原生扩展版本 |

原上采样构造函数的 attention 标志数组有 4 项，而实际只有 3 个 up block；配置需表达实际用到的三项，不能凭配置字面再增加一个模块。原属性头内部的通道复用也先按当前代码保留，不顺带重设计 adapter。

保留模块意味着不能把六张中心图复制成三帧来满足旧 reshape。所有活跃调用链中的 view_num、time_length、num_frames 和分组逻辑必须统一为 V=6、T=1。TCAAttention 仍在这一个时刻的六相机空间 token 上执行注意力；不能因为 T=1 就误判其所有参数无效并冻结。模块仍存在，但不使用邻帧、真实时间差、历史特征或运动补偿。

### 3.4 全分辨率输入与半分辨率高斯

| 实验 | 输入/目标分辨率 | 每个相机高斯网格 | 六相机原始高斯数 |
| --- | --- | --- | ---: |
| 低分辨率 | 112×200 | 56×100 | 33,600 |
| 高分辨率 | 224×400 | 112×200 | 134,400 |

高斯数按原实现每个网格位置一个高斯推导，属于静态配置推算，不是实测可见高斯数。所有高斯合并成一个场景；目标渲染使用完整场景，由原光栅器处理可见性，不沿用 `index % 3` 的三相机条带截取，也不按目标编号选择时间切片。

两档分辨率均可通过 UNet 的三次下采样/两次上采样，但 PD 聚类还有 fold 整除约束。例如低分辨率中间特征为 14×25，合并六相机后 14×150，不能直接通过 fold=4 的旧断言。在聚类内部对特征、几何和有效 token 做一致补齐，计算后还原；补齐 token 不参与中心聚合/损失，不改变真实图像、内参视野和最终高斯数量。保留原 fold 参数，不裁掉输入列，也不把整除失败归因于数据集。

每层几何模块都从加载分辨率的像素 K 缩放到当前特征尺度；最终反投影使用高斯网格 K，目标渲染使用完整图像 K。不得用目标 H/W reshape 半分辨率预测，也不得拿 224×400 的 K 反投影 112×200 网格。

## 4. 数据加载与字段契约

### 4.1 根目录和已有文件

SVF-GS 的 `data/nuScenes` 与 depthsplat 的 `datasets/omniscene` 当前均指向：

~~~text
/home/B_UserData/dongzhipeng/Datasets/dataset_omniscene
~~~

数据集配置将其作为可覆盖的 data_root；无需复制或重新生成数据。实现验证仅按需读取少量 bin，不扫描或审查全量数据。

| 文件/目录 | 使用方式 |
| --- | --- |
| interp_12Hz_trainval/bins_train_3.2m.json | `['bins']` 原顺序作为完整训练池 |
| interp_12Hz_trainval/bins_val_3.2m.json | val、mini、total 的唯一基础列表 |
| interp_12Hz_trainval/bin_infos_3.2m/{bin_token}.pkl | 读取六个相机条目的路径和外参 |
| samples_small / sweeps_small | 已有 224×400 RGB；按实验分辨率在线 resize |
| samples_param_small / sweeps_param_small | JSON 的 camera_intrinsic，随 resize 同步缩放 |
| samples_dptm_small / sweeps_dptm_small | Metric3D `*_dpt.npy`，保持米制，不读取 `*_conf.npy` |
| samples_mask_small / sweeps_mask_small | 已有动态掩码 PNG，不离线重算分割 |
| samples_dpt_small / sweeps_dpt_small | DepthAnything V2 `.npy`，只在 PCC 评估时加载 |

路径替换沿用已有 loader 的 `/datasets/nuScenes → data_root` 规则。不引入 scene_mapping、点云、占据标注或 nuScenes 原始 SDK 数据转换依赖。`sensor2lidar_transform` 只是已有相机外参的坐标参考名称，读取它不等于读取 LiDAR 数据；不访问 `LIDAR_TOP` 点云或用它计算帧数。

### 4.2 split 规则

~~~python
train = bins_train
val   = bins_val[:30000:3000][:10]
mini  = bins_val[0::14][:2048]
total = bins_val
~~~

规则与 SVF-GS 一致；mini 按现有协议最多 2,048 个 bin，total 不切片。实际条数直接由列表长度记录，不扫描文件验证，不硬编码条数断言。训练中 val 和 mini 可能来自相同基础列表，这是参考项目既有协议；不重新拆分。

### 4.3 视角顺序

相机顺序固定为：

~~~text
CAM_FRONT, CAM_FRONT_RIGHT, CAM_FRONT_LEFT,
CAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGHT
~~~

- context：按上述顺序读取各相机 `sensor_info[cam][0]`，共 6 张。
- novel targets：按相机遍历，每个相机依次取 `[1]`、`[2]`，共 12 张，保持 depthsplat/SVF-GS 的交错顺序。
- 最后追加 6 张 context 作为重建目标，共 18 张。
- `novel_12 = target[0:12]`，`input_6 = target[12:18]`，`all_18 = target[0:18]`。

训练、验证、mini、total 均保持这一组装方式。不随机抽取少于 12 个新视角，不把相邻 bin 合成时序输入。

### 4.4 统一 batch 字典

采用 depthsplat 风格的 batch 接口，下面形状已经包含 batch 维，H/W 为加载分辨率：

~~~text
scene/bin_token                     每个 batch 的 bin 标识
context.image                       [1, 6, 3, H, W]，float RGB [0,1]
context.extrinsics                  [1, 6, 4, 4]，OpenCV c2w
context.intrinsics                  [1, 6, 3, 3]，宽高归一化 K
context.intrinsics_pixel            [1, 6, 3, 3]，加载分辨率像素 K
context.metric_depth                [1, 6, H, W]，米，训练/验证按需加载
context.segmentation_label          [1, 6, H, W]，已有掩码派生的静态/动态标签
target.image                        [1, 18, 3, H, W]
target.extrinsics                   [1, 18, 4, 4]
target.intrinsics(_pixel)           [1, 18, 3, 3]
target.loss_mask                    [1, 18, H, W]，float 有效区域权重
target.rel_depth                    [1, 18, H, W]，仅 PCC 评估加载
~~~

数据始终以 RGB 和独立监督字段交付，不预先把动态区域涂黑到 context.image。Metric3D、标签和 DA2 的按需加载开关明确分开，测试不因训练用分割/深度监督而强制读取无关标签。

### 4.5 resize、掩码与相机

RGB 使用与参考 loader 一致的 PIL resize 行为；Metric3D 和相对深度的 resize 使用 bilinear，深度数值不随图像尺寸缩放。K 的第一行乘宽度比，第二行乘高度比；归一化 K 再分别除 W、H。取消 baseline=1、场景尺度归一化、随机 crop 和左右翻转。

掩码有两种用途，必须分开：

1. **target.loss_mask：** 前 12 张读取 PNG，`/255` 得到保留区域权重，双线性缩放后保留 float 边界；动态区域权重为 0。最后 6 张 context 重建目标统一用全 1，完全跟随 SVF-GS。不能直接复用 depthsplat 的 `.bool()`。
2. **context.segmentation_label：** 额外读取已有中心图的真实动态掩码，而不是使用 SVF-GS 为输入返回的全 1 loss mask。按掩码语义映射 static=0、dynamic=1，标签 resize 用 nearest；class=2 不产生天空监督。新视角 loss mask 不参与构造输入语义。

位姿直接使用 OpenCV c2w。SVF-GS loader 的 `c2w @ flip_yz` 是其 OpenGL 接口转换，本项目和 depthsplat 的几何路径不照抄这一步。用 `inverse(c2w)` 构造世界到相机变换，再按原光栅器行向量存储要求转置；投影矩阵显式保留每幅图的 fx、fy、cx、cy。原 Camera 只由 FOV 构造中心主点投影，需新增支持完整 K 的轻量相机适配器，不能把任意 cx、cy 假设成 W/2、H/2。

相机投影面参数沿用 DriveRecon 原接口（znear=0.01、zfar=1e8）；原 CUDA 中 near_n=0.2、far_n=100 的裁剪/畸变辅助计算也应在实现说明中区分，不能把它们误写成数据深度被统一截断到 100 米，更不能直接替换为 depthsplat 的 near/far。

## 5. 主程序、监督与损失

### 5.1 主调用路径

入口 `train_omniscene.py` 提供 train/test 两种 mode；内部流程为：

~~~text
配置加载 → 固定种子/设备 → Dataset/DataLoader → 构建模型与优化器 → 恢复状态
  ├─ train：context → reconstruct() → 一份静态高斯
  │          高斯 + 18 个 target 相机 → 原生 renderer → RGB loss
  │          输入预测深度/分割 + 中心监督 → 辅助 losses → 一次 optimizer.step
  ├─ val：eval/no_grad → 同一损失定义 → 10 个 bin 的平均损失
  └─ mini/total：eval/no_grad → reconstruct() 计时 → RGB/深度渲染 → 两组指标
~~~

`reconstruct(context)` 完成 UNet、adapter、深度解码、uv 偏移反投影、属性激活、六路高斯拼接，返回可立即渲染的最终高斯和训练所需中间预测。不能把仅有 `forward_gaussians()` 输出的 logits 称为“高斯重建完成”。

对旧接口的关键拆分：

- context 图像在模型边界增加 T=1 维；target 始终单独保存，18 不作为输入的 V 或 T。
- 重建函数不接收 target RGB/相对深度；目标位姿只供 renderer 使用。
- 将原函数中需要 GT 深度的辅助 loss 移到训练监督路径。PD 的几何特征仍由网络预测深度生成，推理不依赖 GT 深度或分割标签。
- 原以 `images.size()` 同时推导 target、depth 和 Gaussian grid 的写法改为显式 H/W、Hg/Wg。
- 不对每个测试 bin 优化参数，不在测试中执行 densification、fine deformation 或场景微调。

这能复用 depthsplat “encoder 输出高斯 → decoder 渲染目标”的主流程思想，但不能直接调用其 `ModelWrapper`：本项目的辅助损失、2D Gaussian 参数、相机对象和训练状态均不同。

### 5.2 损失组成

保持当前有效的原项目损失类型/权重，添加用户指定的动态区域屏蔽与静态关闭项：

~~~text
L = L_rgb
  + 1 × L_seg
  + 2 × L_depth_class
  + 2 × L_depth_reg
  + 2 × L_geometry_aux
~~~

| 项目 | 真值、归约与范围 |
| --- | --- |
| L_rgb | 18 张目标 RGB；逐视角 `abs(pred*mask - gt*mask).mean()`，再对 18 路求和，保持 DriveRecon 原 RGB loss 的逐视角求和方式 |
| L_seg | 中心 6 图的真实动态标签；对半分辨率 seg logits 做原 cross entropy，权重 1 |
| L_depth_class | 中心 6 图的 Metric3D 米制深度，resize 到高斯网格；按 200 个采样点映射类别后计算 cross entropy，权重 2 |
| L_depth_reg | 分类期望深度加残差，与中心 Metric3D 深度做原 `compute_depth('l2', ...)`，权重 2 |
| L_geometry_aux | 中间几何 PD_Block 的预测深度监督；保留原归一化、内部缩放和外部权重 2 |
| 跨时间 static/dynamic | 均关闭，不访问 T±1、不使用 means_shift；原 0.1 系数在静态配置中为 0 |
| sky / normal | 保持原有效权重 0，直接不计算；不加载天空 mask |
| SSIM / LPIPS 训练项 | 不增加；原入口虽有 lambda_dssim 参数，当前 Gaussian_LRM 的有效 RGB 训练项为 L1 |

动态区域屏蔽对齐的是 SVF-GS 的**掩码使用方式**：乘到预测与真值，再对完整图像求均值，不除以有效像素数；RGB 的 L1 类型和逐视角求和则遵循 DriveRecon。图像维度不变，不额外加 SVF-GS 的其他网络/损失。

中心 6 图的重建 loss 保持全图，分割 loss 包含动态类别监督，深度辅助项使用中心深度；不能把新视角 loss mask 广播过去而抹掉动态分割监督。也不额外增加 18 路 Metric3D 渲染深度损失或深度置信度加权。

### 5.3 深度单位与分类修正

数据层仅交付米制深度 `d_m`，不把 RGB 的 `/255` 用到文件值上。旧函数内部如必须维持 normalized depth 接口，由明确命名的适配层产生 `d_native = d_m / 255`，不在多个层级重复除乘。

分类中心及目标定义为：

~~~python
centers = linspace(0.1, 400.0, 200)
class_target = round((d_m - 0.1) / (400.0 - 0.1) * 199).clamp(0, 199).long()
depth_pred = (softmax(depth_logits) * centers).sum(-1) + depth_residual
~~~

该修正已获用户确认：不再把米数直接 `.long()` 当作 200 类索引。端点 clamp 是分类表示范围处理，不因此剔除样本或生成坏数据报告。

原 L2 不是未经归一化的米制 MSE：`compute_depth` 对 0.01–255 范围计算归一化深度 MSE；主深度监督外层还使用 d_m>0.1。中间 PD 辅助项原先对 d_native>0.1（即约 25.5 米）选像素，并在 `compute_depth` 后额外除 255。本次按当前代码保留并显式记录这些条件，避免以“读取对齐”为名悄悄改变损失尺度。深度分类范围 0.1–400、L2 的 255 和 PD 辅助范围是三个不同概念。

若定义内的像素集合为空，该 loss 返回与图连接的零标量，其他 loss 继续计算；不丢弃样本、不拒绝整个 bin。这是原代码忽略空/NaN 辅助项意图的明确化，不新增数据集质量判断。

## 6. 训练、验证、mini 测试与续训

### 6.1 迭代顺序

global_step 表示“已经完成的 optimizer 更新次数”，从 0 开始，训练至 100_001。每次取一个训练 batch 做一次更新；不保留原代码同一 batch 连续更新 500 次的循环。

~~~python
while global_step < 100_001:
    batch = next_training_batch()  # 遍历完 train 后进入下一轮
    train_one_optimizer_update(batch)
    global_step += 1

    if global_step % 5_000 == 0:
        save_training_state()

    if global_step % 1_000 == 0:
        validate(val_loader)
        validation_count += 1
        persist_events_only_if_checkpoint_matches_current_step()
        if validation_count % 10 == 0:
            evaluate_mini(reason='periodic')
            persist_evaluation_state_and_notify()

save_training_state()              # step=100_001
evaluate_mini(reason='final')       # 必须执行，不能用 step=100_000 的结果替代
persist_evaluation_state_and_notify()
mark_training_and_final_mini_complete()
~~~

正常完成时有 100 次周期验证、10 次周期 mini，再加最终第 100_001 步的 1 次 mini。原项目没有完整的训练中 mini 功能；本次实现仍补齐，以满足用户要求的最终 mini，并统一周期评估路径。

完整权重只在 5k、10k、…、100k 和 100,001 保存，共 21 份。1k、2k 等验证只写轻量 JSON/日志，不复制权重。若手动配置使 mini 步数不落在保存步上，mini 前补存同一步的状态以保证恢复和评估来源正确；默认配置不会额外增加权重份数。

验证聚合全部 10 个 bin，不只保留最后一个 batch。mini/total 复用同一个 evaluator；评估前切 eval/no_grad，结束恢复 train。评估和可视化消耗的随机状态不改变训练采样序列。

### 6.2 自动恢复

在本实验自己的 work_dir 中查找数值步数最大的完整 checkpoint；优先使用可用的 latest，若 latest 未更新则扫描 checkpoint 步号。仅检查训练状态文件是否完整，不检查数据集。保存采用临时目录写完后原子发布，避免把半写入 checkpoint 当作可恢复状态。

恢复内容包括：

- UNet、adapter 及所有保留模块的参数/buffer；
- 优化器类型、动量、学习率状态、混合精度状态（如实际使用）；
- global_step、epoch、epoch 内 batch 游标及 sampler/generator 状态；
- Python、NumPy、Torch CPU/CUDA RNG；
- validation_count、最近完成的 val/mini step、待完成评估事件、final_mini 完成标记。

只加载模型权重不能算自动续训。原 `Gaussian_LRM.capture/restore` 的不完整状态路径不作为新协议的恢复来源，也不自动混用另一个分辨率的 checkpoint。

恢复后先完成已经到达触发点但尚未完成的验证/mini，再继续训练，避免在 10k 保存后中断就漏掉 mini。若已训练到 100_001、但 final_mini 未完成，只补最终 mini；若训练和最终 mini 都完成，直接报告已完成，不再多训一步。中途被打断的评估可以从该 split 开头重跑，并覆盖同一事件的临时结果，避免把两次半程统计相加。

事件 journal 只能更新当前模型步数对应的 checkpoint。例如模型已到 7k、最新权重仍是 5k 时，6k/7k 的验证结果不能写入 5k 的 `events.json`。中断后从 5k 恢复并重算后续更新/验证；对应步的验证 JSON 覆盖，追加日志可能包含重放记录。首次 5k 保存前中断则从头开始。旧 BF16 参数实验、第二轮 FP32+Adam 实验均不能直接恢复到新 FP32+AdamW 配置；配置比较会拒绝混用。新检查点还记录实际 `optimizer_type`，加载时在修改模型权重之前检查类型，避免 Adam 与 AdamW 相似的 state_dict 格式掩盖错误。现有目录本次不移动、不删除，由用户在新训练启动前自行删除，或另指定空 work_dir。

第 15 节修复后，恢复检查同时比较 `renderer` 配置，禁止把旧光栅器梯度积累的训练状态继续作为新版本正式训练。即使清空优化器，旧权重也仍包含旧梯度历史；正式实验应从头开始。旧权重可用于明确标注来源的诊断和前向一致性检查。入口在 `train_rasterizer.json` / `test_rasterizer.json` 记录实际 `.so` 路径、反向版本和编译时源文件 SHA256，并写入启动日志。

### 6.3 本地日志与飞书

保存训练各 loss、lr、step、验证结果、两组 mini 指标和计时；WandB 如启用，设置 `mode=offline`/`WANDB_MODE=offline`，不使用网络作为 checkpoint 或运行目录管理前提。

飞书沿用 SVF-GS 的调用方式：

~~~python
from auto_monitor.send_feishu import send_feishu
send_feishu(title, body)
~~~

本机可见模块位于 `~/Libraries/auto_monitor/send_feishu/`；在配置中提供模块搜索根目录 `~/Libraries`，通过已有模块配置读取 webhook，不把凭证写进实验配置、结果或仓库。

- 启动通知：实验名、分辨率、work_dir、随机初始化/恢复来源、当前 step/100001、设备、参数统计。
- 每次 mini 完成通知：periodic/final、step、bin 数、all_18 与 novel_12 四项指标、完整重建耗时、评估耗时和训练 ETA。
- 本地先落盘再发通知；发送失败记录结果但不终止训练，可用轻量异步发送避免网络波动阻塞更新。模块当前有 5 秒请求 timeout 和失败返回值，应保留该有界行为。
- 默认单进程仅发送一次；未来若扩展多进程，只由主进程发送。

通知已接入实际训练入口；本次验证全部关闭飞书，没有发送实际消息。

## 7. PCC 与两组质量指标

### 7.1 DepthAnything V2 相对深度

只在 mini/total 的 Dataset 开启 `load_rel_depth`，一次加载全部 18 张目标对应的已有 DA2 文件。参考 SVF-GS/depthsplat 的转换保持一致：

~~~python
disp = load_existing_da2_npy().astype(float32)
disp = bilinear_resize_if_needed(disp, (H, W))
ratio = min(disp.max() / (disp.min() + 0.001), 50.0)
disp_floor = disp.max() / ratio
relative_depth = 1.0 / maximum(disp, disp_floor)
relative_depth = (relative_depth - relative_depth.min()) / (
    relative_depth.max() - relative_depth.min()
)
~~~

这里读到的是参考实现按 disparity 解释的文件，再转换为 relative depth；不能把原始文件直接与渲染 z 作相关。PCC 不用 Metric3D 替代 DA2，不拟合尺度/偏移，也不将 DA2 送入模型。

### 7.2 渲染深度：保持 depthsplat 的累积中心 z 口径

depthsplat 的 `render_depth_cuda(mode='depth')` 先计算每个高斯中心在目标相机中的 z，再把 z 当作颜色，使用黑背景进行 alpha 合成：

~~~text
D(p) = Σ_i [T_i(p) α_i(p) z_center,i]
~~~

它没有除以累计 alpha。DriveRecon 当前 `maps2all()` 返回的 surf_depth 默认是 `allmap[0]/alpha`，不能直接用来对齐。

进一步核对 CUDA 可见：DriveRecon 原 `allmap[0]` 累积的是 surfel 与像素射线交点深度，未必等于高斯中心 z；因此**仅去掉除 alpha 也不保证与 depthsplat 的定义相同**。

已实现专用 `render_pcc_depth()`：

1. 从最终静态高斯中心和目标 w2c 计算中心 z。
2. 以 `[z,z,z]` 作为 `colors_precomp`，保留本项目原生 surfel 光栅器、几何参数、透明度和相机，黑背景单独渲染。
3. 返回单通道累积深度；不经过 RGB 路径的 `clamp(0,1)`，不除 alpha，不用 median/expected depth 替换。
4. 原生 surfel 的空间核与 depthsplat 的 3D Gaussian 核仍属于方法差异；对齐的是深度数值和累积定义，不能声称两个 renderer 数值完全相同。

若保留原 surf_depth/allmap 做可视化或调试，另命名为 expected_surface_z/accumulated_surface_z，正式 PCC 固定使用 `accumulated_center_z`。PCC 深度渲染发生在重建计时结束之后。

### 7.3 计算位置与聚合

单个 bin 的计算步骤：

~~~python
groups = {'all_18': slice(0, 18), 'novel_12': slice(0, 12)}
for group, ids in groups.items():
    psnr = compute_psnr(gt_rgb[ids], pred_rgb[ids]).mean()
    ssim = compute_ssim(gt_rgb[ids], pred_rgb[ids]).mean()
    lpips = compute_lpips(gt_rgb[ids], pred_rgb[ids]).mean()
    pcc = compute_pcc(gt_rel_depth[ids], rendered_depth[ids])
    append_bin_record(bin_token, group, psnr, ssim, lpips, pcc)
~~~

- PSNR：RGB 范围 [0,1]，每视角先计算再平均，不能把 18 图 MSE 混合后只计算一次 PSNR。
- SSIM：沿用参考实现的 skimage，win_size=11、gaussian_weights=True、data_range=1、channel_axis=0。
- LPIPS：沿用 VGG、normalize=True；评估模型 eval/no_grad，权重本地就绪，训练过程中不依赖临时在线下载。
- PCC：沿用参考 `PearsonCorrCoef` 的输入展平方式；单个 bin 的整组视角与像素一起展平，分别求 all_18 和 novel_12 的相关系数，不改成“每图 PCC 再平均”。
- 对 split 内所有 bin 的每项结果做等权算术平均；不要把全测试集像素拼到一起只求一次 Pearson。
- 每个分组/评估事件有独立统计；如果使用有状态 TorchMetrics 对象，明确 reset 或使用相同公式的无状态计算，不能累积混入上一组/上次测试。
- 训练动态掩码不用于这些正式全图指标，不导入 DDAD ego mask、天空排除或透明度阈值筛选。
- 不因分数、深度范围或相关性结果异常而删除 bin；不静默把未定义值改成 0 或从均值中忽略。记录原始结果，不把问题归因于数据审查后终止整个程序。

实现复用了 depthsplat 的 PCC 数值定义、相对深度转换和“逐样本记录后平均”的方法，并补齐 OmniScene 的双分组、统一 mini/total 调用，以及 DriveRecon 的专用深度渲染。当前 depthsplat 的 PandaSet/DDAD 扩展不能整套复制，因为带有本任务不需要的数据检查与掩码协议。

### 7.4 结果文件

~~~text
work_dir/
  resolved_config.py
  checkpoints/step-00005000/...
  latest.json
  final.json
  validation/step-00001000/...
  diagnostics/step-00001000/update.json
  evaluation/mini/step-00010000/periodic/
  evaluation/mini/step-00100001/final/
  evaluation/total/step-00100001/
    per_bin_metrics.csv
    evaluation_summary.json
    parameter_counts.json
    reconstruction_timing.json
~~~

CSV 每个 bin 至少两行，对应 all_18/novel_12，包含 bin_token、group、四个指标；汇总 JSON 保存 split、checkpoint/global_step、分辨率、实际完成 bin 数、两组四指标、深度定义、参数统计及计时摘要。运行未结束的文件标记为 partial，不冒充完整 total 结果；是否完成按本轮遍历是否结束判断，不扫描数据质量。

## 8. 参数量与完整重建耗时

### 8.1 参数量

以本次实际实例化的重建模型为统计对象，用 Parameter 对象去重：

~~~python
trainable = sum(p.numel() for p in unique_model_parameters if p.requires_grad)
frozen = sum(p.numel() for p in unique_model_parameters if not p.requires_grad)
total = trainable + frozen
~~~

保留的 TCA、运动输出、分割头都计入模型，不能为了静态实验的数字好看而漏计。`requires_grad` 与是否在当前样本获得非零梯度不同；特别是原 uv/motion 输出共享一个头，不能把其中未使用的几行当成独立 frozen Parameter。按实际状态报告，不预写冻结参数为零。

另给 UNet、adapter、辅助头等分项便于解释；指标 LPIPS 不计入重建模型，DA2/Metric3D 只读取离线图也不计入。若后续模型前向实际加入冻结网络，则必须纳入冻结参数和耗时，不按“冻结”排除。

### 8.2 计时边界

主要指标 `reconstruction_ms` 与 SVF-GS 的完整 forward 边界对齐：输入 batch 已在设备上并完成数据整理后，在调用 `reconstruct(context)` 前同步 GPU 并开始计时；最终高斯的世界坐标、颜色、尺度、旋转、透明度完成激活与拼接后，同步 GPU 并停止。

包含 UNet/PD/TCA、adapter、预测深度解码、所有必需在线几何计算和最终高斯组装；若重建依赖内部渲染，则也包含内部渲染。DriveRecon 当前静态重建预计不需要目标视角内部渲染。

不计 DataLoader/磁盘读取、离线深度生成、GT 准备、CPU→GPU 传输、18 路最终展示/评估渲染、PCC/LPIPS、文件写入和飞书网络请求。这一边界是“张量送入网络到高斯完成”，不是整条数据 I/O 流水线。可另记 H2D 或总评估时间，但不与 reconstruction_ms 混报。

默认单 GPU、batch=1、eval/no_grad，同步计时；前 5 个 bin 仅不参与耗时统计，仍完整参与质量指标。记录后续每个 bin 的耗时并报告平均值、中位数、P95、计时数量，同时记录 GPU、精度、分辨率、checkpoint。不能只报 UNet 时间、CUDA 异步提交时间或按 18 个目标除出来的单张渲染时间。

实际模型统计为：可训练参数 58,249,264、冻结参数 0、总参数 58,249,264，两档分辨率相同。已完成的 112×200 FP32+AdamW total 重建平均耗时为 19.287 ms/bin（RTX 4090）；这是当前配置的实测，不能替代 224×400 或其他配置的正式训练/评估记录。第 12 节的显存数字仍仅为早期短测。

## 9. 调用命令

以下命令已实现。环境使用 README 的 drivingrecon 环境；已有环境需补充 `pip install lpips==0.1.4`。LPIPS 的 VGG16 权重须事先在本地准备好，具体见第 12 节。

~~~bash
conda activate drivingrecon

# 光栅器反向修复后须重新编译；启动时会检查原生二进制版本。
pip install -e submodules/diff-surfel-rasterization --no-build-isolation

# 两档实验分别训练；重复同一命令时自动恢复本 work_dir。
CUDA_VISIBLE_DEVICES=0 python train_omniscene.py --config configs/omniscene/112x200.py --mode train
CUDA_VISIBLE_DEVICES=0 python train_omniscene.py --config configs/omniscene/224x400.py --mode train

# 正式 total 评估：默认读取相应实验的最终训练 checkpoint。
CUDA_VISIBLE_DEVICES=0 python train_omniscene.py --config configs/omniscene/112x200.py --mode test --test-split total --checkpoint final
CUDA_VISIBLE_DEVICES=0 python train_omniscene.py --config configs/omniscene/224x400.py --mode test --test-split total --checkpoint final
~~~

当前默认配置与已完成的 AdamW 实验一致。若需固定使用该次训练实际保存的配置，也可用：

~~~bash
CUDA_VISIBLE_DEVICES=0 python train_omniscene.py --config work_dirs/omniscene/driverecon_static_t1_112x200/resolved_config.py --mode test --test-split total --checkpoint final
~~~

`final` 在训练结束保存时明确指向 step=100_001，而不是按 mini 最佳分数自动挑模型。手动评估其他 checkpoint 需显式传路径并在结果中记录步数。两档正式结果分别报告，不以低分辨率训练后只调高测试分辨率替代高分辨率实验。

## 10. 实现文件与验证安排

### 10.1 文件职责

| 新增/修改 | 职责 |
| --- | --- |
| configs/omniscene/*.py | 数据、模型、运行节奏及两档实验配置 |
| train_omniscene.py | 入口、配置覆盖、设备初始化、train/test 分派 |
| comp_svfgs/dataset_omniscene.py | 移植固定 split、6→18 视角和 batch 字典 |
| comp_svfgs/omniscene_io.py | RGB/K/Metric3D/动态掩码/DA2 按需读取，不含数据审查 |
| comp_svfgs/model.py、camera.py | 静态 reconstruction 接口、各尺度 K 与原生 renderer 相机适配 |
| scene/PointNet.py、scene/PD_Block.py 及实际调用的注意力模块 | 参数化 V/T，内部特征整除适配，分离预测与监督 |
| scene/GS_LRM.py | 抽取可复用的重建/渲染逻辑；以兼容原入口的方式支持新适配 |
| comp_svfgs/trainer.py、checkpoint.py | 精确步数、val/mini/final、完整自动恢复 |
| comp_svfgs/optimizer.py | 显式构造用户确认保留的 AdamW；仍支持原版 Adam 供历史实验与诊断复现，不静默转换优化器身份 |
| comp_svfgs/diagnostics.py | 只读记录深度分布、精度、优化器类型、参数更新、权重幅值及归一化缩放参数，不修改网络、损失或样本 |
| comp_svfgs/evaluation.py、metrics.py | PCC 深度、四指标、双分组和结果持久化 |
| comp_svfgs/notifications.py | 加载已有 send_feishu，组织通知，网络失败不干扰训练 |
| .gitignore | 实现时补充 work_dirs、必要数据软链接/运行产物忽略规则；已补充相应忽略规则 |

上述拆分是实现目标，不要求复制完整兄弟工程或重构全部 Waymo 代码。实现阶段只按功能需要补齐依赖，保持现有模型与 CUDA 扩展。

### 10.2 验证顺序

1. **配置与 CPU 合成数据验证：** 两档配置展开、固定相机/target 顺序、float loss mask/语义标签分离、米制深度到类别映射、K 的多尺度转换和投影往返。
2. **模型结构验证：** T=1/V=6，无目标 RGB 泄漏，无历史缓存；内部 padding 还原后高斯数量和分辨率正确。用合成样例验证完整 K，不能以真实数据位姿审查替代模型测试。
3. **指标验证：** 与参考 PSNR/SSIM/LPIPS/PCC 函数输入相同张量时一致；组内展平/跨 bin 平均正确；专用深度路径没有 RGB clamp、alpha 归一化或 surfel surface-depth 偷换。
4. **恢复/节奏验证：** 用极短合成训练模拟 checkpoint 后、验证前、mini 中途、最后一步之后的中断；恢复优化器、采样/RNG 和待办评估事件，最终正好完成规定更新数及最终 mini。
5. **后续获准运行后的 GPU 检查：** 两档各进行少量前向、反向、保存恢复和评估，确认原生扩展、显存与计时；通过后再正式训练。若 224×400 需要省显存，优先分块渲染并累加同一次更新的视角损失，或采用激活检查点机制；仍是一个 bin 对应一次 optimizer 更新，不静默增加有效 batch、降低分辨率、减少视角或改变损失权重。

以上检查针对代码与协议，不是对 OmniScene 做质量审查。初次实现的短程验证见第 12 节，两次后续修复分别见第 13、14 节。

## 11. 本次核对的代码依据

- DriveRecon：[train.py](../train.py)、[arguments/__init__.py](../arguments/__init__.py)、[arguments/nvs.py](../arguments/nvs.py)、[配置合并](../utils/params_utils.py)。
- DriveRecon 模型：[GS_LRM.py](../scene/GS_LRM.py)、[实际 UNet](../scene/PointNet.py)、[PD_Block.py](../scene/PD_Block.py)、[Camera](../scene/cameras.py)、[损失函数](../utils/loss_utils.py)。
- DriveRecon 深度：[surfel CUDA forward](../submodules/diff-surfel-rasterization/cuda_rasterizer/forward.cu)、[CUDA 常量](../submodules/diff-surfel-rasterization/cuda_rasterizer/auxiliary.h)。
- SVF-GS：[OmniScene Dataset](../../SVF-GS/data/omniscene_dataset.py)、[数据读取](../../SVF-GS/data/transforms/loading.py)、[主配置](../../SVF-GS/configs/main.yaml)、[损失与测试](../../SVF-GS/model/omni_gs.py)、[训练/通知](../../SVF-GS/trainer.py)、[指标](../../SVF-GS/tools/metrics.py)。
- depthsplat：[OmniScene Dataset](../../depthsplat/src/dataset/dataset_omniscene.py)、[数据读取](../../depthsplat/src/dataset/utils_omniscene.py)、[112×200 配置](../../depthsplat/config/experiment/omniscene_112x200.yaml)、[224×400 配置](../../depthsplat/config/experiment/omniscene_224x400.yaml)。
- depthsplat：[主入口](../../depthsplat/src/main.py)、[ModelWrapper](../../depthsplat/src/model/model_wrapper.py)、[指标](../../depthsplat/src/evaluation/metrics.py)、[深度光栅化](../../depthsplat/src/model/decoder/cuda_splatting.py)、[patch shim](../../depthsplat/src/dataset/shims/patch_shim.py)。

以上以当前可执行代码为依据。兄弟项目旧实验文档中的文件名、PCC 实现状态和裁剪描述可能已经滞后，本方案不直接把旧文档当作当前行为。


## 12. 实现与验证记录（2026-09-26）

本节是首轮正式训练之前的历史记录，BF16 权重、依赖状态及 CUDA 显存数字仅描述当时的版本，不代表本次 FP32 参数版本已经完成 GPU 验证。

### 12.1 已实现内容与具体做法

- 两档配置、静态六相机数据/模型接口、原生 2D Gaussian 渲染、Metric3D 分类/L2 和分割监督、SVF-GS 风格的新视角 RGB 掩码均已接入。
- `train_omniscene.py` 提供配置展开、train/test、data_root/work_dir 覆盖和显式 mini/total；训练器按完成的 optimizer 更新数调度 val、周期 mini 和最终 mini。
- 完整 checkpoint 包含网络、Adam、精度状态、Python/NumPy/Torch/CUDA RNG、采样排列、epoch/游标和 DataLoader generator。评估完成状态单独原子记录到 checkpoint 的 `events.json`，恢复时先补未完成事件。
- 同目录自动恢复通过扫描最新完整 step 完成，不信任过期的 latest 指针；实际指针文件为 `latest.json`、`final.json`。不同分辨率、模型、损失、优化器、精度和训练调度配置不混用。相关约束检查的是实验配置/训练状态，不审查数据集。
- 公用的原 adapter 已抽出到 `scene/gaussian_adapter.py`，原 `GS_LRM.py` 复用该类；`scene/__init__.py` 延迟加载 Waymo/CUDA 特有依赖，使新的网络与 CPU 测试不需要加载原数据预处理栈。
- 静态分支的注意力使用 PyTorch SDPA，保持原权重和注意力计算定义，避免无 xFormers 时显式构造巨大的注意力矩阵；旧分支默认行为保留。
- 18 路 RGB loss 采用“逐视角累计高斯参数梯度，再对网络反传一次”的等价实现，避免同时保留 18 份光栅器反向缓冲。仍是一个 bin、一次 optimizer 更新、18 路 L1 求和，已经与直接求和反传做梯度一致性测试。
- PCC 专用渲染输出累计中心 z，未经过 RGB clamp、alpha 归一化或 surfel 交点深度替换。输出包括两组指标、逐 bin CSV、汇总 JSON、参数量、精度与重建耗时。
- 初次实现沿用原代码的 BF16 参数，使用 `model.parameter_dtype='bfloat16'`；本次已按第 13 节修订为 FP32 参数，保留 BF16 autocast。

### 12.2 已完成验证

| 验证 | 范围与结果 |
| --- | --- |
| CPU 测试 | `python -B -m unittest tests.test_omniscene -v`，14 项通过；使用合成资产，不检查真实数据质量 |
| 数据与几何 | 6→18 顺序、mini/total/val 选择、米制单位、float 掩码/分割标签分离、完整 K 投影往返、深度类别端点通过 |
| 静态网络 | 原网络权重结构保留，T=1/V=6，非整除特征内部补齐，半分辨率输出；改变监督字段不影响重建输出 |
| 指标 | 两组归约、无状态组内 PCC、PSNR 逐图平均、原生 CUDA 累积中心 z、离线加载 VGG 与原 LPIPS 输出一致 |
| 中断恢复 | 验证前、周期 mini 前、最终 mini 前和跨 epoch 中断后，优化器、采样/RNG 与待办评估恢复；完成后再次启动不多训、不重测 |
| RTX 4090 / 112×200 | 1 个真实 train bin、2 次 Adam 更新、checkpoint 重载、验证和 1 个 mini bin 四指标通过；最终 BF16 配置短测峰值 allocated 约 2,119 MiB |
| RTX 4090 / 224×400 | 同样范围通过；最终 BF16 配置短测峰值 allocated 约 7,158 MiB |
| 命令入口 | 合成数据 1 次更新，验证→周期 mini→最终 mini→重复启动→total 读取完整 2 个 bin 的路径通过 |

单 bin 的显存、耗时和随机初始化后分数只用于验证功能，不作为正式 OmniScene 结果或完整数据集显存保证。没有运行 100,001 步训练、2,048 bin mini 或全量 total。验证时飞书关闭，未发送消息。

本地验证产物位于（均由 `.gitignore` 忽略）：

~~~text
work_dirs/verification_final_20260926/verification.json
work_dirs/verification_final_20260926/{112x200,224x400}/verification.json
work_dirs/verification_cli_final_20260926/run/
~~~

验证入口分别是 `tests/smoke_omniscene_cuda.py`（两档各两步、各一个真实 mini bin）和 `tests/verify_omniscene_cli.py`（仅合成数据）。它们不会发送飞书，也不会修改正式实验 work_dir。

### 12.3 依赖与验证边界

- 新增 `lpips==0.1.4`，已写入 requirements.txt。现有 drivingrecon 环境缺少该包；本轮未修改任何 Conda 环境，验证临时使用 `/tmp/driverecon-verification-deps` 中隔离的已安装包副本。正式使用前执行 README 中的 `conda activate drivingrecon`、`pip install lpips==0.1.4`。
- VGG16 权重默认读取 `~/.cache/torch/hub/checkpoints/vgg16-397923af.pth`，也可设置 `evaluation.lpips_vgg_weights`。当前机器已有该文件。训练和评估不会触发权重下载；其他机器需要在启动前显式准备，README 给出了准备命令。
- 原 Waymo 的 `train.py --help` 在当前环境被缺少的 `simple_knn` 阻断，因此没有宣称旧入口完成了运行回归。共享网络的原默认配置和权重结构已保留；新 OmniScene 路径不依赖该模块，原生 surfel CUDA 渲染已实际验证。
- .gitattributes 只让 Git 的空白检查正确识别原 source/README 的 CRLF，不更改运行逻辑；验证产生的原仓库已跟踪 pyc 已恢复，未留下二进制缓存变更。

## 13. 首轮退化后的最小修复（2026-09-26）

本节记录第一次修复的决定与验证。第 14 节曾尝试 AdamW；当前默认选择及必要例外以第 16 节审查后的用户决定为准。

### 13.1 依据与边界

首轮 112×200 完成 100,001 步及最终 2,048-bin mini：all_18 PSNR 约 17.9994、PCC 约 -0.2205。该结果仅为 mini；现有训练产物全部保留，本次不重算指标、不改写检查点。

CPU 诊断中，三个验证样本的几何辅助深度全部饱和至 255 米，主深度分类 argmax 集中于同一类别；主深度采用概率期望加残差，因此不能称为严格常数图。几何辅助验证损失从 8k 至 100k 连续 93 次完全相同。参数和全部 Adam 动量均为 BF16，简单 CPU 算术也证明该精度下小更新和 `0.999` 衰减可能被舍入吞掉；这支持精度修复，但尚未证明它是退化的唯一原因。

本次只把两档实验的参数、梯度累积及 Adam 动量改为 FP32，保留 BF16 autocast。优化器仍为 Adam，lr=4e-4、weight_decay=0.05、betas/eps、所有损失及权重、深度分类/激活、静态六相机协议均保持原定义。精度改变不会增加模型参数个数，但会增加参数与优化器状态的存储量；新版本的实际显存和速度须以后重新测量。

原 `PD_Block.py` 中 `ones_like` 后再 scatter 1 使 `split_mask` 恒为零，作者公开代码同样如此。论文描述阈值分流，无法直接推导一个确定的替代实现；本次不改为 zeros、不自行选择阈值。论文的 AdamW 和损失配方与公开代码也有差异，本次继续以已确认的代码配置为准，不拼接不同配方、不搜索更高分数的设置。

参考：[PyTorch 2.3 autocast 使用指导](https://docs.pytorch.org/docs/2.3/amp.html)、[作者 PD-Block 源码](https://github.com/EnVision-Research/DriveRecon/blob/main/scene/PD_Block.py)、[论文 §3.2、§4.2](https://proceedings.neurips.cc/paper_files/paper/2025/file/f5717c76feff4f751604c0678c46627b-Paper-Conference.pdf)。已有 CPU 诊断位于忽略目录 `work_dirs/diagnostics/depth_cpu_20260926/` 和 `work_dirs/diagnostics/reproduction_audit_20260926/`。

### 13.2 训练状态诊断

- `diagnostics.enabled=True`；第 1 步、每次验证对应的训练更新步、最终更新步记录 `diagnostics/step-XXXXXXXX/update.json` 和 `metrics.jsonl` 的 `health` 记录。
- 记录全部模型参数、已有梯度及 Adam 一阶/二阶动量的 dtype；对输入卷积、深度分类头、深度残差头记录该次更新的参数变化比例、最大变化、梯度范数和非有限梯度数量。只克隆这些小模块的更新前参数，不保留额外完整模型副本。
- 每次验证的 `summary.json` 增加 `diagnostics`，保留每个 bin 的主深度均值、逐相机空间标准差、非有限值比例、分类 argmax 占比与类别数，以及每个几何辅助深度的均值、空间标准差和 0/255 边界饱和比例；汇总为逐 bin 等权平均，不能把类别数等均值当作整个验证集的唯一类别数。
- 诊断不参与损失、不修改梯度、不消耗随机数、不筛除/替换样本，不以某个 PSNR/PCC 阈值自动调整配置或中断训练。非有限统计如实记录。

### 13.3 下次运行的检查点与本次验证范围

本次仅代码开发和 CPU 合成测试，不调用 GPU、不启动正式训练，也不删除已有结果。下次训练前由用户删除旧实验目录，或使用另一个空目录；不把旧 BF16 权重或 Adam 状态作为恢复来源。新配置仍拒绝与旧模型配置混用。

后续先运行 112×200，固定 seed=0 和 100,001 步计划；在 10k 的第一次 mini 后结合诊断日志人工检查，确认参数实际更新、深度不再整体饱和并存在有效学习，再继续同一轮。10k 是检查时点，不是新增的自动提前停止器。若同样退化再次出现，应定位首个异常环节或获取作者实现依据，不自动尝试其他学习率、权重、种子或额外训练轮次。112×200 的正常性确认后再安排 224×400。

本次 CPU 验证已完成：`tests.test_omniscene` 和 `tests.test_omniscene_repair` 共 21 项通过。验证时设置 `CUDA_VISIBLE_DEVICES=''`，屏蔽 CUDA 可用性检查，并将 CUDA 初始化/同步/设备查询替换为报错函数；未运行 GPU smoke 或真实训练入口。

新增回归覆盖两档 FP32 参数/BF16 autocast 配置、实际 CPU autocast 下 Adam 动量精度、诊断不改变梯度/随机状态、5k 保存与 1k 验证独立（以 1:5:10 的小步数合成流程验证）、首次保存前及保存间隔中断后的事件回放、mini/final 中断恢复、旧精度检查点拒绝混用。CPU 通过不代表新精度版本已经在 GPU 上验证或收敛；原生 GPU 扩展和正式训练本次未运行。

## 14. 第二轮退化：权重衰减语义修复（2026-09-27）

> 历史排障记录。第 16 节补充了修正光栅器后恢复 Adam 的诊断和用户保留 AdamW 的决定；论文使用 AdamW 本身不能证明公开代码的 Adam 写错。


### 14.1 现象与 CPU 定位

第二轮采用 FP32 参数/动量、BF16 autocast，但仍沿用公开代码的 Adam。相同 2,048-bin mini 的 all_18 PSNR 从 10k 的 20.924 降至 30k 的 19.859，PCC 从 0.740 降至 0.687。13k 辅助几何深度在全部 10 个验证 bin 上成为常量，14k 主深度分类单一类别占比达到 99.94%，之后反复恢复和失稳。用户已在约 37k 停止训练。

CPU 读取 5k、10k、15k、20k、30k、35k 的现有检查点，发现多处深层卷积与归一化缩放参数被压至极小值。例如 35k：

| 参数 | 平均绝对值 |
| --- | ---: |
| unet.down_blocks.3.nets.0.norm1.weight | 1.85e-35 |
| unet.mid_block.nets.0.norm2.weight | 3.20e-14 |
| adapter.depth_regression_head.2.weight | 3.55e-12 |

使用已保存的同一个六相机样本，只在 CPU 执行编码器至辅助深度输入的部分前向：5k 的几何深度空间标准差约 6.27 米；35k 使用 FP32 算术也只剩约 1.07e-5 米，使用 CPU BF16 autocast 则为 0。这表明输入相关信号在权重/特征层面已经很弱，单纯把退化检查点的前向切为 FP32 无法恢复学习到的几何；不能把此次常量图仅解释为 BF16 tanh 的 255 米饱和。CPU 前向不等于原生 GPU 渲染或正式指标评估。

### 14.2 可复查的机制及修复依据

当前 PyTorch 2.3 的 Adam 把 `weight_decay * parameter` 加进任务梯度后计算自适应动量；AdamW 则独立执行 `parameter *= 1 - lr * weight_decay`。相同的 `weight_decay=0.05` 并不表示相同的衰减行为。

保持 lr=4e-4、weight_decay=0.05、betas=(0.9,0.999)、eps=1e-15，用 FP32 标量和显式零任务梯度做 CPU 对照，10,000 次更新后：

| 初始参数 | Adam | AdamW |
| --- | ---: | ---: |
| 1.0 | 约 -3.18e-40 | 0.818507 |
| 0.02 | 约 2.16e-41 | 0.016370 |

这证明耦合 L2 在弱任务梯度分支上足以造成观察到的强收缩机制；与检查点中的权重和特征消失一致，但没有证明它是本项目退化的唯一原因。不能把零任务梯度对照当作真实训练重现实验。

[论文 §4.2](https://proceedings.neurips.cc/paper_files/paper/2025/file/f5717c76feff4f751604c0678c46627b-Paper-Conference.pdf) 明确指定 AdamW、lr=4e-4、weight_decay=0.05；公开 `scene/GS_LRM.py` 实际使用 Adam。此前仅记录这种不一致，本次在权重收缩证据和 CPU 对照基础上，明确改用论文给出的优化器身份，属于有依据的复现修复尝试，不进行超参数搜索，也不声称已经恢复作者完整训练配方。

原始证据与 CPU 脚本位于 `work_dirs/diagnostics/second_repair_20260926/{audit_decay.py,audit.json,audit.log}`（脚本开始于 9 月 26 日，未修改任何已有检查点）。

### 14.3 实现范围

1. 两档配置显式设置 `optimizer.type='AdamW'`，训练入口和 GPU smoke 脚本共用 `build_optimizer`。只开发 smoke 入口，本次不执行它。旧配置缺少 type 时仍按原 Adam 解释，避免篡改历史配置语义。
2. UNet/adapter 两组的 lr、weight_decay、betas、eps 全部保持原数值，归一化参数和 bias 仍留在原参数组内，不额外增加免衰减分组。保持 FP32 参数/动量与 BF16 autocast。
3. 模型结构、深度分类/残差/tanh、损失和权重、seed=0、静态六相机协议均不改；PD-Block 的全 1 掩码仍按公开代码保留，未猜测阈值或重设计分流。
4. 每 1k 验证、每 5k 保存、每 10 次验证 mini、100,001 步及最终 mini 均保持。旧训练记录保留，用户清理后从头启动；不把已压零的权重和 Adam 状态转为 AdamW 接着训。
5. 启动日志/飞书、诊断及新检查点记录实际优化器类型；恢复同时核对配置和新检查点的类型元数据。同一 AdamW 实验仍可自动续训。
6. 在已有更新诊断中增加参数平均绝对值/最大值、非零梯度元素数/最大梯度；增加深层 GroupNorm、几何融合 BatchNorm 的监测，并单独记录归一化的 `weight`，避免非零 bias 掩盖 scale 压零。仍只读记录，不改变损失、梯度或随机状态，不按分数自动停训或审查样本。

### 14.4 验证与下一轮判读

本次只运行 CPU 合成测试、现有检查点统计及有界部分前向；不运行真实训练、GPU 前向、CUDA 光栅器或全量评估。`tests.test_omniscene` 与 `tests.test_omniscene_repair` 共 **26 项通过**；使用 `CUDA_VISIBLE_DEVICES=''`，CUDA 可用性返回 False，CUDA 初始化/同步/设备查询替换为报错函数。训练入口与 GPU smoke 文件仅作语法检查，未执行。

新增测试确认：UNet/adapter 两组在零任务梯度下按固定系数乘法衰减，衰减不进入自适应动量；CPU autocast 下参数/梯度/动量均保持 FP32；非零 bias 不掩盖归一化 scale 压零；诊断不改变优化更新与随机状态；旧 Adam 配置及新检查点的类型元数据均能阻止错误恢复；AdamW 的参数、动量及训练/验证/mini 事件在恢复后与连续运行一致。`git diff --check` 通过。既有训练结果与用户暂存区均未清理或覆盖。

下一轮不仅检查参数是否发生变化，还要结合归一化缩放参数幅值、辅助深度空间标准差、分类集中程度和损失趋势，重点覆盖前两轮出现异常的 10k–20k 区间。本次修复消除了已定位的优化器语义差异，正常收敛仍需用户之后的训练验证；若再次退化，不自动进入调学习率/损失/种子的循环。

## 15. 第三轮退化：原生光栅器反向修复（2026-09-27）

### 15.1 已确认的现象与证据边界

第三轮工作目录为 `work_dirs/omniscene/driverecon_static_t1_112x200_adamw`，用户停止于 28,054 步。两次 mini 均完整评估 2,048 个 bin，均非 total 正式结果：

| 步数 | 视角组 | PSNR | SSIM | LPIPS | PCC |
| ---: | --- | ---: | ---: | ---: | ---: |
| 10k | all_18 | 20.743747 | 0.566147 | 0.541053 | 0.719483 |
| 20k | all_18 | 15.218267 | 0.253531 | 0.666288 | 0.352688 |
| 10k | novel_12 | 19.581767 | 0.527088 | 0.556369 | 0.712153 |
| 20k | novel_12 | 14.841846 | 0.223764 | 0.673601 | 0.342825 |

逐步训练日志把首次明显异常缩小至 13,100–13,200 步。验证损失从 13k 的 5.175 升至 14k 的 9.044；14k 辅助深度约 99.83% 为零，18k 的 10 个验证 bin 中有 9 个完全为零。这些是预测值统计，与辅助监督的有效像素集合为空是两回事。

CPU 对 5k/10k/15k/25k 检查点做有界部分前向：一个缓存样本的辅助深度输入在 10k 约为 −1.15～0.05，在 15k 已为 −220～−22；纯 FP32 仍饱和。此时深层归一化缩放参数仍约 0.74，不是第 14 节的参数压零现象。仅把 tanh 改为等价 sigmoid 或切换整个前向为 FP32，缺乏挽救上游特征失稳的证据。

用户授权后，从原 10k 完整状态在独立目录用原二进制重放 4,000 次更新，未改配方、数据顺序或优化器，不发送通知、不覆盖原目录。该重放到 14k 时验证损失为 5.273，**未重现原 14k 的坍缩**。因此不能声称已经找到该次突变的唯一触发因素，也不能将此次重放视作确定性重现或修复后的收敛结果。

### 15.2 确定的实现错误

`submodules/diff-surfel-rasterization/cuda_rasterizer/forward.cu::compute_aabb` 计算投影中心时使用 cutoff=3，即 `q=(9,9,-1)`。其函数定义为：

~~~text
d = sum(q * Tw²)
px = sum(q * Tu * Tw) / d
py = sum(q * Tv * Tw) / d
~~~

原 `backward.cu::compute_transmat_aabb` 却按 cutoff=1 计算导数，并遗漏分母求导产生的交叉项。因此对低通滤波中心回传的梯度不是上述前向函数的导数，可把位置、尺度和旋转向错误方向更新。

双精度反例：`Tu=(2,3,4), Tw=(0.2,0.3,2)`。`dpx/dTw` 的正确值为 `(−8.023574, −12.035361, 3.261372)`；旧实现为 `(−0.527479, −0.811249, −1.103032)`，第三项甚至反号。这是实现错误，不是根据验证集指标选择更有利的训练参数。

对输出余切 `(gx,gy)`，修复后的 VJP 为：

~~~text
gTu = gx * q * Tw / d
gTv = gy * q * Tw / d
gTw = q / d * [gx * (Tu − 2*px*Tw) + gy * (Tv − 2*py*Tw)]
~~~

该增量累加到已有变换矩阵梯度，保留射线交点和其他损失路径的梯度。`aabb_math.h` 由实际 CUDA kernel 调用，也供 CPU 编译测试直接调用。前向只将原常数 3 命名为共享 `AABB_CUTOFF`；没有改变数值公式。当前 `TIGHTBBOX=0`；若未来启用随 opacity 变化的边界，编译检查要求先实现对应导数，避免静默套用固定 cutoff 的反向。

### 15.3 修复范围与运行要求

- 只修正上述原生反向；不改网络结构、tanh、PD 分流/分区、类别映射、数据、损失及权重、AdamW、lr、wd、eps、seed 或混合精度。
- 保持每 1k 验证、5k 保存、10 次验证 mini、100,001 次更新及最终 mini。
- 原生扩展发布 `aabb_backward_version=1` 和编译时源文件 SHA256；两档配置要求版本 1。未重编译时给出明确编译指令，不允许悄悄继续使用旧反向。
- 旧权重和优化器仅作诊断来源，新正式实验使用空 work_dir 从头训练；已有记录由用户决定何时清理。
- 原生前向在分母接近零时仍可能产生大导数。此次不新增 epsilon、截断或裁剪；真实导数修正不等于消除所有数值奇异性。

### 15.4 验证与产物

诊断产物位于 `work_dirs/diagnostics/third_repair_20260927/`。`audit.py/audit.json` 保存 CPU 特征及激活梯度统计；`replay.py`、`replay_gpu/trace.jsonl` 和 `complete.json` 保存旧二进制 10k→14k 的隔离重放。该目录的权重、日志和测速均不是正式训练/评估结果。

`tests/test_rasterizer_gradients.py` 编译 CUDA 实际使用的同一 C++ helper，对照 double 自动微分与中心有限差分，并检查梯度累加、旧二进制拒绝使用和旧 renderer 状态拒绝恢复。与既有回归合计 30 项 CPU 测试通过。

`tests/verify_rasterizer_cuda.py` 对真实 CUDA 低通支路做有限差分，覆盖预计算变换矩阵以及 means/scales/rotation 路径；同时比较同一固定检查点的高斯输出和渲染。脚本不做优化更新，先检查 GPU 显存占用低于 1 GB，旧/新结果分别记录到 `native_before.json` / `native_after.json`。

实际验证已完成：

| 检查 | 结果 |
| --- | --- |
| 原生预计算变换矩阵 VJP | 最大绝对差由旧版 0.266147 降为 0.000193；修复后通过 float32 有限差分检查 |
| 原生 means/scales/rotation 链式梯度 | 旧版最大绝对差约 0.166；使用足够区分 float32 舍入的扰动后，新版最大绝对差 0.000291，检查通过 |
| 低通分支覆盖 | 样例的 rho2d=3.887 < rho3d=84.334，像素值及中心梯度非零，不是绕过错误分支的空测试 |
| 前向保持 | 微型渲染以及同一 10k 检查点、缓存 bin 的全部高斯参数和一张渲染图，与旧二进制逐值相同（rtol=atol=0） |
| 原生构建 | 本机 drivingrecon 的 editable 扩展已重新编译；实际版本 1，源文件 SHA256 为 `4ac0ee0af315415e1a361528742dbcf7373a28786ccae975368f92c91be4700f` |
| 两档功能检查 | 112×200、224×400 各用一个训练 bin 做 2 次优化更新、验证、完整状态保存恢复及一个 mini bin 的四指标检查，均通过 |

两档功能检查调用 `tests/smoke_omniscene_cuda.py`，报告保存在 `smoke/verification.json`。其临时权重在 `/tmp` 中验证后自动移除，只保留诊断报告，不改变正式的每 5k 保存配置，也不向飞书发送消息。该检查证明新二进制可用于当前训练链路；两次更新和单 bin 分数不证明长期收敛或正式性能。

### 15.5 两轮交叉复核结论

使用本地 expert-review 技能，由数值实现、复现协议、稳定性三位专家分别独立审查，再结合作者回应进行第二轮交叉审阅。三方同意只修复已证实的原生 VJP，不把 PD 掩码全一、异或 fold、批量大于一的几何排列等其他原代码缺陷一并改写；当前没有证据证明这些缺陷触发了本轮突变。

作者接受全部核查意见：原/新辅助 loss 的单位与梯度一致，18 视角分块链式回传及 K/位姿测试通过；不将辅助预测零误作监督空集合，不将 tanh 饱和或 surrogate 标量作为已证明根因。保留意见是：局部导数修复必须通过实际 CUDA 分支验证，旧优化器历史应隔离，分母近零风险仍存在，最终收敛仍待新的正式训练观察。没有专家认为目前可以保证恢复论文指标。

## 16. 发布代码一致性审查与最终保留项（2026-09-28）

### 16.1 结论与用户决定

用户要求对比实验尽可能遵循公开代码，不以论文的不同写法作为改变优化器或调度器的理由。本次逐项比较暂存区、`main` 的 `scene/GS_LRM.py::training_setup` 和 `train.py` 的实际执行路径。

**保留已经验证的原生反向修复；另保留用户确认的 FP32 参数/动量、AdamW 两项稳定性例外。默认配置与刚完成的完整实验一致。** 这不是论文与公开代码的完全同配置复现，应明确披露这两项例外及静态数据适配。既有网络架构、监督、损失项和权重不变，不继续搜索学习率、衰减、种子或替换模块。

光栅器投影中心 VJP 是通过有限差分确认的实现错误。修复后 FP32+AdamW 完成 100,001 步且 total 指标正常，支持修复有效，但不能推出“它是唯一原因，因此其他改动都可无影响地回退”。本次单独恢复 Adam 的诊断在 4k 又出现辅助几何深度接近常值，证实优化器差异不能排除。第 14 节的零任务梯度对照仅证明衰减语义差异；保留 AdamW 的最终依据是实际退化证据和用户决定，而不是仅因为论文使用 AdamW。

### 16.2 逐项审查

| 项目 | 作者实际运行的代码 | 最终处理及影响 |
| --- | --- | --- |
| 原生反向 | cutoff 与前向不一致，商式求导缺少交叉项 | 保留正确 VJP、共享常数、数值测试及二进制版本检查；这是改变错误梯度的实现修复，前向不变 |
| 优化器 | `torch.optim.Adam`，耦合 L2 | 保留 AdamW 稳定性例外；该改动确实影响衰减与训练轨迹，不能说成无影响整理 |
| 学习率 | 两参数组显式 0.0004，覆盖构造函数全局默认 0.001 | 始终保持有效值 0.0004 |
| 学习率调度 | 只创建 `lrm_scheduler_args`，训练循环未调用 | 始终恒定；没有从原版调度改成论文调度，也不添加 warmup/余弦衰减 |
| weight_decay / betas / eps | 两组 .05；默认 (.9,.999)；1e-15 | 数值全部保持；归一化和 bias 仍在原参数组内，不额外创建免衰减组 |
| 参数、梯度、动量存储 | BF16 | 保留用户确认的 FP32 数值例外，防止小更新被舍入；该差异也会影响更新、存储量和训练轨迹 |
| 前向混合精度 | BF16 | 保持 BF16 autocast，必要几何/光栅接口使用 FP32 |
| 模型、激活与损失 | 已有公开模块与损失配方 | 暂存的退化修复未重设计网络或损失；不改 tanh、PD 掩码、fold、深度范围或损失权重 |
| 保存与验证 | 原 Waymo 循环 | 按用户指定协议保留 1k 验证、5k 保存、10k mini、最终 mini；保存间隔影响中断后重放量，不改变连续训练的更新 |
| 日志与恢复保护 | 无对应 OmniScene 实现 | 保留只读诊断、事件恢复、优化器/二进制身份保护；CPU 测试确认诊断不改变更新和随机状态 |

静态 T=1、六相机、两档实际输入分辨率、Metric3D 米制监督、正确的深度类别映射、动态掩码、一个 bin 一次更新、18/12 路评估均为用户确认的实验适配要求。不能为“回归原版”恢复额外时序/LiDAR 输入、两倍输入放大或原 Waymo 的同一 batch 连续更新 500 次。

### 16.3 回退诊断的实际证据

GPU 启动前显存占用约 265 MiB，满足用户低于 1 GB 才调试的条件。全部诊断使用已修正反向的原生扩展、独立目录、相同种子与样本顺序，从头初始化；不加载已完成权重、不修改正式记录、不发送通知。

**BF16 与 FP32 存储对照：** 各运行 1,000 步。BF16 下，三层被监测归一化 scale 有非零任务梯度，却始终全部等于初始化值 1；FP32 下权重能够更新。这个舍入问题在修正光栅器后仍存在，支持保留 FP32 例外。不能把 FP32 对更新的改善误写为单独保证收敛。

**恢复 Adam 的较长诊断：** FP32 存储、BF16 前向、正确 VJP，其余配置不变；验证结果如下：

| 更新步数 | 主深度空间标准差（米） | 主深度最大类别占比 | 辅助深度空间标准差（米） |
| ---: | ---: | ---: | ---: |
| 1,000 | 38.9811 | 26.57% | 27.0940 |
| 2,000 | 22.8809 | 47.76% | 13.3075 |
| 3,000 | 23.0637 | 66.55% | 5.2905 |
| 4,000 | 29.4787 | 28.69% | 0.006809 |

4k 辅助深度均值约 124.64 米，全部 10 个验证 bin 的空间标准差相同，接近常值；三层被监测 GroupNorm 的 scale 均值约 0.0386。此时主深度仍有空间变化，因此不能声称主深度/PCC 已经全面崩溃，但已不满足避免辅助深度退化的要求。用户据此明确选择保留已完成稳定实验的 AdamW，作为另一个有证据的稳定性例外。

这次计划最多 14k 的诊断在 4,090 步收到 SIGTERM（退出码 143），无 Python 异常记录，发送者和原因未知；不能记作完成 14k，也不能记作 CUDA 错误。未启动新的完整正式训练。证据保存在 `work_dirs/diagnostics/release_recipe_20260928/`：`adam_bf16_1000/`、`adam_fp32_1000/`、`adam_fp32_14000/`（最后一个目录名表示计划上限，实际步数见 `status.json`）。

### 16.4 结果归属与代码验证

已完成的实验仍是 `FP32 + AdamW + corrected AABB backward`，目录 `work_dirs/omniscene/driverecon_static_t1_112x200` 原样保留，默认配置保持兼容：

- checkpoint：`step-00100001`；total：30,080 个 bin，两组均无缺失/重复，四指标均有限。
- all_18：PSNR 23.605495，SSIM 0.732151，LPIPS 0.291745，PCC 0.781584。
- novel_12：PSNR 21.378014，SSIM 0.652182，LPIPS 0.343138，PCC 0.770527。

不能将这些结果改标为原版 Adam 的成绩。若将来另行比较 Adam，须使用不同工作目录并从头训练，不能转换 AdamW 动量继续训练后声称原版复现。

CPU 验证共 32 项：新增测试直接抽取公开 `Gaussian_LRM.training_setup`，给参考 Adam 与工厂 Adam 相同 FP32 参数和梯度，连续更新后参数及动量逐值一致，排除诊断用的 Adam 工厂另有语义错误；保留光栅器 VJP、诊断无副作用、5k 保存和断点恢复测试；Adam/AdamW 两类均覆盖完整状态重载，拒绝把旧 AdamW 权重错误标为 Adam 配置。正式结果目录没有被诊断覆盖，原暂存区保持不变，本次审查改动留在工作区供审阅。
