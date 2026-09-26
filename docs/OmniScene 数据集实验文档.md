# OmniScene 数据集实验文档

> 状态：已按确认方案实现，并完成 CPU 协议测试与有界 CUDA 验证；未启动正式训练或全量评估。
> 第 1–11 节描述确认的实验协议，第 12 节记录 2026-09-26 的实现、验证和运行依赖。
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
| checkpoint_every_steps | 1_000；最后一步另存；每次 mini 前有可恢复的训练状态 |
| use_dynamic_mask | True，只影响训练/验证损失，不屏蔽正式图像指标 |
| shuffle | train=True；val/mini/total=False |
| precision / parameter_dtype | BF16 计算、BF16 网络参数，显式 `model.parameter_dtype=bfloat16`；深度采样点、几何变换、softmax、光栅接口及指标为 FP32 |
| optimizer | Adam；UNet 和 adapter 两组均 lr=4e-4、weight_decay=0.05；betas=(0.9,0.999)、eps=1e-15 |
| lr_scheduler | constant；原代码虽创建衰减函数，实际循环没有调用，不引入 SVF-GS 的 warmup |
| gradient_clip | 不新增裁剪策略，沿用当前有效训练路径 |
| initialization | 无恢复状态时随机初始化；不加载其他方法的预训练权重 |
| seed / deterministic | seed=0，对齐原入口最后生效的 seed；cuDNN deterministic=True 与原 setup_seed 一致，benchmark=False |
| report_to | 默认本地日志/JSON；如开启 WandB，固定 offline，不执行自动 sync |
| feishu.enabled | True，通过已有 `send_feishu` 模块发送启动和 mini 完成通知 |
| resume | auto，恢复当前 work_dir 中最新完整训练状态 |

这里的 BF16 不应依赖构造函数中的裸 `.cuda()`；设备由入口统一管理，避免逻辑设备编号错误。实现时记录参数/计算精度，不把精度更改、学习率调优混入本次适配。

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

    if global_step % 1_000 == 0:
        save_training_state()
        validate(val_loader)
        validation_count += 1
        persist_validation_state()
        if validation_count % 10 == 0:
            evaluate_mini(reason='periodic')
            persist_evaluation_state_and_notify()

save_training_state()              # step=100_001
evaluate_mini(reason='final')       # 必须执行，不能用 step=100_000 的结果替代
persist_evaluation_state_and_notify()
mark_training_and_final_mini_complete()
~~~

正常完成时有 100 次周期验证、10 次周期 mini，再加最终第 100_001 步的 1 次 mini。原项目没有完整的训练中 mini 功能；本次实现仍补齐，以满足用户要求的最终 mini，并统一周期评估路径。

验证聚合全部 10 个 bin，不只保留最后一个 batch。mini/total 复用同一个 evaluator；评估前切 eval/no_grad，结束恢复 train。评估和可视化消耗的随机状态不改变训练采样序列。

### 6.2 自动恢复

在本实验自己的 work_dir 中查找数值步数最大的完整 checkpoint；优先使用可用的 latest，若 latest 未更新则扫描 checkpoint 步号。仅检查训练状态文件是否完整，不检查数据集。保存采用临时目录写完后原子发布，避免把半写入 checkpoint 当作可恢复状态。

恢复内容包括：

- UNet、adapter 及所有保留模块的参数/buffer；
- Adam 状态、学习率状态、混合精度状态（如实际使用）；
- global_step、epoch、epoch 内 batch 游标及 sampler/generator 状态；
- Python、NumPy、Torch CPU/CUDA RNG；
- validation_count、最近完成的 val/mini step、待完成评估事件、final_mini 完成标记。

只加载模型权重不能算自动续训。原 `Gaussian_LRM.capture/restore` 的不完整状态路径不作为新协议的恢复来源，也不自动混用另一个分辨率的 checkpoint。

恢复后先完成已经到达触发点但尚未完成的验证/mini，再继续训练，避免在 10k 保存后中断就漏掉 mini。若已训练到 100_001、但 final_mini 未完成，只补最终 mini；若训练和最终 mini 都完成，直接报告已完成，不再多训一步。中途被打断的评估可以从该 split 开头重跑，并覆盖同一事件的临时结果，避免把两次半程统计相加。

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
  checkpoints/step-00001000/...
  latest.json
  final.json
  validation/step-00001000/...
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

实际模型统计为：可训练参数 58,249,264、冻结参数 0、总参数 58,249,264，两档分辨率相同。速度和显存仅做了短程验证，不能作为正式数据集基准；见第 12 节。

## 9. 调用命令

以下命令已实现。环境使用 README 的 drivingrecon 环境；已有环境需补充 `pip install lpips==0.1.4`。LPIPS 的 VGG16 权重须事先在本地准备好，具体见第 12 节。

~~~bash
conda activate drivingrecon

# 两档实验分别训练；重复同一命令时自动恢复本 work_dir。
CUDA_VISIBLE_DEVICES=0 python train_omniscene.py --config configs/omniscene/112x200.py --mode train
CUDA_VISIBLE_DEVICES=0 python train_omniscene.py --config configs/omniscene/224x400.py --mode train

# 正式 total 评估：默认读取相应实验的最终训练 checkpoint。
CUDA_VISIBLE_DEVICES=0 python train_omniscene.py --config configs/omniscene/112x200.py --mode test --test-split total --checkpoint final
CUDA_VISIBLE_DEVICES=0 python train_omniscene.py --config configs/omniscene/224x400.py --mode test --test-split total --checkpoint final
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

以上检查针对代码与协议，不是对 OmniScene 做质量审查。已完成的短程验证见第 12 节；没有全量数据扫描、坏样本过滤或正式训练。

## 11. 本次核对的代码依据

- DriveRecon：[train.py](../train.py)、[arguments/__init__.py](../arguments/__init__.py)、[arguments/nvs.py](../arguments/nvs.py)、[配置合并](../utils/params_utils.py)。
- DriveRecon 模型：[GS_LRM.py](../scene/GS_LRM.py)、[实际 UNet](../scene/PointNet.py)、[PD_Block.py](../scene/PD_Block.py)、[Camera](../scene/cameras.py)、[损失函数](../utils/loss_utils.py)。
- DriveRecon 深度：[surfel CUDA forward](../submodules/diff-surfel-rasterization/cuda_rasterizer/forward.cu)、[CUDA 常量](../submodules/diff-surfel-rasterization/cuda_rasterizer/auxiliary.h)。
- SVF-GS：[OmniScene Dataset](../../SVF-GS/data/omniscene_dataset.py)、[数据读取](../../SVF-GS/data/transforms/loading.py)、[主配置](../../SVF-GS/configs/main.yaml)、[损失与测试](../../SVF-GS/model/omni_gs.py)、[训练/通知](../../SVF-GS/trainer.py)、[指标](../../SVF-GS/tools/metrics.py)。
- depthsplat：[OmniScene Dataset](../../depthsplat/src/dataset/dataset_omniscene.py)、[数据读取](../../depthsplat/src/dataset/utils_omniscene.py)、[112×200 配置](../../depthsplat/config/experiment/omniscene_112x200.yaml)、[224×400 配置](../../depthsplat/config/experiment/omniscene_224x400.yaml)。
- depthsplat：[主入口](../../depthsplat/src/main.py)、[ModelWrapper](../../depthsplat/src/model/model_wrapper.py)、[指标](../../depthsplat/src/evaluation/metrics.py)、[深度光栅化](../../depthsplat/src/model/decoder/cuda_splatting.py)、[patch shim](../../depthsplat/src/dataset/shims/patch_shim.py)。

以上以当前可执行代码为依据。兄弟项目旧实验文档中的文件名、PCC 实现状态和裁剪描述可能已经滞后，本方案不直接把旧文档当作当前行为。


## 12. 实现与验证记录（2026-09-26）

### 12.1 已实现内容与具体做法

- 两档配置、静态六相机数据/模型接口、原生 2D Gaussian 渲染、Metric3D 分类/L2 和分割监督、SVF-GS 风格的新视角 RGB 掩码均已接入。
- `train_omniscene.py` 提供配置展开、train/test、data_root/work_dir 覆盖和显式 mini/total；训练器按完成的 optimizer 更新数调度 val、周期 mini 和最终 mini。
- 完整 checkpoint 包含网络、Adam、精度状态、Python/NumPy/Torch/CUDA RNG、采样排列、epoch/游标和 DataLoader generator。评估完成状态单独原子记录到 checkpoint 的 `events.json`，恢复时先补未完成事件。
- 同目录自动恢复通过扫描最新完整 step 完成，不信任过期的 latest 指针；实际指针文件为 `latest.json`、`final.json`。不同分辨率、模型、损失、优化器、精度和训练调度配置不混用。相关约束检查的是实验配置/训练状态，不审查数据集。
- 公用的原 adapter 已抽出到 `scene/gaussian_adapter.py`，原 `GS_LRM.py` 复用该类；`scene/__init__.py` 延迟加载 Waymo/CUDA 特有依赖，使新的网络与 CPU 测试不需要加载原数据预处理栈。
- 静态分支的注意力使用 PyTorch SDPA，保持原权重和注意力计算定义，避免无 xFormers 时显式构造巨大的注意力矩阵；旧分支默认行为保留。
- 18 路 RGB loss 采用“逐视角累计高斯参数梯度，再对网络反传一次”的等价实现，避免同时保留 18 份光栅器反向缓冲。仍是一个 bin、一次 optimizer 更新、18 路 L1 求和，已经与直接求和反传做梯度一致性测试。
- PCC 专用渲染输出累计中心 z，未经过 RGB clamp、alpha 归一化或 surfel 交点深度替换。输出包括两组指标、逐 bin CSV、汇总 JSON、参数量、精度与重建耗时。
- 原代码的网络参数本身就是 BF16，因此最终配置显式固定为 `model.parameter_dtype='bfloat16'`。没有仅开启 autocast 而无意改成 FP32 主参数训练；深度采样点仍保留 FP32。

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
