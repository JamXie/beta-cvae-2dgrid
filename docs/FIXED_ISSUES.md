# 原始工程问题与修复记录

下表按“原文件/函数 → 问题 → 修复”记录整理过程中处理的关键问题。
模型数学形式、损失公式与数据列约定未被改动；修复集中在阻塞性错误、
数据泄漏、可移植性与运行安全。

| 原位置 | 问题 | 修复 |
| --- | --- | --- |
| `GPU/FinalModel2.0/*.py` | 全部脚本写 `from model01 import ...`，但实际文件名是 `model.py`，导入直接失败 | 统一改为 `from beta_cvae.models.registry import build_model`，按模型名构造 |
| 全部训练/验证/预测脚本 | 顶层直接执行，数据路径硬编码为 `../data`、`../../202411...`、`H:\` 等 | 改为 `argparse` + `pathlib`，数据与输出路径必须显式传入 |
| 全部 `torch.load` | 未传 `map_location`，在无 GPU 机器上加载 CUDA 检查点失败 | `checkpoint.load_checkpoint(path, device)` 统一传入设备 |
| 全部 `torch.load` | 直接反序列化任意 pickle 对象 | 优先 `weights_only=True`；配置/历史以 JSON 字符串保存，兼容 PyTorch 1.13 |
| `os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'` | 掩盖 OpenMP 运行时冲突，可能造成静默数值错误 | 删除该设置；依赖版本写入 `requirements.txt` |
| `2DGridCVAEtrain.py` | 每个训练步无条件调用 `torch.cuda.empty_cache()`，CPU 训练也执行 | 仅在 `device.type == "cuda"` 时清理 |
| `2DGridCVAEtrain.py` | 测试集取 `A_step=a_step[1]`，即 `Degree=5`，与训练集重叠 | 测试/预测默认角度改为 23；`check_split_disjoint` 断言 train/val/test 互斥 |
| `DataLoadAndPreprocess.py::Grid2DNpyDataset_Val` | 验证角度集合包含 23，与测试角度相同 | 从 `VAL_PRIME_ANGLES` 移除 23 |
| `...RBFIvalidation...For.py` | 硬编码 CVAE04 分支，调用 `decode(latent_z)`，与 `M_Rm_CVAE02.decode(z, z_t, z_a)` 签名不匹配 | 统一 `decode_from_latent(model, z, step, angle)`，按注册表 `decode_style` 分派 |
| `...RBFIvalidation...For.py` | 评估时对条件嵌入做随机采样，同一输入多次运行结果不同 | `z_zt_za` 模式评估时使用嵌入均值，不添加噪声 |
| 全部脚本 | 用 `python-pptx` 导出结果，依赖重且不可复现 | 删除 PPTX 导出，改为 PNG + CSV + JSON |
| `DataLoadAndPreprocess.py::center_and_transform_coordinates` | 原地修改输入数组，参考坐标被重复平移 | 先 `np.array(..., copy=True)` 再平移 |
| `DataLoadAndPreprocess.py::voxel_to_grid` | 只接受 `torch.Tensor`，且从未显式关闭 pickle | 同时接受 ndarray；数据读取使用 `np.load(..., allow_pickle=False)` |
| `...RBFIvalidation...For.py` | 用 `np.arange(0, epochs)` 绘制损失，与实际记录长度可能不一致 | 按实际 loss 列表长度绘图 |
| 可视化脚本 | 调用 `plt.show()`，无图形环境时阻塞；图中中文缺字体 | 模块导入时设 `matplotlib.use("Agg")`；图内文字用英文 |
| 数据划分 | 训练/验证/测试的划分逻辑分散在四个 Dataset 类中 | 合并为 `NpyGridDataset` + `split_samples`，规则集中在 `constants.py` 与 `datasets.py` |
| 模型定义 | 一个 2400+ 行的 `model.py` 混放基础、多分支、残差与最终模型 | 按类族拆为 `basic.py`、`multibranch.py`、`residual.py`、`final.py`，并保留原类名 |
| 模型调用 | 推理脚本按 `decode` 形参个数猜测调用方式，容易错位 | 注册表显式声明 `decode_style`、`encode_takes_condition`、`forward_takes_condition` |
| 数据轴顺序 | 文档与代码对“80/84 对应哪个方向”表述不一致 | 由 padding/mask 反推确认：`[通道, 环向 80, 高度 84]`，并写入 `constants.py` 与 `DATA_FORMAT.md` |
| 真实数据不可得 | 原始数据与权重不在仓库中，克隆后无法直接运行 | 提供 `generate-synthetic` 合成数据生成器与 `convert-csv` 真实数据转换入口 |

## 未改动的部分

为避免与论文实验脱节，以下内容保持原样：

- 所有网络层的通道数、卷积核、步长、激活函数与残差捷径；
- 损失函数中 `logvar` 的写法与 `1e-8` 平滑项（不是标准 ELBO）；
- FE CSV 的列位置约定；
- 训练/验证的“5 的倍数”时间步与质数验证角度思路；
- 论文最终模型名与超参数（`M_Rm_CVAE02`、`latent=16`、`beta=1e-6`）。
