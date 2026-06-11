# BJTU Campus Vision
面向校园场景的图像检索与文字检测系统设计与实现
---

>[!IMPORTANT]
>功能演示视频：
>- 通过网盘分享的文件：录屏2026-06-11 17.32.33.mov
链接: https://pan.baidu.com/s/1ONK1HL7JclmoOH2Adm3psw?pwd=9j2p 提取码: 9j2p

## 目录

<!-- toc:start -->
- [BJTU Campus Vision](#bjtu-campus-vision)
  - [面向校园场景的图像检索与文字检测系统设计与实现](#面向校园场景的图像检索与文字检测系统设计与实现)
  - [目录](#目录)
  - [1. 环境](#1-环境)
  - [2. 数据](#2-数据)
  - [3. 训练与评测](#3-训练与评测)
    - [3.1 图像检索](#31-图像检索)
    - [3.2 文字检测](#32-文字检测)
    - [3.3 端到端展示](#33-端到端展示)
  - [4. 输出](#4-输出)
  - [5. 参考文献](#5-参考文献)
<!-- toc:end -->

---

本项目用于课程实验的两项任务：

- 图像检索：用 `image_retrieval/query/` 到 `image_retrieval/base/` 里做以图搜图。
- 文字检测：用 `object_detection/data/` 中的 LabelMe 标注训练文字区域检测模型。

数据集使用 `/home/junyi/BJTU2026dataset.zip`。代码通过 `uv` 管理环境。

## 1. 环境

配置环境变量：

```bash
cd /path/to/BJTU-campus-vision
export UV_CACHE_DIR=$PWD/.uv-cache
export UV_PYTHON_INSTALL_DIR=$PWD/.uv-python
export HF_HOME=$PWD/.cache/huggingface
uv sync --python 3.12
```

入口：

```bash
uv run bjtu-campus-vision --help
uv run python main.py --help
```

设备参数支持：

```text
--device auto|cuda|mps|cpu
```



运行轻量 demo 前需要确认检测模型权重存在：

```bash
ls outputs/full/detection/model.pt
```

## 2. 数据


准备数据：

```bash
uv run bjtu-campus-vision prepare-data --profile full --archive /path/to/BJTU2026dataset.zip
```

目录应为：

```text
data/raw/BJTU2026/image_retrieval/base/BJTU/
data/raw/BJTU2026/image_retrieval/base/util_pic/
data/raw/BJTU2026/image_retrieval/query/
data/raw/BJTU2026/object_detection/data/
```

`prepare-data` 会生成：

```text
data/processed/<profile>/manifests/base_manifest.jsonl
data/processed/<profile>/manifests/query_manifest.jsonl
data/processed/<profile>/manifests/det_train.jsonl
data/processed/<profile>/manifests/det_val.jsonl
```

说明：

- `full` 使用全部真实数据。
- `mini` 使用真实数据小子集，只用于快速验证代码链路。
- `util_pic` 是检索负样本，评测时不会和 query 匹配。

## 3. 训练与评测

### 3.1 图像检索

检索模型使用 `DINOv2 + GeM`。当前 BJTU2026 full 的配置默认使用 frozen DINOv2 baseline；实测它比 SimSiam 风格适配更稳，检索指标更高。

如果访问 Hugging Face 不稳定，使用镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

快速验证：

```bash
CUDA_VISIBLE_DEVICES=1 HF_ENDPOINT=https://hf-mirror.com uv run bjtu-campus-vision train-retrieval --profile mini --device cuda --epochs 1 --max-batches 1
CUDA_VISIBLE_DEVICES=1 HF_ENDPOINT=https://hf-mirror.com uv run bjtu-campus-vision eval-retrieval --profile mini --device cuda --topk 5
```

正式训练：

```bash
CUDA_VISIBLE_DEVICES=1 HF_ENDPOINT=https://hf-mirror.com uv run bjtu-campus-vision train-retrieval --profile full --device cuda
CUDA_VISIBLE_DEVICES=1 HF_ENDPOINT=https://hf-mirror.com uv run bjtu-campus-vision eval-retrieval --profile full --device cuda --topk 10
```


如果只是验证 CUDA 和训练代码，不想下载 DINOv2 权重，可以临时加：

```bash
--no-pretrained
```

正式训练不要加 `--no-pretrained`。

说明：`configs/full.yaml` 中 `retrieval.train_adapter: false` 时，`train-retrieval` 会保存 frozen DINOv2 encoder checkpoint，不再额外微调适配头。

### 3.2 文字检测

检测模型使用 torchvision 的 `fasterrcnn_mobilenet_v3_large_320_fpn`，默认加载 COCO 预训练权重后替换成单类检测头。首次运行会从 `download.pytorch.org` 下载权重到项目 `.cache/torch/`。

快速验证：

```bash
CUDA_VISIBLE_DEVICES=1 uv run bjtu-campus-vision train-detector --profile mini --device cuda --epochs 1 --max-batches 1
CUDA_VISIBLE_DEVICES=1 uv run bjtu-campus-vision eval-detector --profile mini --device cuda
```

正式训练：

```bash
CUDA_VISIBLE_DEVICES=1 uv run bjtu-campus-vision train-detector --profile full --device cuda
CUDA_VISIBLE_DEVICES=1 uv run bjtu-campus-vision eval-detector --profile full --device cuda
```


如果只是验证 CUDA 和训练代码，不想下载 torchvision 权重，可以临时加：

```bash
--no-pretrained
```

正式训练不要加 `--no-pretrained`。

单图预测：

```bash
CUDA_VISIBLE_DEVICES=1 uv run bjtu-campus-vision predict-detector --profile full --device cuda --image data/raw/BJTU2026/object_detection/data/fhy-0120240508151840.jpg
```

### 3.3 端到端展示

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 HF_ENDPOINT=https://hf-mirror.com uv run torchrun --standalone --nproc_per_node=4 -m cli demo --profile full --device cuda --topk 5
```

每个地点导出 2 组“query + topK 检索结果 + 检测框”，共 24 组。
展示图会从更深的候选 ranking 中补齐视觉去重后的 topK，避免同一画面以不同文件名重复出现；原始全量 ranking 仍保留在 JSON 中便于排查。

## 4. 输出

默认输出目录：

```text
outputs/<profile>/
```

包括：

```text
outputs/<profile>/retrieval/model.pt
outputs/<profile>/retrieval/metrics.json
outputs/<profile>/retrieval/rankings.json
outputs/<profile>/retrieval/plots/*.png
outputs/<profile>/retrieval/examples/*.png
outputs/<profile>/detection/model.pt
outputs/<profile>/detection/metrics.json
outputs/<profile>/detection/visualizations/*.png
outputs/<profile>/detection/predictions/*.png
outputs/<profile>/demo/*.png
outputs/<profile>/demo/rankings.json
outputs/<profile>/test_demo/*.png
outputs/<profile>/test_demo/rankings.json
```

full profile 已生成的最终展示结果位于：

```text
outputs/full/demo/
```

其中：

- `*.png`：最终端到端展示图，包含 query、检索 topK 和文字检测框。
- `rankings.json`：`items` 是最终展示样例，`rankings` 是全量 query 的原始候选结果。
- `outputs/full/test_demo/`：录制视频时快速复现指定样例的输出目录。


## 5. 参考文献

1. Oquab, M., Darcet, T., Moutakanni, T., et al. DINOv2: Learning Robust Visual Features without Supervision. arXiv:2304.07193, 2023.
2. Radenovic, F., Tolias, G., and Chum, O. Fine-Tuning CNN Image Retrieval with No Human Annotation. IEEE TPAMI, 2018.
3. Ren, S., He, K., Girshick, R., and Sun, J. Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks. NeurIPS, 2015.
4. Lin, T.-Y., Dollár, P., Girshick, R., He, K., Hariharan, B., and Belongie, S. Feature Pyramid Networks for Object Detection. CVPR, 2017.
5. Howard, A., Sandler, M., Chu, G., et al. Searching for MobileNetV3. ICCV, 2019.
6. Russell, B. C., Torralba, A., Murphy, K. P., and Freeman, W. T. LabelMe: A Database and Web-Based Tool for Image Annotation. IJCV, 2008.
7. Chen, X. and He, K. Exploring Simple Siamese Representation Learning. CVPR, 2021.
