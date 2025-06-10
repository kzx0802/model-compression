import torch
import torchvision
from torchvision.datasets import CocoDetection
import torchvision.transforms as T
from torch.utils.data import DataLoader
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.utils.prune as prune

# -------------------------
# 1. 数据集准备
# -------------------------
coco_root = 'C:/Users/xianyi.zhao23/Desktop/coco'  # 替换成你的COCO根目录路径
ann_file = coco_root + '/annotations/instances_val2017.json'
img_dir = coco_root + '/val2017/'

transform = T.Compose([
    T.ToTensor()
])

dataset = CocoDetection(img_dir, ann_file, transform=transform)
data_loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

# -------------------------
# 2. 加载预训练模型
# -------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load pre-trained Mask R-CNN model
model = torchvision.models.detection.maskrcnn_resnet50_fpn(
    weights=torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
model.to(device)
model.eval()


# -------------------------
# 3. 定义剪枝函数
# -------------------------
def prune_model(model, ratio):
    prune_report = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.l1_unstructured(module, name="weight", amount=ratio)
            prune_report[name] = {
                "prune_ratio": ratio,
                "remaining_weights": int(torch.sum(module.weight != 0))
            }
    return prune_report


# -------------------------
# 4. 进行推理并保存预测结果
# -------------------------
coco_gt = COCO(ann_file)
results = []

PRUNE_RATIOS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]  # Pruning ratios
mAPs = []

print("Running inference on COCO val2017...")
for ratio in PRUNE_RATIOS:
    print(f"\n>>> Pruning ratio: {ratio:.2f}")

    # Initialize fresh model
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(
        weights=torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
    model.to(device)
    model.eval()

    if ratio > 0:
        report = prune_model(model, ratio)
        print(f"=== Pruning Report (Ratio={ratio}) ===")
        for layer, info in report.items():
            print(
                f"Layer: {layer.ljust(30)} | Pruned: {ratio * 100:5.1f}% | Remaining: {info['remaining_weights']:8d} weights")

    # Evaluate
    results = []
    for batch_idx, (imgs, targets) in enumerate(tqdm(data_loader)):
        img = imgs[0].to(device)
        target = targets[0]  # Get the first target (list of annotations)

        # Get image_id
        if len(target) > 0:
            image_id = target[0]['image_id']
        else:
            # For images with no annotations
            image_id = dataset.coco.imgs[dataset.ids[batch_idx]]['id']

        with torch.no_grad():
            output = model([img])[0]

        boxes = output['boxes'].cpu().numpy()
        scores = output['scores'].cpu().numpy()
        labels = output['labels'].cpu().numpy()

        for box, score, label in zip(boxes, scores, labels):
            if score < 0.05:  # COCO常用score阈值
                continue
            x_min, y_min, x_max, y_max = box
            result = {
                'image_id': int(image_id),
                'category_id': int(label),
                'bbox': [float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)],  # COCO需要xywh
                'score': float(score)
            }
            results.append(result)

    # 保存预测为JSON
    with open(f'maskrcnn_coco_results_ratio_{ratio}.json', 'w') as f:
        json.dump(results, f)

    # 评估 mAP 使用 pycocotools
    coco_dt = coco_gt.loadRes(f'maskrcnn_coco_results_ratio_{ratio}.json')
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()  # 输出 mAP 结果

    # 提取 mAP@0.5:0.95
    mAP = coco_eval.stats[0]
    mAPs.append(mAP)

# -------------------------
# 5. 绘制 mAP 和 pruning ratio 的折线图
# -------------------------
plt.figure(figsize=(8, 6))
plt.plot(PRUNE_RATIOS, mAPs, marker='o', linestyle='-', linewidth=2)
plt.title('mAP vs Pruning Ratio (Mask R-CNN on COCO val2017)')
plt.xlabel('Pruning Ratio')
plt.ylabel('mAP@0.5:0.95')
plt.grid(True)
plt.xticks(PRUNE_RATIOS)
plt.ylim(0, 1)
plt.show()
