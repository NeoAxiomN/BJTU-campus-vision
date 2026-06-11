def box_iou(
	box_a: list[float],
	box_b: list[float],
) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return 0.0 if union <= 0 else intersection / union


def detection_prf(
	pred_boxes: list[list[float]],
	gt_boxes: list[list[float]],
	iou_threshold: float = 0.5,
) -> tuple[float, float, float]:
    matches = 0
    used: set[int] = set()
    for pred in pred_boxes:
        best_idx = None
        best_iou = 0.0
        for idx, gt in enumerate(gt_boxes):
            if idx in used:
                continue
            iou = box_iou(pred, gt)
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
        if best_idx is not None and best_iou >= iou_threshold:
            used.add(best_idx)
            matches += 1
    precision = matches / max(1, len(pred_boxes))
    recall = matches / max(1, len(gt_boxes))
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1
