# -*- coding: utf-8 -*-
"""
矩特征实验
==========
形状绘制用 PIL，连通域分割用 scipy.ndimage.label。
"""

import os
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Noto Sans CJK JP", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

OUT_DIR = "/home/claude/exp3/outputs"
os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# 形状定义：闪电、月牙、带烟囱的房子、心形，坐标相对形状中心
# ============================================================
def _crescent():
    """月牙：大圆弧(外侧) + 小圆弧(内侧) 拼成的多边形"""
    R, r, d = 55.0, 45.0, 25.0
    x_int = (d ** 2 + R ** 2 - r ** 2) / (2 * d)
    y_int = np.sqrt(R ** 2 - x_int ** 2)
    a0 = np.arctan2(-y_int, x_int)
    outer = [(R * np.cos(a), R * np.sin(a)) for a in np.linspace(a0, a0 - (2 * np.pi - 2 * abs(a0)), 40)]
    b0 = np.arctan2(y_int, x_int - d)
    b1 = 2 * np.pi - b0
    inner = [(d + r * np.cos(a), r * np.sin(a)) for a in np.linspace(b0, b1, 40)]
    return outer + inner


def _heart():
    t = np.linspace(0, 2 * np.pi, 80, endpoint=False)
    x = 16 * np.sin(t) ** 3
    y = -(13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t))
    return [(3.4 * a, 3.4 * b) for a, b in zip(x, y)]


BASE_SHAPES = {
    "闪电": [(15, -60), (-25, 5), (-2, 5), (-15, 60), (30, -10), (8, -10), (28, -60)],
    "月牙": _crescent(),
    "房子": [(-50, 40), (-50, -5), (0, -45), (22, -27), (22, -55), (38, -55),
             (38, -14), (50, -5), (50, 40)],
    "心形": _heart(),
}
SHAPE_NAMES = list(BASE_SHAPES.keys())


def make_shape_image(shape_name, canvas_size=256, center=None,
                      angle_deg=0.0, scale=1.0, translate=(0, 0)):
    """按给定旋转角度、缩放比例、平移量，在画布上画出形状，返回二值numpy数组(0/1)"""
    if center is None:
        center = (canvas_size // 2, canvas_size // 2)
    cx, cy = center[0] + translate[0], center[1] + translate[1]

    theta = np.radians(angle_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    transformed = []
    for x, y in BASE_SHAPES[shape_name]:
        xr = x * cos_t - y * sin_t
        yr = x * sin_t + y * cos_t
        transformed.append((xr * scale + cx, yr * scale + cy))

    img = Image.new("L", (canvas_size, canvas_size), 0)
    ImageDraw.Draw(img).polygon(transformed, fill=255)
    return (np.array(img) > 0).astype(np.float64)


# ============================================================
# 矩计算
# ============================================================
def raw_moment(img, p, q):
    """m_{p,q} = sum_x sum_y x^p * y^q * f(x,y)"""
    ys, xs = np.nonzero(img)
    return np.sum((xs.astype(np.float64) ** p) * (ys.astype(np.float64) ** q))


def central_moment(img, p, q, cx, cy):
    """mu_{p,q} = sum (x-cx)^p * (y-cy)^q * f(x,y)"""
    ys, xs = np.nonzero(img)
    return np.sum(((xs - cx) ** p) * ((ys - cy) ** q))


def hu_moments(img):
    """
    原始矩 -> 质心 -> 中心矩 -> 归一化中心矩 -> Hu的7个不变矩
      m00 = 面积；cx = m10/m00, cy = m01/m00
      eta_pq = mu_pq / m00^((p+q)/2+1)
    """
    m00 = raw_moment(img, 0, 0)
    cx = raw_moment(img, 1, 0) / m00
    cy = raw_moment(img, 0, 1) / m00

    eta = {}
    for p in range(4):
        for q in range(4 - p):
            mu = central_moment(img, p, q, cx, cy)
            eta[(p, q)] = mu / (m00 ** ((p + q) / 2.0 + 1.0))

    n20, n02, n11 = eta[(2, 0)], eta[(0, 2)], eta[(1, 1)]
    n30, n03, n21, n12 = eta[(3, 0)], eta[(0, 3)], eta[(2, 1)], eta[(1, 2)]

    phi1 = n20 + n02
    phi2 = (n20 - n02) ** 2 + 4 * n11 ** 2
    phi3 = (n30 - 3 * n12) ** 2 + (3 * n21 - n03) ** 2
    phi4 = (n30 + n12) ** 2 + (n21 + n03) ** 2
    phi5 = (n30 - 3 * n12) * (n30 + n12) * ((n30 + n12) ** 2 - 3 * (n21 + n03) ** 2) + \
           (3 * n21 - n03) * (n21 + n03) * (3 * (n30 + n12) ** 2 - (n21 + n03) ** 2)
    phi6 = (n20 - n02) * ((n30 + n12) ** 2 - (n21 + n03) ** 2) + \
           4 * n11 * (n30 + n12) * (n21 + n03)
    phi7 = (3 * n21 - n03) * (n30 + n12) * ((n30 + n12) ** 2 - 3 * (n21 + n03) ** 2) - \
           (n30 - 3 * n12) * (n21 + n03) * (3 * (n30 + n12) ** 2 - (n21 + n03) ** 2)

    return np.array([phi1, phi2, phi3, phi4, phi5, phi6, phi7])


def log_hu(hu):
    """Hu矩数值跨度很大，取 -sign(h)*log10(|h|) 压缩后再比较"""
    return -np.sign(hu) * np.log10(np.abs(hu) + 1e-30)


# ============================================================
# 实验一：不变性测试
# ============================================================
def run_invariance_test():
    shape = "闪电"
    configs = [
        ("原始", 0, 1.0, (0, 0)),
        ("旋转30°", 30, 1.0, (0, 0)),
        ("旋转90°", 90, 1.0, (0, 0)),
        ("旋转180°", 180, 1.0, (0, 0)),
        ("旋转270°", 270, 1.0, (0, 0)),
        ("平移(+30,+20)", 0, 1.0, (30, 20)),
        ("平移(-25,+40)", 0, 1.0, (-25, 40)),
        ("旋转60°+平移(20,-15)", 60, 1.0, (20, -15)),
        ("缩放0.7倍", 0, 0.7, (0, 0)),
        ("缩放1.4倍", 0, 1.4, (0, 0)),
    ]

    base_lhu = log_hu(hu_moments(make_shape_image(shape)))
    rows, imgs_for_plot = [], []
    for name, angle, scale, trans in configs:
        img = make_shape_image(shape, angle_deg=angle, scale=scale, translate=trans)
        lhu = log_hu(hu_moments(img))
        # 前6个量的平均相对偏差（φ7单独看，因为它数值上特别不稳定）
        dev6 = np.mean(np.abs(lhu[:6] - base_lhu[:6]) / np.abs(base_lhu[:6])) * 100
        rows.append((name, lhu, dev6))
        imgs_for_plot.append((name, img))

    print(f"\n==== 不变性测试（形状：{shape}） ====")
    print(f"{'变换':<22s}" + "".join([f"φ{i+1:<9d}" for i in range(7)]) + "φ1~φ6平均相对偏差")
    for name, lhu, dev6 in rows:
        vals = "".join([f"{v:<10.3f}" for v in lhu])
        print(f"{name:<22s}{vals}{dev6:.2f}%")

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    for ax, (name, img) in zip(axes.flat, imgs_for_plot):
        ax.imshow(img, cmap="gray")
        ax.set_title(name, fontsize=11)
        ax.axis("off")
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "invariance_test.png")
    plt.savefig(out_path, dpi=120)
    plt.close()
    print("saved:", out_path)
    return rows, out_path


# ============================================================
# 实验二：分类
# ============================================================
def generate_dataset(n_per_class=20, seed=0):
    rng = np.random.default_rng(seed)
    X, y, imgs = [], [], []
    for shape in SHAPE_NAMES:
        for _ in range(n_per_class):
            img = make_shape_image(
                shape,
                angle_deg=rng.uniform(0, 360),
                scale=rng.uniform(0.7, 1.3),
                translate=(rng.uniform(-25, 25), rng.uniform(-25, 25)),
            )
            X.append(log_hu(hu_moments(img)))
            y.append(shape)
            imgs.append(img)
    return np.array(X), np.array(y), imgs


def nearest_centroid_classify(X_train, y_train, X_test):
    """
    每类取训练样本均值作为模板，测试样本归到欧式距离最近的模板。
    7个log-Hu分量数值范围差异大（尤其φ7波动特别大），直接算距离会被它主导，
    所以先按训练集统计量做 z-score 标准化，再在标准化空间里算距离。
    """
    classes = sorted(set(y_train))
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-9
    Xtr = (X_train - mean) / std
    Xte = (X_test - mean) / std

    centroids = {c: Xtr[y_train == c].mean(axis=0) for c in classes}
    preds = []
    for x in Xte:
        d = {c: np.linalg.norm(x - centroids[c]) for c in classes}
        preds.append(min(d, key=d.get))
    return np.array(preds), centroids, {"mean": mean, "std": std}


def run_classification_experiment():
    X_train, y_train, _ = generate_dataset(n_per_class=20, seed=0)
    X_test, y_test, test_imgs = generate_dataset(n_per_class=8, seed=1)

    preds, centroids, norm_params = nearest_centroid_classify(X_train, y_train, X_test)
    acc = np.mean(preds == y_test)
    print("\n==== 分类实验 ====")
    print(f"训练集: {len(y_train)}个样本（每类随机旋转/平移/缩放）")
    print(f"测试集: {len(y_test)}个样本")
    print(f"分类准确率: {acc*100:.1f}%")

    classes = SHAPE_NAMES
    confusion = np.zeros((len(classes), len(classes)), dtype=int)
    for t, p in zip(y_test, preds):
        confusion[classes.index(t), classes.index(p)] += 1

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(confusion, cmap="Blues")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    ax.set_xlabel("预测类别")
    ax.set_ylabel("真实类别")
    ax.set_title(f"混淆矩阵（准确率 {acc*100:.1f}%）")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, confusion[i, j], ha="center", va="center",
                    color="white" if confusion[i, j] > confusion.max() / 2 else "black")
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "classification_confusion.png")
    plt.savefig(out_path, dpi=120)
    plt.close()
    print("saved:", out_path)
    return acc, centroids, norm_params, out_path


# ============================================================
# 实验三：综合场景
# ============================================================
def run_composite_scene_test(centroids, norm_params):
    canvas_size = 500
    rng = np.random.default_rng(42)
    canvas = np.zeros((canvas_size, canvas_size), dtype=np.float64)
    ground_truth = []

    grid_positions = [(130, 130), (370, 130), (130, 370), (370, 370)]
    shapes_to_place = SHAPE_NAMES.copy()
    rng.shuffle(shapes_to_place)

    for shape, (gx, gy) in zip(shapes_to_place, grid_positions):
        cx, cy = gx + rng.uniform(-15, 15), gy + rng.uniform(-15, 15)
        single = make_shape_image(shape, canvas_size=canvas_size, center=(cx, cy),
                                  angle_deg=rng.uniform(0, 360), scale=rng.uniform(0.8, 1.1))
        canvas = np.maximum(canvas, single)
        ground_truth.append((shape, cx, cy))

    # 凹形状（星形、L形）栅格化后凹角处可能只斜着相连，用8邻域才会被当成一个连通区域
    labeled, num_features = ndimage.label(canvas, structure=np.ones((3, 3), dtype=int))
    print("\n==== 综合场景测试 ====")
    print(f"摆放了 {len(ground_truth)} 个形状，分割出 {num_features} 个连通区域")

    results = []
    for region_id in range(1, num_features + 1):
        mask = (labeled == region_id).astype(np.float64)
        lhu = (log_hu(hu_moments(mask)) - norm_params["mean"]) / norm_params["std"]
        d = {c: np.linalg.norm(lhu - centroids[c]) for c in centroids}
        pred = min(d, key=d.get)
        ys, xs = np.nonzero(mask)
        rcx, rcy = xs.mean(), ys.mean()
        true_label = min(ground_truth, key=lambda g: (g[1] - rcx) ** 2 + (g[2] - rcy) ** 2)[0]
        results.append((region_id, true_label, pred, rcx, rcy))
        print(f"区域{region_id}: 真实={true_label}, 识别={pred}, "
              f"{'正确' if true_label == pred else '错误'}")

    correct = sum(1 for r in results if r[1] == r[2])
    print(f"综合场景识别准确率: {correct}/{len(results)}")

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(canvas, cmap="gray")
    for _, true_label, pred, cx, cy in results:
        color = "lime" if true_label == pred else "red"
        ax.text(cx, cy, pred, color=color, fontsize=13, ha="center", va="center",
                weight="bold", bbox=dict(facecolor="black", alpha=0.5, pad=1))
    ax.set_title(f"综合场景识别（绿=正确，红=错误）{correct}/{len(results)}")
    ax.axis("off")
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "composite_scene.png")
    plt.savefig(out_path, dpi=120)
    plt.close()
    print("saved:", out_path)
    return results, out_path


if __name__ == "__main__":
    run_invariance_test()
    acc, centroids, norm_params, _ = run_classification_experiment()
    run_composite_scene_test(centroids, norm_params)
