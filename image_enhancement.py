# -*- coding: utf-8 -*-
"""
图像增强实验
==========

核心算法（卷积核生成、直方图均衡化公式、灰度变换）按课件公式手写实现；
实际的滑窗卷积运算借助 scipy.ndimage 完成。
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage

matplotlib.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Noto Sans CJK JP", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

OUT_DIR = "/home/claude/exp2/outputs"
os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# 噪声模型：对应课件 g(x,y) = f(x,y) + n(x,y)
# ============================================================
def add_gaussian_noise(img, sigma=25, seed=0):
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, sigma, img.shape)
    noisy = img.astype(np.float64) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_salt_pepper_noise(img, amount=0.05, seed=0):
    rng = np.random.default_rng(seed)
    noisy = img.copy()
    mask = rng.random(img.shape)
    noisy[mask < amount / 2] = 0          # 胡椒噪声（黑点）
    noisy[mask > 1 - amount / 2] = 255    # 盐噪声（白点）
    return noisy


# ============================================================
# 均值滤波：
#   g(x,y) = 1/((2n+1)(2m+1)) * sum_{i,j} w(x+i,y+j) * ghat(x+i,y+j)
# ============================================================
def mean_filter(img, ksize=5):
    kernel = np.ones((ksize, ksize), dtype=np.float64) / (ksize * ksize)
    return ndimage.convolve(img.astype(np.float64), kernel, mode="reflect")


# ============================================================
# 高斯滤波：
#   f(x,y) = 1/(2*pi*sigma^2) * exp(-(x^2+y^2)/(2*sigma^2))
# ============================================================
def gaussian_kernel(ksize=5, sigma=1.0):
    half = ksize // 2
    x, y = np.meshgrid(np.arange(-half, half + 1), np.arange(-half, half + 1))
    kernel = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    kernel = kernel / (2 * np.pi * sigma ** 2)
    kernel = kernel / kernel.sum()  # 归一化，保证滤波后亮度不变
    return kernel


def gaussian_filter(img, ksize=5, sigma=1.0):
    kernel = gaussian_kernel(ksize, sigma)
    return ndimage.convolve(img.astype(np.float64), kernel, mode="reflect")


# ============================================================
# 中值滤波
# ============================================================
def median_filter(img, ksize=5):
    return ndimage.median_filter(img, size=ksize)


# ============================================================
# 直方图均衡化：离散公式
#   s_k = sum_{j=0}^{k} pr(rj) * (L-1) ,  pr(rj) = nj / n
# ============================================================
def histogram_equalization(img):
    L = 256
    hist, _ = np.histogram(img.flatten(), bins=L, range=(0, L))
    pr = hist / img.size
    cdf = np.cumsum(pr)
    s = np.round(cdf * (L - 1)).astype(np.uint8)  # 映射表 s_k
    equalized = s[img]
    return equalized, hist, np.histogram(equalized.flatten(), bins=L, range=(0, L))[0]


# ============================================================
# 灰度变换：
#   gamma校正: s = c * r^gamma     线性拉伸: s = k*r + b
# ============================================================
def gamma_correction(img, gamma=0.5, c=1.0):
    norm = img.astype(np.float64) / 255.0
    out = c * np.power(norm, gamma)
    return np.clip(out * 255, 0, 255).astype(np.uint8)


def to_uint8(img):
    return np.clip(img, 0, 255).astype(np.uint8)


# ============================================================
# 实验一：去噪滤波对比（标准测试图，高斯噪声 + 椒盐噪声 分别对比）
# ============================================================
def run_denoise_experiment():
    from skimage import data, color
    img = color.rgb2gray(data.chelsea())  
    img = (img * 255).astype(np.uint8)

    noisy_gauss = add_gaussian_noise(img, sigma=25)
    noisy_sp = add_salt_pepper_noise(img, amount=0.06)

    results = {}
    for name, noisy in [("高斯噪声", noisy_gauss), ("椒盐噪声", noisy_sp)]:
        mean_r = to_uint8(mean_filter(noisy, ksize=5))
        gauss_r = to_uint8(gaussian_filter(noisy, ksize=5, sigma=1.5))
        median_r = to_uint8(median_filter(noisy, ksize=5))
        results[name] = (noisy, mean_r, gauss_r, median_r)

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    col_titles = ["原图", "加噪后", "均值滤波", "高斯滤波", "中值滤波"]
    for row, (name, (noisy, mean_r, gauss_r, median_r)) in enumerate(results.items()):
        imgs = [img, noisy, mean_r, gauss_r, median_r]
        for col, (im, title) in enumerate(zip(imgs, col_titles)):
            ax = axes[row, col]
            ax.imshow(im, cmap="gray", vmin=0, vmax=255)
            ax.set_title(f"{name}-{title}", fontsize=11)
            ax.axis("off")

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "denoise_comparison.png")
    plt.savefig(out_path, dpi=120)
    plt.close()
    print("saved:", out_path)
    return out_path


# ============================================================
# 实验二：对比度增强（直方图均衡化 + gamma校正）
# ============================================================
def run_contrast_experiment():
    from skimage import data
    img = data.moon()  # 经典低对比度标准测试图

    eq_img, hist_before, hist_after = histogram_equalization(img)
    gamma_img = gamma_correction(img, gamma=0.5)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    axes[0, 0].imshow(img, cmap="gray", vmin=0, vmax=255)
    axes[0, 0].set_title("原图（低对比度）")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(eq_img, cmap="gray", vmin=0, vmax=255)
    axes[0, 1].set_title("直方图均衡化后")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(gamma_img, cmap="gray", vmin=0, vmax=255)
    axes[0, 2].set_title("gamma校正后 (γ=0.5)")
    axes[0, 2].axis("off")

    axes[1, 0].bar(range(256), hist_before, color="gray")
    axes[1, 0].set_title("原图直方图")
    axes[1, 0].set_xlim(0, 255)

    axes[1, 1].bar(range(256), hist_after, color="gray")
    axes[1, 1].set_title("均衡化后直方图")
    axes[1, 1].set_xlim(0, 255)

    gamma_hist, _ = np.histogram(gamma_img.flatten(), bins=256, range=(0, 256))
    axes[1, 2].bar(range(256), gamma_hist, color="gray")
    axes[1, 2].set_title("gamma校正后直方图")
    axes[1, 2].set_xlim(0, 255)

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "contrast_comparison.png")
    plt.savefig(out_path, dpi=120)
    plt.close()
    print("saved:", out_path)
    return out_path


# ============================================================
# 实验三：处理一张自己拍的照片
# 去噪和对比度增强分开做两组独立对比，
# 跟前面标准图的两组实验思路保持一致，方便对比。
# ============================================================
def run_own_photo(photo_path):
    from PIL import Image
    img = np.array(Image.open(photo_path).convert("L"))  # 转灰度处理，简化流程

    # ---- 去噪对比（在原图上加噪，再分别用三种滤波） ----
    noisy = add_gaussian_noise(img, sigma=20)
    mean_r = to_uint8(mean_filter(noisy, ksize=5))
    gauss_r = to_uint8(gaussian_filter(noisy, ksize=5, sigma=1.2))
    median_r = to_uint8(median_filter(noisy, ksize=5))

    fig1, axes1 = plt.subplots(1, 5, figsize=(22, 5))
    imgs1 = [img, noisy, mean_r, gauss_r, median_r]
    titles1 = ["原图", "加噪声后", "均值滤波", "高斯滤波", "中值滤波"]
    for ax, im, title in zip(axes1, imgs1, titles1):
        ax.imshow(im, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    out_path1 = os.path.join(OUT_DIR, "own_photo_denoise.png")
    plt.savefig(out_path1, dpi=120)
    plt.close()
    print("saved:", out_path1)

    # ---- 对比度增强不叠加去噪结果 ----
    eq_img, hist_before, hist_after = histogram_equalization(img)
    gamma_img = gamma_correction(img, gamma=0.7)

    fig2, axes2 = plt.subplots(1, 3, figsize=(16, 6))
    imgs2 = [img, eq_img, gamma_img]
    titles2 = ["原图", "直方图均衡化后", "gamma校正后 (γ=0.7)"]
    for ax, im, title in zip(axes2, imgs2, titles2):
        ax.imshow(im, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    out_path2 = os.path.join(OUT_DIR, "own_photo_contrast.png")
    plt.savefig(out_path2, dpi=120)
    plt.close()
    print("saved:", out_path2)

    return out_path1, out_path2


if __name__ == "__main__":
    run_denoise_experiment()
    run_contrast_experiment()
    
