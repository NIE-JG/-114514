# -*- coding: utf-8 -*-
"""
张正友标定法 

流程
    1. 求单应矩阵 H（每张图一个）
    2. 建立内参求解方程 (Vb = 0)
    3. SVD 求解 b
    4. 由 b 求内参 A，再由 A 和各 H 求每张图外参 (R, t)
    5. 极大似然估计：把上面结果作为初值
    6. 非线性优化（Levenberg-Marquardt）联合优化内参、畸变系数、所有外参

"""

import os
import glob
import numpy as np
import cv2
from scipy.optimize import least_squares

# ===================== 配置区 =====================
IMAGE_DIR = "./calib_images"      # 存放"标定用"照片的文件夹（多角度多张，用来求内参）
IMAGE_EXT = "*.jpg"               # 图片后缀，按需改成 *.png / *.jpeg
CHECKERBOARD = (9, 6)             # 棋盘格内角点个数 (宽方向个数, 高方向个数)，注意是"内角点"不是格子数
SQUARE_SIZE = 20.0                # 每个格子的物理边长，单位 mm（不影响角度计算，只影响平移的物理单位）

# 下面这两张是"移动前/移动后"专用的两张图，跟标定图分开放
POSE_IMAGE_DIR = "./pose_images"
POSE_IMAGE_BEFORE = "before.jpg"
POSE_IMAGE_AFTER = "after.jpg"
# ============================================================================


# ---------------------------------------------------------------------------
# 第 0 步：读取图片 + 检测棋盘格角点
# ---------------------------------------------------------------------------
def detect_corners(image_dir, ext, checkerboard):
    """返回：图片文件名列表、每张图的像素角点列表(N,2)、图像尺寸(w,h)"""
    files = sorted(glob.glob(os.path.join(image_dir, ext)))
    if len(files) == 0:
        raise FileNotFoundError(f"在 {image_dir} 下没找到 {ext} 图片，请检查路径")

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    img_points = []   # 每张图对应的角点像素坐标 (N,2)
    used_files = []
    img_size = None

    for f in files:
        img = cv2.imread(f)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img_size is None:
            img_size = (gray.shape[1], gray.shape[0])  # (w, h)

        found, corners = cv2.findChessboardCorners(
            gray, checkerboard,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        if not found:
            print(f"[跳过] 未检测到角点: {f}")
            continue

        corners_sub = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        img_points.append(corners_sub.reshape(-1, 2))
        used_files.append(f)

    if len(img_points) < 3:
        raise RuntimeError("有效标定图片少于 3 张，至少需要 3~4 张不同角度的图才能求解内参")

    print(f"共检测到 {len(img_points)} 张有效标定图（总共尝试 {len(files)} 张）")
    return used_files, img_points, img_size


def build_world_points(checkerboard, square_size):
    """棋盘格在自身坐标系下的 3D 点，Z=0 平面，单位 mm。返回 (N,3)"""
    w, h = checkerboard
    obj = np.zeros((w * h, 3), dtype=np.float64)
    grid = np.mgrid[0:w, 0:h].T.reshape(-1, 2)
    obj[:, :2] = grid * square_size
    return obj


# ---------------------------------------------------------------------------
# 第 1 步：求单应矩阵 H （每张图：世界平面 Z=0 -> 图像）
# ---------------------------------------------------------------------------
def _normalize_points(pts):
    """DLT 数值稳定性所需的归一化：平移到质心为原点，均方距离为 sqrt(2)"""
    pts = np.asarray(pts, dtype=np.float64)
    centroid = pts.mean(axis=0)
    shifted = pts - centroid
    mean_dist = np.mean(np.sqrt(np.sum(shifted ** 2, axis=1)))
    scale = np.sqrt(2) / (mean_dist + 1e-12)
    T = np.array([
        [scale, 0, -scale * centroid[0]],
        [0, scale, -scale * centroid[1]],
        [0, 0, 1]
    ])
    pts_h = np.hstack([pts, np.ones((pts.shape[0], 1))])
    pts_norm = (T @ pts_h.T).T
    return pts_norm[:, :2], T


def compute_homography(obj_pts_xy, img_pts):
    """
    obj_pts_xy: (N,2) 世界坐标系下棋盘格点 (x_w, y_w)，Z=0 已忽略
    img_pts:    (N,2) 对应的像素坐标 (u, v)
    返回 3x3 单应矩阵 H，满足  z*[u,v,1]^T = H * [x_w, y_w, 1]^T
    对应 PPT: z_w[u v 1]^T = H_3x3 [x_w y_w 1]^T
    """
    obj_n, T_obj = _normalize_points(obj_pts_xy)
    img_n, T_img = _normalize_points(img_pts)

    N = obj_n.shape[0]
    A = np.zeros((2 * N, 9))
    for i in range(N):
        X, Y = obj_n[i]
        u, v = img_n[i]
        A[2 * i]     = [-X, -Y, -1, 0, 0, 0, u * X, u * Y, u]
        A[2 * i + 1] = [0, 0, 0, -X, -Y, -1, v * X, v * Y, v]

    # 齐次方程 Ah=0 的解是 A 的最小奇异值对应的右奇异向量（对应 PPT: A=UΣV^T, 取 V 最后一列）
    _, _, Vt = np.linalg.svd(A)
    H_n = Vt[-1].reshape(3, 3)

    # 反归一化: H = T_img^-1 * H_n * T_obj
    H = np.linalg.inv(T_img) @ H_n @ T_obj
    H = H / H[2, 2]
    return H


# ---------------------------------------------------------------------------
# 第 2 步：建立内参约束方程 Vb = 0
# ---------------------------------------------------------------------------
def _v_ij(H, i, j):
    """
    对应 PPT: V_ij = [h_i1 h_j1, h_i1 h_j2+h_i2 h_j1, h_i2 h_j2,
                       h_i3 h_j1+h_i1 h_j3, h_i3 h_j2+h_i2 h_j3, h_i3 h_j3]^T
    这里 H 的列向量记作 h1=H[:,0], h2=H[:,1], h3=H[:,2]
    i, j 取 1 或 2 (对应 h1, h2)
    """
    h = [None, H[:, 0], H[:, 1], H[:, 2]]  # 1-based
    hi, hj = h[i], h[j]
    return np.array([
        hi[0] * hj[0],
        hi[0] * hj[1] + hi[1] * hj[0],
        hi[1] * hj[1],
        hi[2] * hj[0] + hi[0] * hj[2],
        hi[2] * hj[1] + hi[1] * hj[2],
        hi[2] * hj[2],
    ])


def solve_intrinsics(H_list):
    """
    堆叠所有图的 v12^T b=0 和 (v11-v22)^T b=0，构成 V_{2n x 6} b = 0
    SVD 求 b，再按 PPT 闭式公式提取内参 A
    """
    V = []
    for H in H_list:
        v12 = _v_ij(H, 1, 2)
        v11 = _v_ij(H, 1, 1)
        v22 = _v_ij(H, 2, 2)
        V.append(v12)
        V.append(v11 - v22)
    V = np.array(V)  # (2n, 6)

    _, _, Vt = np.linalg.svd(V)
    b = Vt[-1]  # [B11, B12, B22, B13, B23, B33]
    B11, B12, B22, B13, B23, B33 = b

    # 闭式解（PPT Ch2 §2.1 "Homography has 8 DOFs..." 那页公式）
    v0 = (B12 * B13 - B11 * B23) / (B11 * B22 - B12 ** 2)
    lam = B33 - (B13 ** 2 + v0 * (B12 * B13 - B11 * B23)) / B11
    alpha = np.sqrt(lam / B11)                      # kx
    beta = np.sqrt(lam * B11 / (B11 * B22 - B12 ** 2))  # ky
    gamma = -B12 * alpha ** 2 * beta / lam
    u0 = gamma * v0 / beta - B13 * alpha ** 2 / lam

    A = np.array([
        [alpha, gamma, u0],
        [0,     beta,  v0],
        [0,     0,     1]
    ])
    return A


# ---------------------------------------------------------------------------
# 第 3 步：由内参 A 和每张图的 H 求外参 (R, t)
# ---------------------------------------------------------------------------
def solve_extrinsics(A, H_list):
    A_inv = np.linalg.inv(A)
    Rs, ts = [], []
    for H in H_list:
        h1, h2, h3 = H[:, 0], H[:, 1], H[:, 2]
        lam = 1.0 / np.linalg.norm(A_inv @ h1)
        r1 = lam * (A_inv @ h1)
        r2 = lam * (A_inv @ h2)
        r3 = np.cross(r1, r2)
        t = lam * (A_inv @ h3)

        R = np.column_stack([r1, r2, r3])
        # 噪声导致 R 不严格满足正交，用 SVD 求最接近的旋转矩阵（PPT: "Best rotation matrix
        # can be obtained through SVD"）
        U, _, Vt = np.linalg.svd(R)
        R_ortho = U @ Vt
        if np.linalg.det(R_ortho) < 0:   # 保证是旋转矩阵而不是反射
            U[:, -1] *= -1
            R_ortho = U @ Vt

        Rs.append(R_ortho)
        ts.append(t)
    return Rs, ts


# ---------------------------------------------------------------------------
# 第 4 / 5 步：极大似然估计 + 非线性优化（Levenberg-Marquardt）
# 同时优化：内参(5) + 径向畸变(k1,k2) + 每张图外参(6: 旋转向量3+平移3)
# ---------------------------------------------------------------------------
def _project_points(obj_pts, rvec, tvec, A, k1, k2):
    """用当前参数把世界坐标投影到像素坐标，含径向畸变（对应 PPT Ch2 畸变模型）"""
    R, _ = cv2.Rodrigues(rvec)
    Xc = (R @ obj_pts.T).T + tvec  # (N,3) 相机坐标系下坐标

    x = Xc[:, 0] / Xc[:, 2]
    y = Xc[:, 1] / Xc[:, 2]
    r2 = x ** 2 + y ** 2

    x_d = x * (1 + k1 * r2 + k2 * r2 ** 2)
    y_d = y * (1 + k1 * r2 + k2 * r2 ** 2)

    fx, gamma_s, u0 = A[0, 0], A[0, 1], A[0, 2]
    fy, v0 = A[1, 1], A[1, 2]

    u = fx * x_d + gamma_s * y_d + u0
    v = fy * y_d + v0
    return np.column_stack([u, v])


def _pack_params(A, k1, k2, Rs, ts):
    fx, gamma_s, u0 = A[0, 0], A[0, 1], A[0, 2]
    fy, v0 = A[1, 1], A[1, 2]
    intr = [fx, gamma_s, u0, fy, v0, k1, k2]
    extr = []
    for R, t in zip(Rs, ts):
        rvec, _ = cv2.Rodrigues(R)
        extr.extend(rvec.flatten().tolist())
        extr.extend(t.tolist())
    return np.array(intr + extr, dtype=np.float64)


def _unpack_params(params, n_images):
    fx, gamma_s, u0, fy, v0, k1, k2 = params[:7]
    A = np.array([[fx, gamma_s, u0], [0, fy, v0], [0, 0, 1]])
    rest = params[7:].reshape(n_images, 6)
    rvecs = rest[:, :3]
    tvecs = rest[:, 3:]
    return A, k1, k2, rvecs, tvecs


def refine_calibration(obj_pts, img_points, A0, Rs0, ts0):
    """
    对应 PPT Ch2 §2.1 "4) Maximum-likelihood estimation" +
    "Basic numerical optimization" 里的 Levenberg-Marquardt
    用 scipy.optimize.least_squares(method='lm') 实现阻尼高斯牛顿
    """
    n_images = len(img_points)
    x0 = _pack_params(A0, 0.0, 0.0, Rs0, ts0)

    def residuals(params):
        A, k1, k2, rvecs, tvecs = _unpack_params(params, n_images)
        res = []
        for i in range(n_images):
            proj = _project_points(obj_pts, rvecs[i], tvecs[i], A, k1, k2)
            res.append((proj - img_points[i]).flatten())
        return np.concatenate(res)

    result = least_squares(residuals, x0, method='lm', max_nfev=2000)

    A, k1, k2, rvecs, tvecs = _unpack_params(result.x, n_images)
    Rs = [cv2.Rodrigues(rv)[0] for rv in rvecs]
    ts = [tv for tv in tvecs]

    rmse = np.sqrt(np.mean(result.fun ** 2))
    print(f"非线性优化后重投影 RMSE = {rmse:.4f} 像素")
    return A, k1, k2, Rs, ts


# ---------------------------------------------------------------------------
# 第 6a 步：单独用 solvePnP 求 pose_images 里那两张图各自的外参
# ---------------------------------------------------------------------------
def detect_single_image_corners(path, checkerboard):
    """检测单张图片的棋盘格角点，返回 (N,2) 像素坐标；检测失败返回 None"""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"读取不到图片: {path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    found, corners = cv2.findChessboardCorners(
        gray, checkerboard,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    if not found:
        return None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners_sub = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return corners_sub.reshape(-1, 2)


def solve_pose_with_pnp(obj_pts, img_pts_2d, A, k1, k2):
    """
    内参、畸变已知的情况下，单张图用 solvePnP 求外参 (R,t)
    """
    dist_coeffs = np.array([k1, k2, 0, 0, 0], dtype=np.float64)  # [k1,k2,p1,p2,k3]
    ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts_2d, A, dist_coeffs)
    if not ok:
        raise RuntimeError("solvePnP 求解失败，检查角点是否检测正确")
    R, _ = cv2.Rodrigues(rvec)
    t = tvec.flatten()
    return R, t


def compute_pose_change(A, k1, k2, checkerboard, square_size,
                         pose_dir, before_name, after_name):
    """读取 pose_images 里的 before/after 两张图，各自求外参，再算相对位移和转角"""
    obj_pts = build_world_points(checkerboard, square_size)

    before_path = os.path.join(pose_dir, before_name)
    after_path = os.path.join(pose_dir, after_name)

    corners_before = detect_single_image_corners(before_path, checkerboard)
    corners_after = detect_single_image_corners(after_path, checkerboard)

    if corners_before is None:
        raise RuntimeError(f"在 {before_path} 里没检测到棋盘格角点")
    if corners_after is None:
        raise RuntimeError(f"在 {after_path} 里没检测到棋盘格角点")

    R_before, t_before = solve_pose_with_pnp(obj_pts, corners_before, A, k1, k2)
    R_after, t_after = solve_pose_with_pnp(obj_pts, corners_after, A, k1, k2)

    disp, angle, R_rel, t_rel, displacement_vec, rot_axis = compute_relative_pose(
        R_before, t_before, R_after, t_after
    )
    print(f"\n===== 移动前后位姿变化 ({before_name} -> {after_name}) =====")
    print(f"相机位移: {disp:.2f} mm")
    print(f"  位移向量 (棋盘坐标系下 X,Y,Z 方向分量, mm): "
          f"[{displacement_vec[0]:.2f}, {displacement_vec[1]:.2f}, {displacement_vec[2]:.2f}]")
    print(f"相机旋转角度: {angle:.2f} 度")
    print(f"  旋转轴方向 (单位向量): "
          f"[{rot_axis[0]:.3f}, {rot_axis[1]:.3f}, {rot_axis[2]:.3f}]")
    return disp, angle, R_rel, t_rel


# ---------------------------------------------------------------------------
# 第 6b 步：两次拍摄之间的相对位移 / 转角（通用函数，被上面调用）
# ---------------------------------------------------------------------------
def compute_relative_pose(R_i, t_i, R_j, t_j):
    """
    (R_i, t_i), (R_j, t_j)：分别是第 i、j 张图的外参，满足 X_cam = R X_world + t
    也就是标定板固定不动，相机在两次拍摄间发生了运动。

    返回：
        camera_displacement_mm : 相机光心在世界(标定板)坐标系下的位移量
        rotation_angle_deg     : 相机姿态的相对旋转角(绕某轴转过的角度)
        R_rel, t_rel            : 相对位姿 (从姿态 i 到姿态 j)
    """
    # 相机光心在世界坐标系下的位置: C = -R^T t
    C_i = -R_i.T @ t_i
    C_j = -R_j.T @ t_j
    camera_displacement = np.linalg.norm(C_j - C_i)

    # 相对旋转：R_rel 满足 R_j = R_rel @ R_i
    R_rel = R_j @ R_i.T
    t_rel = t_j - R_rel @ t_i

    rvec_rel, _ = cv2.Rodrigues(R_rel)
    angle_deg = np.degrees(np.linalg.norm(rvec_rel))
    axis = rvec_rel.flatten() / (np.linalg.norm(rvec_rel) + 1e-12)   # 旋转轴（单位向量）

    return camera_displacement, angle_deg, R_rel, t_rel, (C_j - C_i), axis


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    files, img_points, img_size = detect_corners(IMAGE_DIR, IMAGE_EXT, CHECKERBOARD)
    obj_pts = build_world_points(CHECKERBOARD, SQUARE_SIZE)   # (N,3)
    obj_xy = obj_pts[:, :2]                                    # H 只用到平面 (x_w,y_w)

    # ---- 1. 每张图求 H ----
    H_list = [compute_homography(obj_xy, pts) for pts in img_points]

    # ---- 2+3. 求内参 A（闭式解）----
    A0 = solve_intrinsics(H_list)
    print("闭式解内参矩阵 A0 =\n", A0)

    # ---- 4. 求每张图外参 ----
    Rs0, ts0 = solve_extrinsics(A0, H_list)

    # ---- 5. 极大似然估计 / 非线性优化，联合精修全部参数 ----
    A, k1, k2, Rs, ts = refine_calibration(obj_pts, img_points, A0, Rs0, ts0)

    print("\n优化后内参矩阵 A =\n", A)
    print(f"径向畸变系数: k1={k1:.6f}, k2={k2:.6f}")

    # ---- 6. 用 pose_images 里那两张图（移动前/移动后）单独求外参并算相对位姿 ----
    if os.path.isdir(POSE_IMAGE_DIR):
        compute_pose_change(
            A, k1, k2, CHECKERBOARD, SQUARE_SIZE,
            POSE_IMAGE_DIR, POSE_IMAGE_BEFORE, POSE_IMAGE_AFTER
        )
    else:
        print(f"\n（未找到 {POSE_IMAGE_DIR} 文件夹，跳过位移/转角计算。"
              f"把 before.jpg / after.jpg 放进去再跑）")

    return A, k1, k2, Rs, ts, files


if __name__ == "__main__":
    main()
