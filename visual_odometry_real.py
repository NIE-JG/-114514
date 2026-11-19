# -*- coding: utf-8 -*-
"""
真实视频单目视觉里程计实验
算法：ORB特征提取 -> BFMatcher匹配 -> 比率检验 -> RANSAC本质矩阵
      -> recoverPose恢复相对位姿 -> 累积相机轨迹

安装依赖：
    pip install opencv-python numpy matplotlib pandas

运行：
    python visual_odometry_real.py --video video_20261119_120843.mp4

说明：
1. 默认只处理视频前7.2秒，避开后段明显转向/场景变化。
2. 视频按约5帧/秒抽帧，并缩放到540×960以降低计算量。
3. 未进行相机标定，fx、fy采用近似值700像素；结果用于演示算法流程。
4. 单目recoverPose只能恢复平移方向，尺度未知，轨迹坐标不是米。
"""
import argparse
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def read_video_samples(path, end_time=7.2, sample_interval=0.2):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError("无法打开视频：" + path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = count / fps if fps else 0
    frames = []
    for t in np.arange(0, min(end_time, duration) + 1e-6, sample_interval):
        cap.set(cv2.CAP_PROP_POS_MSEC, float(t * 1000))
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (540, 960), interpolation=cv2.INTER_AREA)
        frames.append((float(t), gray))
    cap.release()
    if len(frames) < 2:
        raise RuntimeError("有效图像帧不足。")
    return frames, fps, count, duration

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="输入手机拍摄的视频")
    parser.add_argument("--output", default="vo_real_output", help="结果输出目录")
    parser.add_argument("--end-time", type=float, default=7.2)
    parser.add_argument("--interval", type=float, default=0.2)
    parser.add_argument("--fx", type=float, default=700.0)
    parser.add_argument("--fy", type=float, default=700.0)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    frames, fps, count, duration = read_video_samples(
        args.video, args.end_time, args.interval
    )
    h, w = frames[0][1].shape
    K = np.array([[args.fx, 0, w/2],
                  [0, args.fy, h/2],
                  [0, 0, 1]], dtype=np.float64)

    orb = cv2.ORB_create(nfeatures=2500, scaleFactor=1.2, nlevels=8,
                         edgeThreshold=19, fastThreshold=10)
    features = [orb.detectAndCompute(item[1], None) for item in frames]
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    R_world_camera = np.eye(3)
    camera_center = np.zeros(3)
    trajectory = [camera_center.copy()]
    records = []
    match_vis = None

    for i in range(len(frames)-1):
        kp1, des1 = features[i]
        kp2, des2 = features[i+1]
        if des1 is None or des2 is None:
            trajectory.append(camera_center.copy())
            records.append([i, frames[i][0], len(kp1), len(kp2), 0, 0, "描述子不足"])
            continue

        knn = matcher.knnMatch(des1, des2, k=2)
        good = [m for m, n in knn if m.distance < 0.75*n.distance]
        if len(good) < 8:
            trajectory.append(camera_center.copy())
            records.append([i, frames[i][0], len(kp1), len(kp2), len(good), 0, "匹配点不足"])
            continue

        p1 = np.float32([kp1[m.queryIdx].pt for m in good])
        p2 = np.float32([kp2[m.trainIdx].pt for m in good])
        E, mask = cv2.findEssentialMat(
            p1, p2, K, method=cv2.RANSAC, prob=0.999, threshold=1.2
        )
        if E is None or mask is None:
            trajectory.append(camera_center.copy())
            records.append([i, frames[i][0], len(kp1), len(kp2), len(good), 0, "本质矩阵估计失败"])
            continue
        if E.shape[0] > 3:
            E = E[:3, :]
        inliers, R, t, pose_mask = cv2.recoverPose(E, p1, p2, K, mask=mask)
        if inliers >= 8:
            camera_center = camera_center + R_world_camera @ (-R.T @ t.reshape(3))
            R_world_camera = R_world_camera @ R.T
            status = "成功"
        else:
            status = "位姿内点较少"
        trajectory.append(camera_center.copy())
        records.append([i, frames[i][0], len(kp1), len(kp2), len(good), int(inliers), status])

        if i == 0:
            match_vis = cv2.drawMatches(
                frames[i][1], kp1, frames[i+1][1], kp2, good[:100], None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )

    trajectory = np.asarray(trajectory)
    pd.DataFrame(trajectory, columns=["X", "Y", "Z"]).to_csv(
        os.path.join(args.output, "trajectory.csv"), index=False, encoding="utf-8-sig"
    )
    stats = pd.DataFrame(records, columns=[
        "相邻帧对序号", "起始时间(s)", "前帧ORB特征数", "后帧ORB特征数",
        "比率检验后匹配数", "recoverPose内点数", "状态"
    ])
    stats.to_csv(os.path.join(args.output, "match_statistics.csv"),
                 index=False, encoding="utf-8-sig")

    cv2.imwrite(os.path.join(args.output, "first_last_frames.png"),
                np.hstack([frames[0][1], frames[-1][1]]))
    if match_vis is not None:
        cv2.imwrite(os.path.join(args.output, "feature_matches.png"), match_vis)

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(trajectory[:,0], trajectory[:,1], trajectory[:,2], "-o", markersize=2)
    ax.scatter(*trajectory[0], marker="s", label="Start")
    ax.scatter(*trajectory[-1], marker="^", label="End")
    ax.set_xlabel("X (arbitrary scale)")
    ax.set_ylabel("Y (arbitrary scale)")
    ax.set_zlabel("Z (arbitrary scale)")
    ax.set_title("Monocular VO camera trajectory")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.output, "trajectory.png"), dpi=180)
    plt.close()

    print("原视频帧率：%.2f FPS，帧数：%d，时长：%.2f s" % (fps, count, duration))
    print("抽取帧数：%d，相邻帧运动估计次数：%d" % (len(frames), len(frames)-1))
    print("平均匹配数：%.1f" % stats["比率检验后匹配数"].mean())
    print("平均recoverPose内点数：%.1f" % stats["recoverPose内点数"].mean())
    print("结果目录：", os.path.abspath(args.output))
    print("注意：单目轨迹尺度未知，不能将坐标解释为米。")

if __name__ == "__main__":
    main()
