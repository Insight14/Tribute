#!/usr/bin/env python3
"""
tools/extract_poses.py
Build-time pose extraction tool for Kintsugi CR7.
Samples video frames at 12fps, extracts 2D skeletal keypoints using KeypointRCNN (PyTorch),
derives 15-joint skeleton (pelvis = hip midpoint, neck = shoulder midpoint),
normalizes by torso length, smooths with Savitzky-Golay filter,
reduces to 8-12 keyframes with root trajectory, and outputs poses.generated.json.
"""

import os
import sys
import json
import argparse
import numpy as np
import cv2
from scipy.signal import savgol_filter

import torch
import torchvision
from torchvision.models.detection import keypointrcnn_resnet50_fpn, KeypointRCNN_ResNet50_FPN_Weights
from torchvision.transforms import functional as F

JOINT_NAMES = [
    'head', 'neck', 'chest', 'pelvis',
    'l_shoulder', 'l_elbow', 'l_wrist',
    'r_shoulder', 'r_elbow', 'r_wrist',
    'l_hip', 'l_knee', 'l_ankle',
    'r_hip', 'r_knee', 'r_ankle',
    'eye'
]

def init_pose_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weights = KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
    model = keypointrcnn_resnet50_fpn(weights=weights).to(device)
    model.eval()
    return model, device

def extract_clip_poses(video_path, model, device, target_fps=12):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open {video_path}")
        return None

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / src_fps
    sample_interval = max(1, int(round(src_fps / target_fps)))

    frame_idx = 0
    extracted_frames = []

    print(f"Extracting '{os.path.basename(video_path)}': {total_frames} frames @ {src_fps:.1f} fps (duration {duration:.2f}s)...")

    prev_player_box = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_interval == 0:
            t = (frame_idx / total_frames)
            orig_h, orig_w, _ = frame.shape
            
            # Downscale for ultra-fast inference (800px max dimension)
            scale_factor = 800.0 / max(orig_h, orig_w) if max(orig_h, orig_w) > 800 else 1.0
            inf_w = int(orig_w * scale_factor)
            inf_h = int(orig_h * scale_factor)
            resized_frame = cv2.resize(frame, (inf_w, inf_h))
            
            img_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            tensor = F.to_tensor(img_rgb).to(device)
            
            with torch.no_grad():
                output = model([tensor])[0]
            
            scores = output['scores'].cpu().numpy()
            boxes = output['boxes'].cpu().numpy()
            kps_all = output['keypoints'].cpu().numpy()

            if len(scores) > 0 and scores[0] > 0.25:
                best_person = 0
                if prev_player_box is not None and len(boxes) > 1:
                    prev_center = (prev_player_box[:2] + prev_player_box[2:]) * 0.5
                    min_dist = float('inf')
                    for b_idx in range(min(5, len(boxes))):
                        if scores[b_idx] < 0.25:
                            continue
                        cur_center = (boxes[b_idx][:2] + boxes[b_idx][2:]) * 0.5
                        d = np.linalg.norm(cur_center - prev_center)
                        if d < min_dist:
                            min_dist = d
                            best_person = b_idx

                prev_player_box = boxes[best_person]
                kps = kps_all[best_person]

                # Normalize landmarks to 0-1 range based on inference dimensions
                nose = kps[0, :2] / np.array([inf_w, inf_h])
                l_eye = kps[1, :2] / np.array([inf_w, inf_h])
                r_eye = kps[2, :2] / np.array([inf_w, inf_h])
                l_sh = kps[5, :2] / np.array([inf_w, inf_h])
                r_sh = kps[6, :2] / np.array([inf_w, inf_h])
                l_el = kps[7, :2] / np.array([inf_w, inf_h])
                r_el = kps[8, :2] / np.array([inf_w, inf_h])
                l_wr = kps[9, :2] / np.array([inf_w, inf_h])
                r_wr = kps[10, :2] / np.array([inf_w, inf_h])
                l_hip = kps[11, :2] / np.array([inf_w, inf_h])
                r_hip = kps[12, :2] / np.array([inf_w, inf_h])
                l_knee = kps[13, :2] / np.array([inf_w, inf_h])
                r_knee = kps[14, :2] / np.array([inf_w, inf_h])
                l_ank = kps[15, :2] / np.array([inf_w, inf_h])
                r_ank = kps[16, :2] / np.array([inf_w, inf_h])

                neck = (l_sh + r_sh) * 0.5
                pelvis = (l_hip + r_hip) * 0.5
                chest = (neck + pelvis) * 0.5
                head = nose
                eye = (l_eye + r_eye) * 0.5

                raw_joints = {
                    'head': head, 'neck': neck, 'chest': chest, 'pelvis': pelvis,
                    'l_shoulder': l_sh, 'l_elbow': l_el, 'l_wrist': l_wr,
                    'r_shoulder': r_sh, 'r_elbow': r_el, 'r_wrist': r_wr,
                    'l_hip': l_hip, 'l_knee': l_knee, 'l_ankle': l_ank,
                    'r_hip': r_hip, 'r_knee': r_knee, 'r_ankle': r_ank,
                    'eye': eye
                }

                extracted_frames.append({
                    't': t,
                    'frame_idx': frame_idx,
                    'raw_joints': raw_joints,
                    'pelvis_root': pelvis.copy()
                })

        frame_idx += 1

    cap.release()
    print(f"Extracted {len(extracted_frames)} valid pose frames from video.")
    return extracted_frames, duration

def normalize_and_smooth(extracted_frames, target_torso_length=0.26, mirror=False):
    if not extracted_frames or len(extracted_frames) < 3:
        return None

    num_frames = len(extracted_frames)
    joint_series = {name: np.zeros((num_frames, 2)) for name in JOINT_NAMES}
    roots = np.zeros((num_frames, 2))
    times = np.array([f['t'] for f in extracted_frames])

    for i, f in enumerate(extracted_frames):
        pelvis = f['raw_joints']['pelvis']
        neck = f['raw_joints']['neck']
        torso_len = np.linalg.norm(neck - pelvis)
        if torso_len < 0.01:
            torso_len = target_torso_length

        scale = target_torso_length / torso_len
        roots[i] = pelvis

        for name in JOINT_NAMES:
            rel = f['raw_joints'][name] - pelvis
            norm_rel = rel * scale
            if mirror:
                norm_rel[0] = -norm_rel[0]
            
            joint_series[name][i] = np.array([0.50, 0.48]) + norm_rel

    # Smooth with Savitzky-Golay filter
    window_len = min(7, num_frames if num_frames % 2 == 1 else num_frames - 1)
    if window_len >= 5:
        for name in JOINT_NAMES:
            joint_series[name][:, 0] = savgol_filter(joint_series[name][:, 0], window_len, 2)
            joint_series[name][:, 1] = savgol_filter(joint_series[name][:, 1], window_len, 2)
        roots[:, 0] = savgol_filter(roots[:, 0], window_len, 2)
        roots[:, 1] = savgol_filter(roots[:, 1], window_len, 2)

    # Frame-to-frame kinetic velocity to pick 8-12 keyframes
    velocities = np.zeros(num_frames)
    for i in range(1, num_frames):
        dt = max(0.001, times[i] - times[i-1])
        v_sum = 0
        for name in JOINT_NAMES:
            v_sum += np.linalg.norm(joint_series[name][i] - joint_series[name][i-1]) / dt
        velocities[i] = v_sum

    target_kfs = 10
    keyframe_indices = [0, num_frames - 1]

    step = (num_frames - 1) / (target_kfs - 1)
    for k in range(1, target_kfs - 1):
        center = int(round(k * step))
        window = range(max(1, center - 2), min(num_frames - 1, center + 3))
        best_idx = max(window, key=lambda idx: velocities[idx])
        if best_idx not in keyframe_indices:
            keyframe_indices.append(best_idx)

    keyframe_indices = sorted(list(set(keyframe_indices)))

    keyframes = []
    root_keyframes = []

    t_start = times[keyframe_indices[0]]
    t_end = times[keyframe_indices[-1]]
    t_range = max(0.001, t_end - t_start)

    for idx in keyframe_indices:
        norm_t = float((times[idx] - t_start) / t_range)
        frame_joints = {}
        for name in JOINT_NAMES:
            frame_joints[name] = {
                'x': round(float(joint_series[name][idx, 0]), 4),
                'y': round(float(joint_series[name][idx, 1]), 4)
            }
        keyframes.append({
            't': round(norm_t, 3),
            'joints': frame_joints
        })
        root_keyframes.append([
            round(float(roots[idx, 0]), 4),
            round(float(roots[idx, 1]), 4)
        ])

    return {
        'keyframes': keyframes,
        'root': root_keyframes
    }

def main():
    parser = argparse.ArgumentParser(description="Extract 2D pose keyframes from reference videos for Kintsugi CR7")
    parser.add_argument("--clip", type=str, default="bicycleKick", help="Clip name (e.g. bicycleKick)")
    parser.add_argument("--ref_dir", type=str, default="/Users/sourish/Desktop/kintsugi-cr7/reference", help="Directory containing reference clips")
    parser.add_argument("--out", type=str, default="/Users/sourish/Desktop/kintsugi-cr7/poses.generated.json", help="Output JSON path")
    parser.add_argument("--mirror", action="store_true", help="Mirror pose horizontally")
    args = parser.parse_args()

    model, device = init_pose_model()

    video_path = None
    for ext in ['.mov', '.mp4', '.mkv', '.webm']:
        cand = os.path.join(args.ref_dir, args.clip + ext)
        if os.path.exists(cand):
            video_path = cand
            break

    if not video_path:
        print(f"Error: Video for clip '{args.clip}' not found in {args.ref_dir}")
        sys.exit(1)

    extracted_data, duration = extract_clip_poses(video_path, model, device, target_fps=12)
    if not extracted_data:
        print("Extraction failed.")
        sys.exit(1)

    result = normalize_and_smooth(extracted_data, mirror=args.mirror)
    if not result:
        print("Normalization failed.")
        sys.exit(1)

    poses_db = {}
    if os.path.exists(args.out):
        try:
            with open(args.out, 'r') as f:
                poses_db = json.load(f)
        except Exception:
            poses_db = {}

    poses_db[args.clip] = {
        'duration': round(float(duration), 2),
        'root': result['root'],
        'keyframes': result['keyframes']
    }

    with open(args.out, 'w') as f:
        json.dump(poses_db, f, indent=2)

    print(f"\n=======================================================")
    print(f"Pose extraction complete! Saved to: {args.out}")
    print(f"Clip: '{args.clip}' | Duration: {duration:.2f}s | Keyframes: {len(result['keyframes'])}")
    print(f"=======================================================")

if __name__ == '__main__':
    main()
