#!/usr/bin/env python3
"""
qc_checker.py
Công cụ dòng lệnh (CLI) kiểm tra tính hợp lệ hình học của nhãn 3D Point Cloud Bounding Box
và điểm mốc chuyển động cơ thể (2D Body Keypoints) trong quy trình QC dữ liệu huấn luyện AI.
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

CLASS_DIMENSION_BOUNDS = {
    "Car": {"length": (2.5, 7.0), "width": (1.2, 2.8), "height": (1.1, 2.5)},
    "Pedestrian": {"length": (0.2, 1.5), "width": (0.2, 1.5), "height": (0.8, 2.2)},
    "RobotArm": {"length": (0.2, 2.0), "width": (0.1, 1.5), "height": (0.2, 2.0)},
}


def validate_3d_box(box: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    required_keys = ["id", "class_name", "center", "dimensions", "yaw", "occlusion_level"]
    for key in required_keys:
        if key not in box:
            return [f"Thiếu trường bắt buộc: '{key}'"]

    dims = box["dimensions"]
    length = dims.get("length", 0)
    width = dims.get("width", 0)
    height = dims.get("height", 0)

    if length <= 0 or width <= 0 or height <= 0:
        errors.append(f"Kích thước hộp bao 3D phải dương (L={length}, W={width}, H={height})")

    cls_name = box["class_name"]
    if cls_name in CLASS_DIMENSION_BOUNDS:
        bounds = CLASS_DIMENSION_BOUNDS[cls_name]
        for dim_name, val in [("length", length), ("width", width), ("height", height)]:
            lo, hi = bounds[dim_name]
            if val > 0 and not (lo <= val <= hi):
                errors.append(f"{cls_name} có {dim_name}={val}m vượt ngưỡng vật lý [{lo}, {hi}]m")

    yaw = box["yaw"]
    if not (-math.pi <= yaw <= math.pi):
        errors.append(f"Góc hướng Yaw={yaw} rad vượt ngoài khoảng chuẩn [-pi, pi]")

    occ = box["occlusion_level"]
    if occ not in (0, 1, 2):
        errors.append(f"Mức độ che khuất occlusion_level={occ} không hợp lệ (chỉ nhận 0, 1, 2)")

    return errors


def validate_pose_keypoints(person: Dict[str, Any], image_w: int, image_h: int) -> List[str]:
    errors: List[str] = []
    kps = person.get("keypoints", [])
    if len(kps) != 17:
        return [f"COCO Pose yêu cầu đủ 17 keypoints, hiện có {len(kps)}"]

    visible_count = 0
    for idx, kp in enumerate(kps):
        if len(kp) != 3:
            errors.append(f"Keypoint #{idx} sai định dạng [x, y, v]")
            continue
        x, y, v = kp
        if v not in (0, 1, 2):
            errors.append(f"Keypoint #{idx} có cờ hiển thị v={v} không hợp lệ")
        if v > 0:
            visible_count += 1
            if not (0 <= x <= image_w and 0 <= y <= image_h):
                errors.append(f"Keypoint #{idx} ({x}, {y}) tràn khỏi khung ảnh {image_w}x{image_h}")

    if visible_count == 0:
        errors.append("Đối tượng không có keypoint nào hiển thị (visible_count=0)")

    return errors


def check_batch_file(batch_path: Path, output_report: Path | None = None) -> int:
    payload = json.loads(batch_path.read_text(encoding="utf-8"))
    frames = payload.get("frames", [])
    print(f"=== RÀ SOÁT CHẤT LƯỢNG NHÃN 3D & POSE: {batch_path.name} ({len(frames)} frames) ===")

    total_issues = 0
    frame_reports = {}

    for frame in frames:
        fid = frame.get("frame_id", "unknown")
        img_w = frame.get("image_width", 1920)
        img_h = frame.get("image_height", 1080)
        issues: List[str] = []

        for box in frame.get("boxes_3d", []):
            for err in validate_3d_box(box):
                issues.append(f"[3D ID={box.get('id')}] {err}")

        for pose in frame.get("poses_2d", []):
            for err in validate_pose_keypoints(pose, img_w, img_h):
                issues.append(f"[Pose ID={pose.get('person_id')}] {err}")

        if issues:
            total_issues += len(issues)
            frame_reports[fid] = issues
            print(f"  [REJECT] {fid}: {len(issues)} lỗi QC:")
            for item in issues:
                print(f"           - {item}")
        else:
            print(f"  [PASS]   {fid}: Đạt chuẩn hình học 3D & 2D Keypoints")

    summary = {
        "batch_file": str(batch_path),
        "total_frames": len(frames),
        "rejected_frames": len(frame_reports),
        "total_issues": total_issues,
        "issues_by_frame": frame_reports,
    }

    if output_report:
        output_report.parent.mkdir(parents=True, exist_ok=True)
        output_report.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"=> Đã lưu báo cáo QC tại: {output_report}")

    return 0 if total_issues == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Kiểm tra chất lượng nhãn 3D LiDAR Bounding Box và 2D Body Keypoints.")
    parser.add_argument("batch_json", type=Path, help="Đường dẫn file JSON chứa batch annotation")
    parser.add_argument("--report", type=Path, default=None, help="Đường dẫn xuất file báo cáo lỗi JSON")
    args = parser.parse_args()

    if not args.batch_json.exists():
        print(f"[LỖI] Không tìm thấy file: {args.batch_json}", file=sys.stderr)
        return 2
    return check_batch_file(args.batch_json, args.report)


if __name__ == "__main__":
    sys.exit(main())
