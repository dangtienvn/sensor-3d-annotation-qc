# 3D Sensor & Human Pose Annotation QC CLI

Công cụ dòng lệnh Python chạy trên **Linux Terminal** phục vụ kiểm soát chất lượng (QC) tập nhãn **3D Point Cloud Bounding Box (LiDAR)** và **2D Human Pose Keypoints (COCO-17)** trước khi bàn giao dataset huấn luyện mô hình AI.

## Tính năng chính

1. **Kiểm tra ràng buộc hình học 3D Bounding Box:**
   - Ràng buộc kích thước vật lý hợp lệ (`length`, `width`, `height > 0` và đối chiếu ngưỡng thực tế theo phân lớp đối tượng).
   - Kiểm tra góc hướng di chuyển `yaw` nằm đúng chuẩn $[-\pi, \pi]$ rad và mức độ che khuất (`occlusion_level`).
2. **Kiểm tra khung xương chuyển động con người (COCO-17 Keypoints):**
   - Xác minh đủ 17 điểm khớp cơ thể, cờ trạng thái hiển thị hợp lệ và phát hiện tọa độ bị tràn ra ngoài độ phân giải khung hình camera.
3. **Báo cáo lỗi QC tự động:**
   - Phân tách rõ các frame đạt chuẩn (`PASS`) và frame cần gán nhãn lại (`REJECT`) kèm báo cáo tổng hợp định dạng JSON.

## Cấu trúc thư mục

```text
pose-bbox-validator/
├── qc_checker.py                      # Script CLI kiểm tra chất lượng nhãn 3D & Pose
├── sample_data/
│   └── frames_batch_01.json           # Batch dữ liệu mẫu gồm frame sạch và frame lỗi hình học
├── tests/
│   └── test_qc_checker.py             # Unit tests với pytest
└── README.md
```

## Cách chạy trên Linux Terminal

```bash
# 1. Rà soát chất lượng batch nhãn mẫu và xuất file report
python3 qc_checker.py sample_data/frames_batch_01.json --report outputs/batch_qc_report.json

# 2. Chạy kiểm thử tự động
PYTHONPATH=. pytest -v
```

## Tác giả

- **Đặng Thanh Tiến** — [github.com/dangtienvn](https://github.com/dangtienvn)
- Giấy phép: MIT License
