from qc_checker import validate_3d_box, validate_pose_keypoints


def test_valid_3d_box():
    box = {
        "id": 1,
        "class_name": "Car",
        "center": {"x": 10.0, "y": 5.0, "z": 0.0},
        "dimensions": {"length": 4.5, "width": 1.8, "height": 1.5},
        "yaw": 0.5,
        "occlusion_level": 0,
    }
    assert validate_3d_box(box) == []


def test_invalid_yaw_and_negative_length():
    box = {
        "id": 2,
        "class_name": "Car",
        "center": {"x": 10.0, "y": 5.0, "z": 0.0},
        "dimensions": {"length": -1.0, "width": 1.8, "height": 1.5},
        "yaw": 3.8,
        "occlusion_level": 1,
    }
    errs = validate_3d_box(box)
    assert len(errs) == 2


def test_out_of_bounds_keypoint():
    kps = [[100, 100, 2] for _ in range(17)]
    kps[0] = [1950, 100, 2]
    errs = validate_pose_keypoints({"person_id": 1, "keypoints": kps}, 1920, 1080)
    assert any("tràn khỏi khung ảnh" in e for e in errs)
