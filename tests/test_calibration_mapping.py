import copy
import importlib.util
from pathlib import Path

import cv2
import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location("mapping_calibration", Path(__file__).resolve().parents[1] / "scripts/calibrate_rig.py")
mapping = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mapping)


def candidate_fixture():
    return {"verified": True, "cameras": {
        "camera_%d" % i: {"source_camera_id": "cam%d" % j,
                           "topic_mapping_verified": True,
                           "source_mask_path": "/raw_mask_%d.png" % i,
                           "T_body_camera": np.eye(4).tolist()}
        for j, i in enumerate(range(2, 6))}}


def test_all_24_mappings_are_distinct_and_raw_masks_do_not_move():
    candidate = candidate_fixture()
    original = copy.deepcopy(candidate)
    values = mapping.mapping_candidates(candidate)
    assert len(values) == 24
    assert len({tuple(sorted(v["mapping"].items())) for v in values}) == 24
    for value in values:
        assert value["candidate"]["verified"] is False
        assert value["candidate"]["topic_mapping_verified"] is False
        for camera in candidate["cameras"]:
            assert value["candidate"]["cameras"][camera]["source_mask_path"] == candidate["cameras"][camera]["source_mask_path"]
            assert value["candidate"]["cameras"][camera]["topic"] == mapping.RAW_TOPICS[camera]
            assert value["candidate"]["cameras"][camera]["topic_mapping_verified"] is False
    assert candidate == original


def test_duplicate_physical_model_ids_are_rejected():
    candidate = candidate_fixture()
    candidate["cameras"]["camera_3"]["source_camera_id"] = "cam0"
    with pytest.raises(ValueError, match="distinct"):
        mapping.mapping_candidates(candidate)


def test_single_candidate_and_no_controls_can_never_verify_mapping():
    score = {"id": "candidate", "mapping": {"camera_2": "cam0"}, "train_nmi": 1., "validation_nmi": 1.}
    gate = mapping.mapping_evidence_gate([score], 20, [])
    assert not gate["verified"] and not gate["topic_mapping_verified"]
    assert "fewer_than_two_distinct_mapping_candidates" in gate["reasons"]
    assert "missing_valid_negative_controls" in gate["reasons"]


def test_good_weak_scores_do_not_certify_absolute_calibration():
    scores = [{"id": "a", "mapping": {"camera_2": "cam0"}, "train_nmi": .5, "validation_nmi": .5},
              {"id": "b", "mapping": {"camera_2": "cam1"}, "train_nmi": .1, "validation_nmi": .1}]
    controls = [{"valid": True, "candidate_id": "a", "nmi": .05, "shift_px": s} for s in [-120, 120]]
    assert mapping.mapping_evidence_gate(scores, 4, controls)["status"] == "insufficient_evidence"
    gate = mapping.mapping_evidence_gate(scores, 4, controls, independent_locations=2)
    assert gate["status"] == "comparatively_supported_unverified"
    assert not gate["topic_mapping_verified"]
    scores[1]["validation_nmi"] = .6
    gate = mapping.mapping_evidence_gate(scores, 4, controls, independent_locations=2)
    assert gate["selected_on_train"] == "a"
    assert gate["status"] == "insufficient_evidence"


def test_static_image_check_uses_real_tracks_and_rejects_motion_and_textureless_frames():
    rng = np.random.default_rng(1)
    image = np.zeros((240, 320, 3), np.uint8)
    for x, y in rng.integers([10, 10], [300, 220], (150, 2)):
        cv2.rectangle(image, (int(x), int(y)), (int(x)+4, int(y)+4), (255, 255, 255), -1)
    assert mapping.image_stationarity(image, image)["passed"]
    shifted = cv2.warpAffine(image, np.float32([[1, 0, 8], [0, 1, 0]]), (320, 240))
    assert not mapping.image_stationarity(image, shifted)["passed"]
    assert not mapping.image_stationarity(np.zeros_like(image), np.zeros_like(image))["passed"]


def test_mapping_projection_rejects_size_mismatch_before_scoring():
    camera = {"model": "triple_sphere", "width": 1280, "height": 1080,
              "intrinsics": {"fx": 600, "fy": 600, "cx": 640, "cy": 540},
              "distortion": [-.2, -.06, .59, 0., 0.]}
    with pytest.raises(ValueError, match="size differs"):
        mapping.mapping_projection_score(np.array([[0., 0., 2.]]), np.array([10]),
            np.zeros((540, 640, 3), np.uint8), np.eye(4), camera)
