import numpy as np
import pytest

from scene_change.harness.evaluate import frame_metrics, pixel_metrics


def test_frame_metrics_follow_the_paslcd_protocol():
    gt = np.zeros((3, 4, 4), bool)
    gt[0, :2, :2] = True                      # 4 change pixels
    gt[1, 0, 0] = True
    pred = np.zeros_like(gt)
    pred[0, :2, :1] = True                    # 2 of 4 found: IoU 0.5, F1 2/3
    pred[2, 3, 3] = True                      # false alarm on a change-free frame
    m = frame_metrics(pred, gt)
    assert m["frames_with_change"] == 2
    assert m["IoU"] == pytest.approx((0.5 + 0.0) / 2)
    assert m["F1"] == pytest.approx((2 / 3 + 0.0) / 2)
    assert m["false_alarm_rate"] == pytest.approx(1 / 16)
    assert m["frames_with_false_alarm"] == 1
    p = pixel_metrics(pred, gt)
    assert p["IoU_change"] == pytest.approx(2 / 6)


@pytest.mark.slow
def test_end_to_end_on_a_generated_scenario():
    from scene_change.harness.evaluate import evaluate_3d, gt_masks, predicted_masks
    from scene_change.harness.scenario import build_scenario
    from scene_change.pipeline import build, inspect

    sc = build_scenario(7, "mixed", fast=True)
    model = build(sc["baseline"], rooms=sc["rooms"])
    result = inspect(model, sc["inspection"])
    ev = evaluate_3d(sc, model, result)
    assert ev["registration"]["trans_cm_median"] < 6.0
    assert ev["objects"]["recall"] >= 0.6
    gt = gt_masks(sc["gt_arrays"], sc["gt"]["changes"])
    fm = frame_metrics(predicted_masks(result, sc["inspection"], model.gaussians.voxel), gt)
    assert fm["F1"] > 0.4
