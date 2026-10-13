# Baseline: off-the-shelf pose estimation on swim video

**Date:** 2026-10-13
**Model:** YOLO11m-pose, COCO-17 keypoints, no fine-tuning
**Data:** 6 clips, 3 side view and 3 head-on, all above water, indoor and outdoor
**Conclusion:** unusable as-is. Fine-tuning on labeled swim frames is the critical path,
not an optimisation.

## Detection rates

Share of frames in which the joint was found with confidence > 0.5.

| Clip | View | Any detection | Shoulders | Elbows | Wrists |
|---|---|---|---|---|---|
| c2ec76ea | side, close | 75% | 67% | 62% | 60% |
| 3ecd94c6 | side, close | 72% | 68% | 57% | 49% |
| 3f95fbb0 | head-on, far | 49% | 40% | 33% | 38% |
| 5ec20645 | side, low/far | 39% | 21% | 18% | 15% |
| ebeafa58 | head-on, backlit | 36% | 11% | 19% | 27% |
| 89a9cfd5 | head-on, far | 14% | 10% | 8% | 6% |

Raw numbers in `baseline_summary.json`.

These rates **overstate** performance, because many detections are confidently wrong.

## Failure modes

1. **Garbled skeletons.** On close side views the model finds the swimmer but fits a
   standing pose to a horizontal body: limbs drawn across the torso, joints landing on
   splash.
2. **Wrong subject.** On one head-on clip it locked onto a person standing on the pool
   deck. A bystander is a far easier detection than a small, half-submerged swimmer.
3. **Body parts read as whole people.** On one side view it treated the underwater legs
   as a separate person and fitted a full skeleton to them.
4. **Occlusion gaps.** During the catch and under splash there is no detection at all.

The failures are systematic rather than random, which is what fine-tuning fixes.

## Downstream consequence

Running `swimform.metrics` on the best clip returned head-lift angles of 80° and 126°
on separate stroke cycles — physically impossible. Bad keypoints produce bad angles,
and a confident-looking score from bad keypoints is worse than no score.

## What this implies for the product

- **A swimmer-selection step is required before pose**, and is not in the spec.
  Something must decide which detection is the athlete — lane position plus motion
  continuity is the obvious first attempt. The deck-bystander failure is what happens
  without it.
- **Far head-on framing may not be recoverable** at the distances in these clips. That
  is a recording-guidance finding: the spec should constrain how far the camera sits.
- **Frame timing from screen recordings is unusable.** These clips were all recorded at
  60fps but reported 25.4, 57.0, 29.8, 24.9, 29.2 and 33.7fps after screen capture.
  Any tempo or timing metric needs original files.

## Reproducing

```bash
python -m swimform.pose out/ yolo11m-pose.pt clips/*.mov
python -m swimform.metrics out/*_pose.json
```

Re-run this after each fine-tuning round and compare against the table above. It is the
regression test for the model.
