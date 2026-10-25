# Labeling loop

Model guesses, you correct, model retrains on your corrections, guesses get better.
Round 1 is mostly manual; by round 3 you are mostly nudging points a few pixels.

Everything here uses the 15-point schema in `src/swimform/schema.py`. The Label Studio
config (`labeling/labeling_config.xml`) is generated from it
(`python -m swimform.labeling.config > labeling/labeling_config.xml`), so names and order
can't drift. Label names are matched by meaning, so a project that already uses
`left_shoulder` or `neck` still converts.

## Round 0 → round 1: starting from the stock model

There are no 15-point weights yet, so round 1's guesses come from the stock COCO-17
model. Its twelve limb points pre-fill directly; the neck and waist start at the
shoulder and hip midpoints (nudge them onto the spine); the top of the head starts
empty (the stock model only knows the face).

    python -m swimform.pose pose_out_r0 yolo11m-pose.pt clips/*.mp4
    python -m swimform.labeling.prep --videos clips/ --pose pose_out_r0 --out round1 --per-clip 30 \
        --model-version round0-coco

From round 2 on, run `swimform.pose` with your own `best.pt`: it detects the 15-point
model and writes every point.

## What clips to collect

Swimmer variety beats frame count. For round 1, aim for **8–10 different swimmers**,
each with:

- a side clip of a 50 (down and back, camera on one deck): both arms get a near view;
- a head-on clip from the end wall;
- deck-level where that's all you have; underwater side clips where you can (they are
  the only way to label the catch).

Name clips `<swimmer>_<view>_<n>.mp4` (e.g. `jen_side_1.mp4`). The val split holds out
whole clips, so frames from one swim never land on both sides of the split.

## The three spine points

- **top_of_head**: the crown, the highest point of the skull along the body line. Not
  the face, the forehead or the cap's logo.
- **spine_at_neck**: the base of the neck, between the shoulders (the bony bump at the
  top of the spine, C7). From the side it sits just behind the near shoulder.
- **spine_at_waist**: the low back, midway between the hips, on the spine.

Left and right are always the swimmer's own, never the screen's. In the config, left
points are warm colours and right points cool, so a swap stands out. Hotkeys 1–0 and
q–t select the points in schema order.

## One-time setup (Windows, 3070 Ti)

    pip install label-studio ultralytics
    set LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
    set LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=C:\swim\round1
    label-studio start

In the UI: new project -> Labeling Setup -> Code -> paste `labeling/labeling_config.xml`.
Then Settings -> Cloud Storage -> Add Local Files, point it at the same folder,
and Import `tasks.json`.

## Each round

    # 1. pick frames and pre-fill the current model's guesses
    python -m swimform.labeling.prep --videos clips/ --pose pose_out/ --out roundN --per-clip 40 \
        --model-version roundN-1

    # 2. label in Label Studio, then export as JSON

    # 3. build the training set (val is split BY CLIP, not by frame)
    python -m swimform.labeling.convert --export export.json --images roundN/images --out dataset_rN

    # 4. fine-tune
    yolo pose train data=dataset_rN/data.yaml model=yolo11m-pose.pt epochs=100 imgsz=960 batch=8

    # 4b. grade it against the held-out clips (exit code 0 = passes the bar)
    python -m swimform.scorecard --data dataset_rN/data.yaml \
        --weights runs/pose/train_prev/weights/best.pt runs/pose/train/weights/best.pt \
        --out scorecard_rN --draw 12

    # 5. re-run the harness with the new weights to generate round N+1's guesses
    python -m swimform.pose pose_out_rN runs/pose/train/weights/best.pt clips/*.mp4

Keep every round's labels. Round N+1's dataset is all rounds combined, not just the new frames.

## Labeling rules — decide these once and never change them mid-project

- **Place a point only where you can locate the joint**, seen or confidently inferred
  from the limb either side of it. If you are guessing, leave it out: a missing point
  trains fine, a wrong one teaches the model to be wrong.
- **Mark every point visible or occluded.** Underwater and splash-hidden joints are
  `occluded`. This distinction is the whole reason a swimming model beats a generic one.
- **Mark a frame `unusable`** when you cannot place at least a shoulder and a hip.
  Those frames still matter — they tell you what share of a real video is unreadable.
- **Elbow means the joint**, not the visible skin edge. Under refraction the limb bends;
  follow the bone, not the silhouette.
- **One swimmer per frame — always the near-lane athlete.** Ignore neighbours and anyone
  on deck. This is what stops the model locking onto a bystander.

## Frame budget

Round 1: ~250 frames (8–10 swimmers x 30). Enough to see movement, not enough to ship.
Realistic target is 300-500 frames across 10+ different swimmers, pools, and lighting.
Frames from one clip are near duplicates, so swimmer variety counts far more than frame count.

## Where it pays to spend the $5k

Not on "train a model." On labeled frames: a vendor can place points, you review.
Your own time is better spent on the technique labels (dropped elbow / head lift and
how severe), which nobody else can produce.

## Is the model good enough?

`swimform.scorecard` grades a model the way the analysis uses it. A keypoint at or above
confidence 0.5 is trusted, and everything below it is ignored. So the numbers that matter are:

- **confident-wrong**: trusted points more than 10% of body size off your label. These
  become wrong measurements that nothing warns about, so this must stay the smallest
  number (draft bar: 5% or less for every keypoint).
- **coverage**: of the points you marked visible, how many the model trusts. Low coverage
  only costs checks ("couldn't check"), never correctness.
- **left/right swaps**, **phantom points** (confident where you placed nothing) and
  **swimmer missed**.

Every number is per keypoint, with visible and occluded points split. The table names the
frames to look at first, and `--draw` saves them with your label in green and the model's
in red. Pass two sets of weights to see whether a round actually improved things.
The bar lives in `scorecard.BAR` and is a draft.
