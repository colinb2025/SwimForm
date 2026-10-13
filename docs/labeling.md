# Labeling loop

Model guesses, you correct, model retrains on your corrections, guesses get better.
Round 1 is mostly manual; by round 3 you are mostly nudging points a few pixels.

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

Round 1: ~120 frames here. Enough to see movement, not enough to ship.
Realistic target is 300-500 frames across 10+ different swimmers, pools, and lighting.
Frames from one clip are near duplicates, so swimmer variety counts far more than frame count.

## Where it pays to spend the $5k

Not on "train a model." On labeled frames: a vendor can place points, you review.
Your own time is better spent on the technique labels (dropped elbow / head lift and
how severe), which nobody else can produce.
