"""
The Label Studio side of the 15-point schema: the labeling config, and how to map a
model's keypoints (COCO-17 from the stock model, or 15-point from ours) onto it.

Everything is generated from swimform.schema, so the labeling UI, the dataset the
converter writes and the analysis can never disagree about names or order.

  python -m swimform.labeling.config > labeling/labeling_config.xml
"""
from xml.sax.saxutils import quoteattr

from ..schema import KEYPOINTS

# One colour per point; left side warm, right side cool, spine neutral, so a swapped
# left/right is visible at a glance while labeling.
COLOURS = {
    "top_of_head": "#ffffff", "spine_at_neck": "#d0d0d0", "spine_at_waist": "#909090",
    "l_shoulder": "#e6194b", "l_elbow": "#f58231", "l_wrist": "#ffe119",
    "l_hip": "#f032e6", "l_knee": "#fabebe", "l_ankle": "#9a6324",
    "r_shoulder": "#4363d8", "r_elbow": "#46f0f0", "r_wrist": "#3cb44b",
    "r_hip": "#911eb4", "r_knee": "#aaffc3", "r_ankle": "#008080",
}
HOTKEYS = "1234567890qwert"   # in schema order: press the key, then click the joint

GUIDE = ("Mark the near-lane swimmer only. Top of head = crown, not the face. "
         "Neck = base of the neck between the shoulders (the bump at the top of the spine). "
         "Waist = low back, midway between the hips. "
         "Left/right are the SWIMMER'S, not the screen's. Warm colours = left, cool = right. "
         "Leave out any joint you would be guessing; mark underwater or splash-hidden ones 'occluded'.")

# Other spellings a Label Studio project or model might use for the same point.
ALIASES = {
    "head": "top_of_head", "head_top": "top_of_head", "crown": "top_of_head",
    "neck": "spine_at_neck", "spine_neck": "spine_at_neck",
    "waist": "spine_at_waist", "mid_hip": "spine_at_waist", "spine_waist": "spine_at_waist",
    "low_back": "spine_at_waist",
}
for _k in KEYPOINTS:
    if _k[:2] in ("l_", "r_"):
        side = "left" if _k[0] == "l" else "right"
        ALIASES[f"{side}_{_k[2:]}"] = _k
        ALIASES[f"{side} {_k[2:]}"] = _k


def canonical(name):
    """A label as exported -> its schema name, or None if it isn't one of ours."""
    n = name.strip().lower().replace("-", "_")
    if n in KEYPOINTS:
        return n
    return ALIASES.get(n) or ALIASES.get(n.replace("_", " "))


def build_config():
    labels = "\n".join(
        f'    <Label value="{k}" background="{COLOURS[k]}" hotkey="{HOTKEYS[i]}"/>'
        for i, k in enumerate(KEYPOINTS))
    return f"""<View>
  <Header value={quoteattr(GUIDE)} size="6"/>
  <KeyPointLabels name="kp" toName="img" strokeWidth="3" opacity="0.9">
{labels}
  </KeyPointLabels>
  <Choices name="vis" toName="img" perRegion="true" required="true">
    <Choice value="visible"/>
    <Choice value="occluded"/>
  </Choices>
  <Choices name="frame" toName="img" required="true">
    <Choice value="usable"/>
    <Choice value="unusable"/>
  </Choices>
  <Image name="img" value="$image" zoom="true" zoomControl="true" rotateControl="false"/>
</View>
"""


def model_points(kp):
    """
    A model's keypoints for one frame -> {schema name: (x, y, conf, derived)}.

    15-point output maps straight across. COCO-17 output (the stock model, before our
    first fine-tune) shares the twelve limb points; its face points don't map to the top
    of the head, so that is left for the labeler. The two spine points are pre-placed
    at the shoulder and hip midpoints, flagged as derived: a starting position to nudge,
    no better than that.
    """
    out = {}
    for name, v in (kp or {}).items():
        k = canonical(name)
        if k and v is not None:
            out[k] = (float(v[0]), float(v[1]), float(v[2]), False)
    for mid, (a, b) in (("spine_at_neck", ("l_shoulder", "r_shoulder")),
                        ("spine_at_waist", ("l_hip", "r_hip"))):
        if mid not in out and a in out and b in out:
            pa, pb = out[a], out[b]
            out[mid] = ((pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2, min(pa[2], pb[2]), True)
    return out


if __name__ == "__main__":
    print(build_config(), end="")
