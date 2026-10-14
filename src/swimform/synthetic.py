"""
Synthetic freestyle pose tracks with injectable technique faults.

Lets metrics, rules, the two-view fusion and anything downstream run end to end
before the fine-tuned model exists. Each fault is named by its `issue_id` in
rules/swim_rules.xlsx and takes a severity in [0, 1], so a test can ask "does a
severe dropped elbow trip the dropped_elbow rule, and does a clean swimmer trip
nothing?" against the real workbook.

One 3D body, two cameras. Joints are generated in body coordinates

    X forward (direction of travel), Y down (depth), Z lateral (+ = swimmer's left)

and each camera is a projection of the same body: the side camera (on the swimmer's
right) sees (X, Y); the head-on camera sees (Z, Y), with the swimmer's right on
image-left. Body roll is a real rotation about X, and whether a joint is underwater
follows from its true depth. Because both views come from one body, the 3D stroke
fused from them can be checked against ground truth (`truth3d`).

This is a kinematic sketch, not a simulator: shapes and timings are close enough to
exercise the code, and lengths may stretch slightly where two planar motions combine.

  python -m swimform.synthetic out/ --faults dropped_elbow=1 late_breath=0.6

Body units: half shoulder width 0.2, shoulder-to-hip 0.5, upper arm 0.3, forearm 0.3,
thigh 0.42, shin 0.40.
"""
import argparse, json, os
import numpy as np

from .posedata import PoseTrack
from .schema import HEAD_ON, IDX, KEYPOINTS, NUM_KEYPOINTS, SIDE

FAULTS = (
    "dropped_elbow", "straight_arm_pull", "short_pull", "high_head_position",
    "head_lift_on_breath", "late_breath", "low_body_rotation", "rotation_asymmetry",
    "crossover_entry", "sinking_hips", "excessive_knee_bend", "wide_kick",
    "over_gliding", "wide_recovery_straight_arm",
)

L_UP, L_FORE, L_THIGH, L_SHIN = 0.30, 0.30, 0.42, 0.40
HALF_W, HIP_HALF_W = 0.20, 0.13
SURFACE_Y = 0.03   # joints deeper than this are underwater

# jitter (fraction of frame), per-point dropout, splash rate, whole-frame loss
QUALITY = {
    "clean":   (0.000, 0.00, 0.0, 0.00),
    "typical": (0.003, 0.02, 1.0, 0.00),
    "poor":    (0.007, 0.08, 2.0, 0.04),
}


class Swimmer:
    def __init__(self, faults=None, cycle_s=1.6, breath_side="r", breath_every=1):
        """breath_every: breathe once every N stroke cycles; 0 = no breath in the clip."""
        faults = dict(faults or {})
        for k in faults:
            if k not in FAULTS:
                raise KeyError(f"unknown fault {k!r}; known: {', '.join(FAULTS)}")
        self.faults = {k: float(np.clip(faults.get(k, 0.0), 0, 1)) for k in FAULTS}
        self.cycle_s = cycle_s
        self.breath_side = breath_side
        self.breath_every = breath_every

    def __getitem__(self, k):
        return self.faults[k]

    # Arm timing, as fractions of one arm's cycle: glide -> pull -> recovery.
    @property
    def glide(self):
        return 0.12 + 0.43 * self["over_gliding"]

    @property
    def pull(self):
        return 0.55 * (1 - self.glide)

    @property
    def pull_end(self):
        return self.glide + self.pull

    @property
    def recovery(self):
        return 1 - self.pull_end

    @property
    def breath_center(self):
        """Breathing-arm phase where the head is most turned: at pull end, or as late as
        the hand's re-entry (the workbook's 'still turned as the arm returns to the front')."""
        return self.pull_end + self.recovery * self["late_breath"]


# ------------------------------------------------------------------ helpers
def _smooth(s):
    return s * s * (3 - 2 * s)


def _bump(phase, center, width):
    d = (phase - center + 0.5) % 1.0 - 0.5
    return np.where(np.abs(d) < width, 0.5 * (1 + np.cos(np.pi * d / width)), 0.0)


def _breath(t, sw, arm_phase):
    if sw.breath_every == 0:   # holds breath for the whole clip
        return np.zeros_like(t)
    b = _bump(arm_phase, sw.breath_center, 0.18)
    if sw.breath_every > 1:
        k = np.floor(t / sw.cycle_s + 0.5 - sw.breath_center).astype(int)
        b = np.where(k % sw.breath_every == 0, b, 0.0)
    return b


def _kick(t, sw, offset):
    """6-beat kick: thigh swing and knee flex (radians) for one leg."""
    ph = 2 * np.pi * 3 / sw.cycle_s * t + offset
    hip = np.radians(8) * np.sin(ph)
    kmax = np.radians(28 + 42 * sw["excessive_knee_bend"])
    knee = kmax * np.clip(np.sin(ph + np.pi / 3), 0, None)
    return hip, knee


def _polar(r, ang):
    """Point at distance r, angle `ang` below forward horizontal, in the (X, Y) plane."""
    return r * np.stack([np.cos(ang), np.sin(ang)], axis=-1)


def _roll(yz, rho):
    """Rotate (Y, Z) about the X axis; positive rho drives the right side (-Z) deeper."""
    y, z = yz[..., 0], yz[..., 1]
    c, s = np.cos(rho), np.sin(rho)
    return np.stack([y * c - z * s, y * s + z * c], axis=-1)


# Side-plane pull: upper-arm angle a and forearm angle b, degrees below forward
# horizontal, against pull progress s. Forearm vertical (b=90) early with the
# elbow still high (a~30) is a good catch; ~120 deg at the elbow mid-pull.
_S = [0.0, 0.2, 0.5, 0.75, 1.0]
_A = [10, 30, 60, 115, 165]
_B = [10, 90, 120, 150, 170]


def _pull_angles(s, sw):
    a = np.radians(np.interp(s, _S, _A))
    b = np.radians(np.interp(s, _S, _B))
    # Dropped elbow: forearm stays near horizontal early; elbow sinks under the hand.
    early = np.clip((0.5 - s) / 0.15, 0, 1)
    b = b - sw["dropped_elbow"] * early * (b - np.minimum(a, np.radians(20)))
    # Straight-arm pull: forearm stays in line with the upper arm through mid-pull.
    mid = np.clip(1 - np.abs(s - 0.5) / 0.35, 0, 1)
    b = b - sw["straight_arm_pull"] * mid * (b - (a + np.radians(8)))
    return a, b


def _arm(p, sw, side, sh):
    """
    3D elbow and wrist for one arm at arm phase p.
    side: -1 right arm, +1 left arm. sh: (T, 3) shoulder position.
    Glide and pull: sagittal shape from the pull schedule, lateral path from the
    entry line. Recovery: elbow swings up and out to the side, forearm hangs below it.
    """
    n = len(p)
    el, wr = np.zeros((n, 3)), np.zeros((n, 3))
    s_end = 1.0 - 0.4 * sw["short_pull"]
    a1 = np.radians(10.0)
    E1 = _polar(L_UP, a1)
    W1 = E1 + _polar(L_FORE, a1)
    ae, be = _pull_angles(np.array([s_end]), sw)
    E0 = _polar(L_UP, ae)[0]
    W0 = E0 + _polar(L_FORE, be)[0]
    z_entry = side * (HALF_W - 0.26 * sw["crossover_entry"])   # crossover moves it inward

    def place(m, e_xy, w_xy, e_z, w_z):
        el[m, :2] = sh[m, :2] + e_xy
        wr[m, :2] = sh[m, :2] + w_xy
        el[m, 2], wr[m, 2] = e_z, w_z

    g = p < sw.glide
    place(g, E1, W1, 0.5 * (sh[g, 2] + z_entry) + side * 0.03, z_entry)

    m = (p >= sw.glide) & (p < sw.pull_end)
    s = (p[m] - sw.glide) / sw.pull * s_end
    a, b = _pull_angles(s, sw)
    e_xy = _polar(L_UP, a)
    w_xy = e_xy + _polar(L_FORE, b)
    w_z = np.interp(s / s_end, [0, .5, 1], [z_entry, side * .08, side * .15])
    place(m, e_xy, w_xy, 0.5 * (sh[m, 2] + w_z) + side * 0.06, w_z)

    m = p >= sw.pull_end
    s = _smooth((p[m] - sw.pull_end) / sw.recovery)
    lift = np.sin(np.pi * s)
    se = s ** 0.8                                   # elbow leads the hand slightly
    # High-elbow recovery: elbow out to the side near shoulder height, hand hanging
    # below it just above the water.
    e_n = E0 + (E1 - E0) * se[:, None] + np.array([0, -0.10]) * lift[:, None]
    w_n = W0 + (W1 - W0) * s[:, None] + np.array([0, -0.02]) * lift[:, None]
    ez_n = side * (HALF_W + 0.25 * lift) * (1 - s) + (0.5 * (sh[m, 2] + z_entry) + side * 0.03) * s
    wz_n = side * (HALF_W + 0.25 * lift) * (1 - s) + z_entry * s
    # Straight-arm recovery: a rigid spoke swung over the top.
    ang = np.radians(np.interp(s, [0, 1], [np.degrees(np.arctan2(W0[1], W0[0])) % 360, 370]))
    w_f = _polar(0.58, ang)
    e_f = 0.5 * w_f
    wz_f = side * (HALF_W + 0.10 * lift) * (1 - s) + z_entry * s
    ez_f = 0.5 * (sh[m, 2] + wz_f)
    k = sw["wide_recovery_straight_arm"]
    place(m, (1 - k) * e_n + k * e_f, (1 - k) * w_n + k * w_f,
          (1 - k) * ez_n + k * ez_f, (1 - k) * wz_n + k * wz_f)
    return el, wr


def roll_amplitudes(sw):
    """(peak roll toward the right side, toward the left side), degrees."""
    r = 40.0 - 28.0 * sw["low_body_rotation"]
    return r, r * (1 - 0.45 * sw["rotation_asymmetry"])


def truth3d(sw, t):
    """Ground-truth 3D joints (T, 15, 3) in body units: X fwd, Y down, Z lateral (+left)."""
    n = len(t)
    J = np.zeros((n, NUM_KEYPOINTS, 3))
    phi = (t / sw.cycle_s) % 1.0
    high, sink = sw["high_head_position"], sw["sinking_hips"]

    def put(k, xyz):
        J[:, IDX[k]] = xyz

    rr, rl = roll_amplitudes(sw)
    wave = np.sin(2 * np.pi * (phi - (sw.glide + sw.pull / 2) + 0.25))  # +1: right shoulder deepest
    rho = np.radians(np.where(wave > 0, rr, rl) * wave)

    neck = np.stack([np.zeros(n), 0.012 * np.sin(4 * np.pi * phi), np.zeros(n)], -1)
    put("spine_at_neck", neck)
    for sd, z in (("r", -HALF_W), ("l", HALF_W)):
        yz = _roll(np.tile([0.0, z], (n, 1)), rho)
        put(f"{sd}_shoulder", neck + np.stack([np.zeros(n), yz[:, 0], yz[:, 1]], -1))

    turn = _breath(t, sw, phi if sw.breath_side == "r" else (phi + 0.5) % 1)
    zside = -1.0 if sw.breath_side == "r" else 1.0
    head_y = -0.04 - 0.13 * high + (0.03 - 0.11 * sw["head_lift_on_breath"]) * turn
    put("top_of_head", neck + np.stack([np.full(n, 0.20), head_y, zside * 0.09 * turn], -1))

    waist = neck + np.array([-0.50, 0.02 + 0.09 * high + 0.09 * sink, 0.0])
    put("spine_at_waist", waist)
    for sd, z in (("r", -HIP_HALF_W), ("l", HIP_HALF_W)):
        yz = _roll(np.tile([0.0, z], (n, 1)), 0.7 * rho)
        put(f"{sd}_hip", waist + np.stack([np.zeros(n), yz[:, 0], yz[:, 1]], -1))

    leg_tilt = np.radians(4 + 12 * sink + 4 * high)
    for sd, side, off, koff in (("r", -1.0, 0.0, 0.0), ("l", 1.0, 0.5, np.pi)):
        sh = J[:, IDX[f"{sd}_shoulder"]]
        e, w = _arm((phi + off) % 1, sw, side, sh)
        put(f"{sd}_elbow", e)
        put(f"{sd}_wrist", w)
        hip_a, knee_f = _kick(t, sw, koff)
        th = np.pi - leg_tilt + hip_a                   # pointing back, slightly down
        hp = J[:, IDX[f"{sd}_hip"]]
        kn_xy = hp[:, :2] + _polar(L_THIGH, th)
        an_xy = kn_xy + _polar(L_SHIN, th + knee_f)     # knee flex lifts the heel
        spread = side * 0.20 * sw["wide_kick"] * np.abs(np.sin(3 * np.pi * phi))
        put(f"{sd}_knee", np.column_stack([kn_xy, hp[:, 2] * 0.85]))
        put(f"{sd}_ankle", np.column_stack([an_xy, hp[:, 2] * 0.72 + spread]))
    return J


# ------------------------------------------------------------------ public
def generate(view, swimmer=None, *, duration_s=12.0, fps=30.0, size=(1920, 1080),
             facing="right", camera="underwater", quality="typical", seed=0, t0=0.0):
    """
    A synthetic PoseTrack for one camera view of the swimmer.

    camera: "underwater" sees below-water joints; "deck" sees them less the deeper they
            are (shallow joints stay readable, the catch does not), as on real deck footage.
    facing: side view only — direction of travel in the image.
    t0:     start time in seconds, so two views can come from different laps.
    """
    if view not in (SIDE, HEAD_ON):
        raise ValueError(f"view must be {SIDE!r} or {HEAD_ON!r}")
    if camera not in ("underwater", "deck"):
        raise ValueError("camera must be 'underwater' or 'deck'")
    sw = swimmer or Swimmer()
    rng = np.random.default_rng(seed)
    jitter, dropout, splash, frame_loss = QUALITY[quality]
    W, H = size
    n = int(round(duration_s * fps))
    t = t0 + np.arange(n) / fps
    J = truth3d(sw, t)
    under = J[..., 1] > SURFACE_Y

    if view == SIDE:
        P, scale, cx, cy = J[..., [0, 1]], 0.40, 0.62, 0.40
    else:
        P, scale, cx, cy = J[..., [2, 1]], 0.60, 0.50, 0.45
    # Body units -> pixels, isotropic so angles survive the projection.
    px = np.empty_like(P)
    px[..., 0] = (cx + P[..., 0] * scale) * W
    px[..., 1] = cy * H + P[..., 1] * scale * W

    conf = np.full((n, NUM_KEYPOINTS), 0.9)
    if view == SIDE:  # camera on the swimmer's right: left side is the far side
        for k in KEYPOINTS:
            if k.startswith("l_"):
                conf[:, IDX[k]] = 0.62
    if camera == "underwater":
        conf = np.where(under, conf * 0.95, conf)
    else:
        # From the deck, visibility fades with depth: a shoulder rolled ~13cm under is
        # still readable, a hand at the catch (~40cm down) is not. Body units ~ metres.
        d = np.clip(J[..., 1] - SURFACE_Y, 0, None)
        fade = np.where(d <= 0.2, 1 - 0.5 * d / 0.2, np.maximum(0.35, 0.5 - (d - 0.2) * 1.5))
        conf = conf * fade
    conf *= rng.uniform(0.96, 1.0, conf.shape)
    px += rng.normal(0, jitter * W, px.shape)

    if splash:  # hand entry throws spray: wrist and elbow confidence collapse briefly
        for sd, off in (("r", 0.0), ("l", 0.5)):
            p = (t / sw.cycle_s + off) % 1.0
            hit = ((p > 0.95) | (p < 0.03)) & (rng.random(n) < 0.5 * splash)
            for j in ("wrist", "elbow"):
                i = IDX[f"{sd}_{j}"]
                conf[hit, i] = rng.uniform(0.1, 0.3, hit.sum())
                px[hit, i] += rng.normal(0, 0.03 * W, (hit.sum(), 2))

    if view == SIDE and facing == "left":
        px[..., 0] = W - px[..., 0]

    lost = rng.random((n, NUM_KEYPOINTS)) < dropout
    lost |= (rng.random(n) < frame_loss)[:, None]
    conf = np.clip(conf, 0, 1)
    conf[lost] = 0.0
    px[lost] = np.nan

    data = np.concatenate([px, conf[..., None]], -1)
    meta = {"source": "synthetic", "seed": seed, "quality": quality, "camera": camera,
            "facing": facing if view == SIDE else None, "t0": t0,
            "truth": {"faults": sw.faults, "cycle_s": sw.cycle_s, "breath_side": sw.breath_side}}
    return PoseTrack(data, fps, size, view, meta)


def main():
    ap = argparse.ArgumentParser(description="Write synthetic side and head-on pose JSON.")
    ap.add_argument("out")
    ap.add_argument("--faults", nargs="*", default=[], help="issue_id=severity, e.g. dropped_elbow=0.8")
    ap.add_argument("--quality", default="typical", choices=list(QUALITY))
    ap.add_argument("--camera", default="underwater", choices=["underwater", "deck"])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    faults = {k: float(v) for k, v in (f.split("=") for f in a.faults)}
    sw = Swimmer(faults)
    os.makedirs(a.out, exist_ok=True)
    for view in (SIDE, HEAD_ON):
        tr = generate(view, sw, quality=a.quality, camera=a.camera, seed=a.seed)
        path = os.path.join(a.out, f"synthetic_{view}_pose.json")
        with open(path, "w") as f:
            json.dump(tr.to_json(), f)
        print(path)


if __name__ == "__main__":
    main()
