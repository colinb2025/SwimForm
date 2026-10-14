"""
Synthetic freestyle pose tracks with injectable technique faults.

Lets metrics, rules, the two-view merge and anything downstream run end to end
before the fine-tuned model exists. Each fault is named by its `issue_id` in
rules/swim_rules.xlsx and takes a severity in [0, 1], so a test can ask "does a
severe dropped elbow trip the dropped_elbow rule, and does a clean swimmer trip
nothing?" against the real workbook.

This is a kinematic sketch, not a simulator: 2D projected positions whose shapes
and timings are close enough to exercise the code. Lengths are allowed to
foreshorten in projection, as they do on camera.

  python -m swimform.synthetic out/ --faults dropped_elbow=1 late_breath=0.6

Body units: shoulder-to-hip 0.5, upper arm 0.3, forearm 0.3, thigh 0.42, shin 0.40.
y points down (deeper) throughout, like image coordinates.
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

# jitter (fraction of frame), per-point dropout, splash rate, whole-frame loss
QUALITY = {
    "clean":   (0.000, 0.00, 0.0, 0.00),
    "typical": (0.003, 0.02, 1.0, 0.00),
    "poor":    (0.007, 0.08, 2.0, 0.04),
}


class Swimmer:
    def __init__(self, faults=None, cycle_s=1.6, breath_side="r", breath_every=1):
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
        """Breathing-arm phase where the head is most turned: at pull end, or later."""
        return self.pull_end + 0.50 * self["late_breath"]


# ------------------------------------------------------------------ helpers
def _smooth(s):
    return s * s * (3 - 2 * s)


def _bump(phase, center, width):
    d = (phase - center + 0.5) % 1.0 - 0.5
    return np.where(np.abs(d) < width, 0.5 * (1 + np.cos(np.pi * d / width)), 0.0)


def _breath(t, sw, arm_phase):
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
    return r * np.stack([np.cos(ang), np.sin(ang)], axis=-1)


# Side-view pull: upper-arm angle a and forearm angle b, degrees below forward
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


def _side_arm(p, sw):
    """(elbow, wrist, underwater) relative to the shoulder for arm phase p."""
    n = len(p)
    el, wr = np.zeros((n, 2)), np.zeros((n, 2))
    under = np.zeros(n, bool)
    s_end = 1.0 - 0.4 * sw["short_pull"]
    a1, b1 = np.radians(10.0), np.radians(10.0)
    E1, W1 = _polar(L_UP, a1), _polar(L_UP, a1) + _polar(L_FORE, b1)
    ae, be = _pull_angles(np.array([s_end]), sw)
    E0 = _polar(L_UP, ae)[0]
    W0 = E0 + _polar(L_FORE, be)[0]

    g = p < sw.glide
    el[g], wr[g], under[g] = E1, W1, True

    m = (p >= sw.glide) & (p < sw.pull_end)
    s = (p[m] - sw.glide) / sw.pull * s_end
    a, b = _pull_angles(s, sw)
    el[m] = _polar(L_UP, a)
    wr[m] = el[m] + _polar(L_FORE, b)
    under[m] = True

    m = p >= sw.pull_end
    s = _smooth((p[m] - sw.pull_end) / sw.recovery)
    lift = np.sin(np.pi * s)[:, None]
    se = s ** 0.6                       # elbow leads the hand
    e_n = E0 + (E1 - E0) * se[:, None] + np.array([0, -0.28]) * lift
    w_n = W0 + (W1 - W0) * s[:, None] + np.array([0, -0.12]) * lift
    # Straight-arm recovery: the arm swings over as a near-rigid spoke.
    ang = np.radians(np.interp(s, [0, 1], [np.degrees(np.arctan2(W0[1], W0[0])) % 360, 370]))
    w_f = _polar(0.58, ang)
    e_f = 0.5 * w_f
    k = sw["wide_recovery_straight_arm"]
    el[m] = (1 - k) * e_n + k * e_f
    wr[m] = (1 - k) * w_n + k * w_f
    return el, wr, under


def _side(t, sw):
    n = len(t)
    P = np.zeros((n, NUM_KEYPOINTS, 2))
    U = np.zeros((n, NUM_KEYPOINTS), bool)
    phi = (t / sw.cycle_s) % 1.0
    high, sink = sw["high_head_position"], sw["sinking_hips"]

    def put(k, xy, under=False):
        P[:, IDX[k]] = xy
        U[:, IDX[k]] = under

    bob = 0.012 * np.sin(4 * np.pi * phi)
    neck = np.stack([np.zeros(n), bob], -1)
    turn = _breath(t, sw, phi if sw.breath_side == "r" else (phi + 0.5) % 1)
    put("spine_at_neck", neck)
    head = neck + [0.20, -0.04 - 0.13 * high]
    head[:, 1] += 0.03 * turn
    put("top_of_head", head)
    waist = neck + [-0.50, 0.02 + 0.09 * high + 0.09 * sink]
    put("spine_at_waist", waist, True)
    for sd, dx in (("r", 0.01), ("l", -0.01)):
        put(f"{sd}_shoulder", neck + [dx, 0])
        put(f"{sd}_hip", waist + [dx, 0], True)
    leg_tilt = np.radians(4 + 12 * sink + 4 * high)
    for sd, off, koff in (("r", 0.0, 0.0), ("l", 0.5, np.pi)):
        e, w, u = _side_arm((phi + off) % 1, sw)
        sh = P[:, IDX[f"{sd}_shoulder"]]
        put(f"{sd}_elbow", sh + e, u)
        put(f"{sd}_wrist", sh + w, u)
        hip_a, knee_f = _kick(t, sw, koff)
        th = np.pi - leg_tilt + hip_a          # pointing back and slightly down
        hp = P[:, IDX[f"{sd}_hip"]]
        kn = hp + _polar(L_THIGH, th)
        an = kn + _polar(L_SHIN, th + knee_f)  # knee flex lifts the heel
        put(f"{sd}_knee", kn, True)
        put(f"{sd}_ankle", an, True)
    return P, U


def _head_on_arm(p, sw, sign, sh):
    """sign -1: swimmer's right arm, which appears on image-left when head-on."""
    xe = sign * (0.20 - 0.26 * sw["crossover_entry"])
    n = len(p)
    wr = np.zeros((n, 2))
    under = np.zeros(n, bool)
    g = p < sw.glide
    wr[g], under[g] = [xe, 0.06], True
    m = (p >= sw.glide) & (p < sw.pull_end)
    s = (p[m] - sw.glide) / sw.pull
    wr[m] = np.stack([np.interp(s, [0, .5, 1], [xe, sign * .08, sign * .15]),
                      np.interp(s, [0, .5, 1], [.06, .22, .12])], -1)
    under[m] = True
    m = p >= sw.pull_end
    s = _smooth((p[m] - sw.pull_end) / sw.recovery)
    wr[m] = np.stack([sign * .15 + (xe - sign * .15) * s + sign * .22 * np.sin(np.pi * s),
                      .12 - .06 * s - .30 * np.sin(np.pi * s)], -1)
    el = sh + 0.5 * (wr - sh)
    el[:, 0] += sign * 0.06
    lift = np.zeros(n)
    lift[m] = 0.08 * np.sin(np.pi * s)
    el[:, 1] -= lift
    return el, wr, under


def roll_amplitudes(sw):
    """(peak roll toward the right side, toward the left side), degrees."""
    r = 40.0 - 28.0 * sw["low_body_rotation"]
    return r, r * (1 - 0.45 * sw["rotation_asymmetry"])


def _head_on(t, sw):
    n = len(t)
    P = np.zeros((n, NUM_KEYPOINTS, 2))
    U = np.zeros((n, NUM_KEYPOINTS), bool)
    phi = (t / sw.cycle_s) % 1.0
    rr, rl = roll_amplitudes(sw)
    wave = np.sin(2 * np.pi * (phi - (sw.glide + sw.pull / 2) + 0.25))  # +1: right shoulder deepest
    roll = np.radians(np.where(wave > 0, rr, rl) * wave)

    def put(k, xy, under=False):
        P[:, IDX[k]] = xy
        U[:, IDX[k]] = under

    put("spine_at_neck", np.zeros((n, 2)))
    w = 0.20
    put("r_shoulder", np.stack([-w * np.cos(roll), w * np.sin(roll)], -1))
    put("l_shoulder", np.stack([w * np.cos(roll), -w * np.sin(roll)], -1))
    turn = _breath(t, sw, phi if sw.breath_side == "r" else (phi + 0.5) % 1)
    bside = -1.0 if sw.breath_side == "r" else 1.0
    lift = 0.03 - 0.11 * sw["head_lift_on_breath"]       # + rolls down, - lifts up
    put("top_of_head", np.stack([bside * 0.09 * turn,
                                 -0.14 - 0.05 * sw["high_head_position"] + lift * turn], -1))
    waist = np.tile([0.0, 0.10 + 0.04 * sw["sinking_hips"]], (n, 1))
    put("spine_at_waist", waist, True)
    hr, hw = 0.7 * roll, 0.13
    put("r_hip", waist + np.stack([-hw * np.cos(hr), hw * np.sin(hr)], -1), True)
    put("l_hip", waist + np.stack([hw * np.cos(hr), -hw * np.sin(hr)], -1), True)
    for sd, sign, off, koff in (("r", -1.0, 0.0, 0.0), ("l", 1.0, 0.5, np.pi)):
        sh = P[:, IDX[f"{sd}_shoulder"]]
        e, wr, u = _head_on_arm((phi + off) % 1, sw, sign, sh)
        put(f"{sd}_elbow", e, u)
        put(f"{sd}_wrist", wr, u)
        hip_a, knee_f = _kick(t, sw, koff)
        hp = P[:, IDX[f"{sd}_hip"]]
        spread = sign * 0.20 * sw["wide_kick"] * np.abs(np.sin(3 * np.pi * phi))
        kn = np.stack([hp[:, 0] * .85, hp[:, 1] + .06 + .10 * hip_a], -1)
        an = np.stack([kn[:, 0] * .85 + spread, kn[:, 1] + .05 + .06 * knee_f], -1)
        put(f"{sd}_knee", kn, True)
        put(f"{sd}_ankle", an, True)
    return P, U


# ------------------------------------------------------------------ public
def generate(view, swimmer=None, *, duration_s=12.0, fps=30.0, size=(1920, 1080),
             facing="right", camera="underwater", quality="typical", seed=0):
    """
    A synthetic PoseTrack for one camera view.

    camera: "underwater" sees below-water joints; "deck" mostly does not
            (their confidence falls under the trust threshold), as on real deck footage.
    facing: side view only — direction of travel in the image.
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
    t = np.arange(n) / fps

    if view == SIDE:
        P, U = _side(t, sw)
        scale, cx, cy = 0.40, 0.62, 0.40
    else:
        P, U = _head_on(t, sw)
        scale, cx, cy = 0.60, 0.50, 0.45

    # Body units -> pixels, isotropic so angles survive the projection.
    px = np.empty_like(P)
    px[..., 0] = (cx + P[..., 0] * scale) * W
    px[..., 1] = cy * H + P[..., 1] * scale * W

    conf = np.full((n, NUM_KEYPOINTS), 0.9)
    if view == SIDE:  # camera on the swimmer's right: left side is the far side
        for k in KEYPOINTS:
            if k.startswith("l_"):
                conf[:, IDX[k]] = 0.62
    under_factor = 0.95 if camera == "underwater" else 0.45
    conf = np.where(U, conf * under_factor, conf)
    if view == HEAD_ON and camera == "deck":
        for k in ("l_knee", "r_knee", "l_ankle", "r_ankle"):
            conf[:, IDX[k]] = 0.3
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
            "facing": facing if view == SIDE else None,
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
