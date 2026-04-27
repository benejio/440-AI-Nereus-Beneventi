"""
Nereus-Captured.py

Purpose:
    Runtime video monitoring pipeline for detecting a dog, classifying a likely
    pooping posture, tracking the event over time, and saving evidence images
    and optional annotated video output.

What this script does:
    1. Opens a video source from either:
       - a live camera index (for example: 0), or
       - a video file path.
    2. Runs YOLOv8 object detection on each frame to locate dogs.
    3. Expands the detected dog bounding box and passes that crop to a
       MobileNetV2 binary classifier trained to predict:
       - notpoop
       - poop
    4. Loads the classifier threshold tau from the training outputs when
       available, so the runtime decision boundary matches validation tuning.
    5. Smooths classifier probabilities with EMA and applies hysteresis
       thresholds to reduce flicker and false transitions.
    6. Tracks motion in the scene and estimates camera motion so the script can
       better reason about dog movement versus background movement.
    7. Estimates the dog's heading over multiple frames and computes a
       direction-aware crop behind the dog, intended to isolate the likely
       waste-drop region.
    8. Uses a state machine to move through event phases:
       - IDLE
       - DOG_SEEN
       - SQUAT
       - WAIT_CLEAR
    9. Saves:
       - periodic clean background snapshots,
       - post-event full-frame snapshots,
       - cropped target-region images,
       - optional annotated MP4 video output.
   10. Displays a live annotated preview window and allows quitting with 'q'.

Main components:
    - DogDetector:
        Uses YOLOv8 COCO weights to detect dogs in each frame.
    - SquatClassifier:
        Loads a trained MobileNetV2 model and predicts poop/not-poop posture
        from a dog crop, using test-time augmentation with horizontal flip.
    - MotionDetector:
        Uses frame differencing to detect sustained scene motion.
    - CameraMotion:
        Uses optical flow to estimate global camera/background motion and reduce
        confusion from camera shake or scene drift.
    - NereusWatch:
        Orchestrates video input, detection, classification, tracking, state
        transitions, image saving, and visualization.

Inputs expected:
    - YOLOv8 model weights:
        yolov8n.pt
    - Trained posture classifier weights:
        mobilenetv2_dogpoop.pt
    - Training metadata:
        tau.txt
        class_indices.json
    - Video source:
        webcam index or video file path from --src

Outputs produced:
    - Clean reference snapshots saved to OUT_DIR
    - Post-event snapshots saved to OUT_DIR
    - Region crops saved to OUT_DIR
    - Optional annotated MP4 recording saved to OUT_DIR

Important behavior notes:
    - This script does not train a model. It is an inference/runtime watcher.
    - The classifier threshold can be overridden from the command line with
      --tau, otherwise it attempts to use tau.txt from training.
    - The "butt region" crop is estimated heuristically from the dog's
      bounding box and motion direction. It is not a segmentation model.
    - Detection quality depends heavily on:
        * lighting,
        * camera angle,
        * dog size in frame,
        * motion blur,
        * model quality,
        * threshold tuning.
    - Hard-coded paths currently assume a local Windows directory layout.

Typical use:
    - Live webcam monitoring:
        python Nereus-Captured.py --src 0
    - Process a saved clip:
        python Nereus-Captured.py --src path_to_video.mp4
    - Record annotated output:
        python Nereus-Captured.py --src 0 --record
    - Force left/right aim for the target crop:
        python Nereus-Captured.py --src 0 --aim left

In plain English:
    This file watches video, finds the dog, estimates whether the dog is in a
    pooping posture, waits long enough to avoid a noisy trigger, then saves
    images around the likely event and marks the area where waste is expected
    to appear.
"""

print(f"[BOOT] Running: {__file__}")

import cv2, time, os, math, json, argparse 
from datetime import datetime, timedelta
from pathlib import Path
from ultralytics import YOLO
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import mobilenet_v2
import re
from collections import deque
from clip_event_scorer import ClipEventScorer
import numpy as np

# ---------- PATHS (relative to this Python file) ----------
BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR
TAU_TXT = BASE_DIR / "tau.txt"
CLASS_IDX_JSON = BASE_DIR / "class_indices.json"
MODEL_WEIGHTS = BASE_DIR / "mobilenetv2_dogpoop.pt"
YOLO_WEIGHTS = BASE_DIR / "yolov8n.pt"

# ---------- CONFIG ----------
OUT_DIR = str(BASE_DIR / "nereus_out")  # Folder where snapshots, crops, and optional recorded video are saved
SNAP_EVERY_MIN = 30                     # Save a clean reference frame this often when the scene is idle
MIN_MOTION_FRAMES = 2                   # Number of consecutive frames with motion before motion is considered real
MOTION_THRESH = 20                      # Pixel-difference threshold used by frame differencing for motion detection
DOG_CONF_THRESH = 0.45                  # Minimum YOLO confidence required to accept a dog detection
SQUAT_CONF_THRESH = 0.45                # Minimum classifier probability required to consider the dog in a poop/squat pose
POST_CLEAR_COOLDOWN_SEC = 1.0           # How long to wait after the dog disappears before clearing event state
POSE_HOLD_SEC = 0.04                    # How long the squat condition must remain true before triggering the event
EMA_ALPHA = 0.3                         # prob smoothing; 0.2-0.4 works well
TAU_ON_DELTA = 0.00                     # enter when >= tau*
TAU_OFF_DELTA = 0.2                     # exit when < (tau* - delta)
IOU_HOLD_MIN = 0.50                     # require bbox IoU across frames during hold
VEL_WIN = 3                             # frames to average motion
VEL_THRESH = 1.0                        # px/frame; below this = “stationary”
AIM_SHIFT_FRAC_X = 0.28                 # how much to shift horizontally (fraction of bbox w)
AIM_SHIFT_FRAC_Y = 0.10                 # how much to shift vertically   (fraction of bbox h)
BUTT_BOTTOM_FRAC = 0.45                 # portion of bbox height kept at bottom
BUTT_EXT_FRAC   = 0.30                  # extend below bbox
BUTT_WIDTH_FRAC = 0.55                  # width of the band (fraction of bbox w)
HEADING_LOOKBACK = 12                   # frames to look back for robust heading (≈0.4s at 30 fps)
HEADING_MIN_PX   = 10                   # require at least this many pixels net displacement


os.makedirs(OUT_DIR, exist_ok=True)


class CameraMotion:
    def __init__(self):
        self.prev_gray = None
        self.prev_pts  = None
        self.cum_dx = 0.0
        self.cum_dy = 0.0

    def update(self, frame, exclude_box=None):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.prev_gray is None:
            self.prev_gray = gray
            self.prev_pts = cv2.goodFeaturesToTrack(gray, 200, 0.01, 8)
            return (0.0, 0.0)

        if self.prev_pts is None or len(self.prev_pts) < 40:
            self.prev_pts = cv2.goodFeaturesToTrack(self.prev_gray, 200, 0.01, 8)

        if self.prev_pts is None:
            self.prev_gray = gray
            return (0.0, 0.0)

        # track
        nxt, st, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, self.prev_pts, None, winSize=(21,21), maxLevel=3)
        good_old = self.prev_pts[st==1] if nxt is not None else None
        good_new = nxt[st==1] if nxt is not None else None

        if exclude_box is not None and good_old is not None and len(good_old) > 0:
            x1,y1,x2,y2 = map(int, exclude_box)
            keep = []
            for i,(x,y) in enumerate(good_old.reshape(-1,2)):
                if not (x1 <= x <= x2 and y1 <= y <= y2):
                    keep.append(i)
            if keep:
                good_old = good_old[keep]
                good_new = good_new[keep]

        if good_old is None or len(good_old) < 10:
            self.prev_gray = gray
            self.prev_pts  = cv2.goodFeaturesToTrack(gray, 200, 0.01, 8)
            return (0.0, 0.0)

        flow = (good_new - good_old).reshape(-1,2)
        dx = float(np.median(flow[:,0]))
        dy = float(np.median(flow[:,1]))

        # advance
        self.prev_gray = gray
        self.prev_pts  = cv2.goodFeaturesToTrack(gray, 200, 0.01, 8)

        self.cum_dx += dx
        self.cum_dy += dy
        return (dx, dy)


# ---------- UTIL ----------
def now_str():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def save_img(path, frame):
    return cv2.imwrite(path, frame)

def draw_box(img, xyxy, color=(0,255,0), label=None):
    x1,y1,x2,y2 = map(int, xyxy)
    cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
    if label:
        cv2.putText(img, label, (x1, max(0, y1-8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def expand_box(xyxy, w_img, h_img, pad=0.18):
    x1,y1,x2,y2 = map(int, xyxy)
    w = x2 - x1; h = y2 - y1
    dx = int(pad * w); dy = int(pad * h)
    nx1 = max(0, x1 - dx); ny1 = max(0, y1 - dy)
    nx2 = min(w_img-1, x2 + dx); ny2 = min(h_img-1, y2 + dy)
    return (nx1, ny1, nx2, ny2)


def iou(a, b):
    ax1, ay1, ax2, ay2 = map(int, a)
    bx1, by1, bx2, by2 = map(int, b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    aA = area(a); bA = area(b)
    return inter / (aA + bA - inter + 1e-6)

def area(xyxy):
    x1,y1,x2,y2 = xyxy
    return max(0, x2-x1) * max(0, y2-y1)

def _safe_load_state_dict(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)

def _load_tau(default_tau=0.53):
    try:
        txt = TAU_TXT.read_text(encoding="utf-8", errors="ignore")
        m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", txt)
        if m:
            tau = float(m.group(0))
            if 0.0 <= tau <= 1.0:
                return tau
        print(f"[WARN] Invalid tau in {TAU_TXT} -> using default {default_tau}")
    except Exception as e:
        print(f"[WARN] Could not read {TAU_TXT}: {e} -> using default {default_tau}")
    return default_tau


def _load_class_indices():
    try:
        mapping = json.loads(CLASS_IDX_JSON.read_text())
        return mapping
    except Exception:
        print("[WARN] class_indices.json not found; assuming index 1 == 'poop'.")
        return {0: "notpoop", 1: "poop"}

# ---------- DOG DETECTOR ----------
class DogDetector:
    def __init__(self, conf=0.35):
        self.model = YOLO(str(YOLO_WEIGHTS))   # COCO
        self.conf = conf
        self.dog_idx = 16  # COCO class index for dog

    def infer(self, frame):
        res = self.model.predict(frame, conf=self.conf, verbose=False)[0]
        out = []
        for b in res.boxes:
            cls = int(b.cls.item())
            if cls == self.dog_idx:
                xyxy = b.xyxy.squeeze().tolist()
                conf = float(b.conf.item())
                out.append((xyxy, conf))
        return out

# --- MobileNetV2 poop/not-poop classifier on dog crops ---
class SquatClassifier:
    def __init__(self, weights_path=MODEL_WEIGHTS, device=None, thr=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = mobilenet_v2(weights=None)
        self.model.classifier[1] = nn.Linear(self.model.last_channel, 2)  # [not, poop]
        state = _safe_load_state_dict(str(weights_path), self.device)
        self.model.load_state_dict(state)
        self.model.eval().to(self.device)
        self.thr = SQUAT_CONF_THRESH if thr is None else float(thr)

        self.tx = T.Compose([
            T.ToPILImage(),
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ])

    def is_pooping_pose(self, frame, dog_xyxy):
        x1,y1,x2,y2 = map(int, dog_xyxy)
        crop = frame[max(0,y1):max(0,y2), max(0,x1):max(0,x2)]
        if crop.size == 0:
            return False, 0.0

        # OpenCV BGR -> RGB
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        with torch.inference_mode():
            x1t = self.tx(crop_rgb).unsqueeze(0).to(self.device)
            x2t = torch.flip(x1t, dims=[-1]) 
            logits1 = self.model(x1t)
            logits2 = self.model(x2t)
            p1 = torch.softmax(logits1, dim=1)[0,1]
            p2 = torch.softmax(logits2, dim=1)[0,1]
            prob = float((p1 + p2) / 2.0)

        return (prob >= self.thr), prob



# ---------- MOTION DETECTOR ----------
class MotionDetector:
    def __init__(self):
        self.prev_gray = None
        self.motion_counter = 0

    def has_motion(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7,7), 0)
        if self.prev_gray is None:
            self.prev_gray = gray
            return False
        diff = cv2.absdiff(gray, self.prev_gray)
        self.prev_gray = gray

        _, th = cv2.threshold(diff, MOTION_THRESH, 255, cv2.THRESH_BINARY)
        motion_pixels = cv2.countNonZero(th)
        moving = motion_pixels > 2000  # tune for scene
        if moving:
            self.motion_counter += 1
        else:
            self.motion_counter = 0
        return self.motion_counter >= MIN_MOTION_FRAMES

# ---------- REGION CROPPING ----------
def crop_butt_region(full_frame, dog_xyxy, ux=0.0, uy=0.0, aim_override="auto"):
    """
    Direction-aware butt crop using heading (ux,uy).
    Ensures the final crop is BEHIND the dog (opposite heading).
    """
    H, W = full_frame.shape[:2]
    x1,y1,x2,y2 = map(int, dog_xyxy)
    w, h = x2-x1, y2-y1
    if w < 10 or h < 10:
        return None

    # Base band near the rump area
    band_top = y1 + int((1.0 - BUTT_BOTTOM_FRAC) * h)
    band_bot = min(H-1, y2 + int(BUTT_EXT_FRAC * h))
    cx = (x1 + x2) // 2
    half_w = int(0.5 * BUTT_WIDTH_FRAC * w)
    cy_band = (band_top + band_bot) // 2

    # Initial shift
    if aim_override == "left":
        dx = -int(AIM_SHIFT_FRAC_X * w)
        dy =  int(-uy * AIM_SHIFT_FRAC_Y * h)
    elif aim_override == "right":
        dx =  int(AIM_SHIFT_FRAC_X * w)
        dy =  int(-uy * AIM_SHIFT_FRAC_Y * h)
    else:
        dx = int(-ux * AIM_SHIFT_FRAC_X * w)     # opposite heading = butt side
        dy = int(-uy * AIM_SHIFT_FRAC_Y * h)

    # Build box with initial shift
    bx1 = max(0, cx - half_w + dx)
    bx2 = min(W-1, cx + half_w + dx)
    by1 = max(0, band_top + dy)
    by2 = min(H-1, band_bot + dy)

    # ---- BEHIND CHECK ----
    # vector from dog center to aim center
    ax = (bx1 + bx2) // 2
    ay = (by1 + by2) // 2
    vx, vy = ax - cx, ay - cy_band
    dot = vx * ux + vy * uy
    # If dot>0, box is AHEAD (in heading direction). Flip horizontally (and vertically a bit) to force BEHIND.
    if aim_override == "auto" and (ux*ux + uy*uy) > 1e-6 and dot > 0:
        dx, dy = -dx, -dy
        bx1 = max(0, cx - half_w + dx)
        bx2 = min(W-1, cx + half_w + dx)
        by1 = max(0, band_top + dy)
        by2 = min(H-1, band_bot + dy)

    if bx2 - bx1 < 10 or by2 - by1 < 10:
        return None
    return full_frame[by1:by2, bx1:bx2].copy(), (bx1,by1,bx2,by2)


# ---------- ALARM ----------
def alarm():
    print("[ALARM] Dog detected!")

# ---------- STATE MACHINE ----------
class States:
    IDLE = "IDLE"
    DOG_SEEN = "DOG_SEEN"
    SQUAT = "SQUAT"
    WAIT_CLEAR = "WAIT_CLEAR"

# ---------- INPUT PARSING (camera vs file) ----------  # NEW
def parse_src(src_str: str):
    """
    If src_str is all digits -> camera index (int).
    Otherwise -> treat as file path.
    Examples:
      --src 0           (default webcam)
      --src 1           (other camera)
      --src C:/clip.mp4 (video file)
    """
    if src_str.isdigit():
        return int(src_str)
    return src_str

class NereusWatch:
    def __init__(self, src, tau_override=None, record=False, aim_override="auto"):
        # Pull tau* from training outputs
        global SQUAT_CONF_THRESH
        SQUAT_CONF_THRESH = _load_tau(default_tau=SQUAT_CONF_THRESH)
        self.class_map = _load_class_indices()
        print(f"[INIT] tau*={SQUAT_CONF_THRESH:.2f}  class_map={self.class_map}")

        self.src = src                              # NEW
        self.is_file = not isinstance(src, int)     # NEW
        self.cap = cv2.VideoCapture(self.src)
        self.pose_hold_start = None

        self.motion = MotionDetector()
        self.detector = DogDetector(conf=DOG_CONF_THRESH)
        self.squat = SquatClassifier(weights_path=MODEL_WEIGHTS, thr=SQUAT_CONF_THRESH)
        self.clip_scorer = ClipEventScorer()
        if tau_override is not None:
            self.squat.thr = float(tau_override)
        # recompute hysteresis gates from current thr
        self.tau_on  = self.squat.thr + TAU_ON_DELTA
        self.tau_off = max(0.0, self.squat.thr - TAU_OFF_DELTA)

        self.clip_ema = {
            "pee": 0.0,
            "poop": 0.0,
            "neutral": 0.0,
        }
        self.state = States.IDLE
        self.last_clean_snapshot_time = datetime.min
        self.last_clean_path = None
        self.target_region_box = None
        self.last_seen_time = datetime.min
        self.prev_box = None
        self.prev_center = None
        self.prob_ema = None
        self.is_squatting = False  # hysteresis memory

        self.lock_dir = None  # (ux, uy) frozen at hold start
        self.center_hist = deque(maxlen=VEL_WIN)
        self.last_dir = (0.0, 0.0)   # unit vector of last reliable heading

        self.aim_override = aim_override
        self.cam = CameraMotion()


        assert self.cap.isOpened(), f"Video source not opened: {self.src}"
        self.record = record
        # figure out fps/size
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps < 1:  # some webcams/files report 0
            fps = 30.0
        w  = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h  = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.writer = None
        if self.record:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out_path = os.path.join(OUT_DIR, f"{now_str()}_annotated.mp4")
            self.writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
            print(f"[REC] Writing annotated MP4 to: {out_path}")

    def maybe_clean_snapshot(self, frame, motion):
        due = datetime.now() - self.last_clean_snapshot_time >= timedelta(minutes=SNAP_EVERY_MIN)
        if due and not motion:
            path = os.path.join(OUT_DIR, f"{now_str()}_clean.jpg")
            save_img(path, frame)
            self.last_clean_snapshot_time = datetime.now()
            self.last_clean_path = path
            print(f"[SNAP] Clean snapshot saved: {path}")

    def process(self):
        while True:
            ok, frame = self.cap.read()
            if not ok:
                # If it's a file, stop at EOF; if it's a camera, retry briefly  # NEW
                if self.is_file:
                    print("[INFO] End of video or read error; exiting.")
                    break
                else:
                    time.sleep(0.05)
                    continue

            motion = self.motion.has_motion(frame)
            self.maybe_clean_snapshot(frame, motion)

            # Detect dogs
            dogs = self.detector.infer(frame)
            dogs = [(b,c) for (b,c) in dogs if c >= DOG_CONF_THRESH]
            dogs.sort(key=lambda bc: area(bc[0]), reverse=True)

            if self.state == States.IDLE:
                if dogs:
                    alarm()
                    self.state = States.DOG_SEEN
                    self.last_seen_time = datetime.now()

            elif self.state == States.DOG_SEEN:
                if dogs:
                    self.last_seen_time = datetime.now()
                    box, conf = dogs[0]
                    H, W = frame.shape[:2]
                    box_clf = expand_box(box, W, H, pad=0.18)

                    is_squat_raw, prob = self.squat.is_pooping_pose(frame, box_clf)

                    # --- EMA smoothing ---
                    if self.prob_ema is None:
                        self.prob_ema = prob
                    else:
                        self.prob_ema = EMA_ALPHA * prob + (1 - EMA_ALPHA) * self.prob_ema

                    # --- Hysteresis on smoothed prob ---
                    if not self.is_squatting:
                        is_squat = self.prob_ema >= self.tau_on
                    else:
                        is_squat = self.prob_ema >= self.tau_off
                    self.is_squatting = is_squat

                    # --- Pose hold timing ---
                    now = datetime.now()
                    if is_squat:
                        if self.pose_hold_start is None:
                            self.pose_hold_start = now
                        held = (now - self.pose_hold_start).total_seconds()
                    else:
                        self.pose_hold_start = None
                        self.lock_dir = None
                        held = 0.0

                    x1c, y1c, x2c, y2c = map(int, box_clf)
                    clip_top = y1c + int(0.40 * (y2c - y1c))
                    clip_crop = frame[max(0, clip_top):max(0, y2c), max(0, x1c):max(0, x2c)]

                    clip_label = "none"
                    clip_scores = {"pee": 0.0, "poop": 0.0, "neutral": 0.0}
                    clip_allowed = (
                        clip_crop.size != 0
                        and is_squat
                        and held >= POSE_HOLD_SEC
                    )

                    if clip_allowed:
                        clip_result = self.clip_scorer.score_bgr_crop(clip_crop)
                        clip_label = clip_result.label
                        clip_scores = clip_result.scores

                        for key in self.clip_ema:
                            self.clip_ema[key] = 0.10 * clip_scores[key] + 0.90 * self.clip_ema[key]

                        clip_event_label = "none"
                        clip_event_score = max(self.clip_ema["pee"], self.clip_ema["poop"])
                        clip_runner_up = min(self.clip_ema["pee"], self.clip_ema["poop"])

                        if self.clip_ema["pee"] >= self.clip_ema["poop"]:
                            clip_event_label = "pee"
                        else:
                            clip_event_label = "poop"

                        clip_is_event = (
                            clip_event_score >= 0.12 and
                            clip_event_score > self.clip_ema["neutral"] + 0.04 and
                            clip_event_score > clip_runner_up + 0.02
                        )

                        if clip_is_event:
                            print(
                                f"[CLIP-CONFIRM] label={clip_event_label} "
                                f"score={clip_event_score:.3f} "
                                f"pee={self.clip_ema['pee']:.3f} "
                                f"poop={self.clip_ema['poop']:.3f} "
                                f"neutral={self.clip_ema['neutral']:.3f}"
                            )
                    else:
                        clip_label = "gated"
                        clip_scores = {"pee": 0.0, "poop": 0.0, "neutral": 0.0}
                        clip_event_label = "none"
                        clip_event_score = 0.0
                        clip_is_event = False

                    draw_box(frame, box, (0,255,0), None)

                    x1b, y1b, x2b, y2b = map(int, box)
                    label_line1 = f"dog {conf:.2f} | mob={prob:.2f} ema={self.prob_ema:.2f}"
                    label_line2 = f"clip={clip_event_label}:{clip_event_score:.2f}"
                    label_line3 = f"p={self.clip_ema['pee']:.2f} o={self.clip_ema['poop']:.2f} n={self.clip_ema['neutral']:.2f}"

                    text_y1 = y1b - 40
                    text_y2 = y1b - 25
                    text_y3 = y1b - 10

                    if text_y1 < 20:
                        text_y1 = min(frame.shape[0] - 54, y2b + 16)
                        text_y2 = min(frame.shape[0] - 36, y2b + 34)
                        text_y3 = min(frame.shape[0] - 18, y2b + 52)

                    cv2.putText(frame, label_line1, (x1b, text_y1),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                    cv2.putText(frame, label_line2, (x1b, text_y2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                    cv2.putText(frame, label_line3, (x1b, text_y3),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

                    now = datetime.now()

                    # estimate camera motion (exclude dog box if we have one)
                    exclude = dogs[0][0] if dogs else None
                    cam_dx, cam_dy = self.cam.update(frame, exclude_box=exclude)

                    # dog center in image coords
                    cx = int((box[0] + box[2]) / 2)
                    cy = int((box[1] + box[3]) / 2)

                    # **stabilized center** (subtract cumulative camera motion)
                    stab_cx = cx - self.cam.cum_dx
                    stab_cy = cy - self.cam.cum_dy
                    self.center_hist.append((stab_cx, stab_cy))


                    # velocity estimate for direction-aware crop (below)
                    cx = int((box[0] + box[2]) / 2)
                    cy = int((box[1] + box[3]) / 2)
                    self.center_hist.append((cx, cy))

                    # mean velocity over window
                    if len(self.center_hist) >= 2:
                        diffs = [(self.center_hist[i+1][0]-self.center_hist[i][0],
                                  self.center_hist[i+1][1]-self.center_hist[i][1])
                                 for i in range(len(self.center_hist)-1)]
                        vx_mean = sum(d[0] for d in diffs) / len(diffs)
                        vy_mean = sum(d[1] for d in diffs) / len(diffs)
                    else:
                        vx_mean, vy_mean = 0.0, 0.0

                    # choose direction: if moving, use current; else keep last reliable
                    speed = (vx_mean**2 + vy_mean**2) ** 0.5
                    if speed >= VEL_THRESH:
                        ux, uy = self._robust_heading()
                        self.last_dir = (ux, uy)
                    else:
                        ux, uy = self.last_dir

                    self.prev_center = cx

                    if is_squat:
                        cv2.putText(frame, f"squat hold: {held:.1f}/{POSE_HOLD_SEC:.1f}s",
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

                        if held >= POSE_HOLD_SEC:
                            ux_use, uy_use = self._robust_heading(lookback=4, min_px=2)

                            if self.lock_dir is not None:
                                ux_use = 0.6 * ux_use + 0.4 * self.lock_dir[0]
                                uy_use = 0.6 * uy_use + 0.4 * self.lock_dir[1]

                            if abs(ux_use) < 0.15:
                                ux_vote = self._face_lr_from_edges(frame, box)
                                if ux_vote != 0.0:
                                    ux_use = ux_vote

                            got = crop_butt_region(frame, box, ux_use, uy_use, aim_override=self.aim_override)
                            if got:
                                _, reg = got
                                self.target_region_box = reg
                                self.state = States.SQUAT
                                self.pose_hold_start = None
                                print(f"[POSE] Held {POSE_HOLD_SEC:.2f}s (ema={self.prob_ema:.2f}); region @ {reg}.")
                    else:
                        self.pose_hold_start = None
                        self.lock_dir = None

                else:
                    if (datetime.now() - self.last_seen_time).total_seconds() > POST_CLEAR_COOLDOWN_SEC:
                        self.state = States.IDLE
                        self.target_region_box = None
                        self.pose_hold_start = None
                        self.is_squatting = False
                        self.prev_box = None
                        self.prob_ema = None

            elif self.state == States.SQUAT:
                if dogs:
                    self.last_seen_time = datetime.now()
                    for (b, c) in dogs:
                        draw_box(frame, b, (0,165,255), "dog (squat phase)")
                else:
                    if (datetime.now() - self.last_seen_time).total_seconds() > POST_CLEAR_COOLDOWN_SEC:
                        post_path = os.path.join(OUT_DIR, f"{now_str()}_post.jpg")
                        save_img(post_path, frame)
                        print(f"[POST] Post-event snapshot saved: {post_path}")

                        if self.target_region_box is not None:
                            x1,y1,x2,y2 = self.target_region_box
                            crop = frame[y1:y2, x1:x2].copy()
                            crop_path = os.path.join(OUT_DIR, f"{now_str()}_region.jpg")
                            save_img(crop_path, crop)
                            if self.last_clean_path is not None:
                                clean = cv2.imread(self.last_clean_path)
                                if clean is not None:
                                    clean_crop = clean[y1:y2, x1:x2]
                                    clean_crop_path = os.path.join(OUT_DIR, f"{now_str()}_region_clean.jpg")
                                    save_img(clean_crop_path, clean_crop)
                                    print(f"[CROP] Region crops saved: {crop_path} (post), {clean_crop_path} (clean)")
                                else:
                                    print("[WARN] Could not load last clean snapshot for crop comparison.")
                        self.state = States.IDLE
                        self.target_region_box = None

            # ----- Optional: visualize (press q to quit) -----
            disp = frame.copy()
            if self.target_region_box:
                draw_box(disp, self.target_region_box, (255,0,0), "butt-aim region")

            # write annotated frame
            if self.writer is not None:
                self.writer.write(disp)

            cv2.imshow("Nereus Watch (q=quit)", disp)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        if getattr(self, "writer", None) is not None:
            self.writer.release()
        cv2.destroyAllWindows()

    def _robust_heading(self, lookback=None, min_px=None):
        look = lookback if lookback is not None else HEADING_LOOKBACK
        need = min_px if min_px is not None else HEADING_MIN_PX
        if len(self.center_hist) < 2:
            return self.last_dir
        look = min(look, len(self.center_hist)-1)
        x0, y0 = self.center_hist[-look-1]
        x1, y1 = self.center_hist[-1]
        dx, dy = x1 - x0, y1 - y0
        dist = (dx*dx + dy*dy) ** 0.5
        if dist >= need:
            ux, uy = dx / max(dist, 1e-6), dy / max(dist, 1e-6)
            self.last_dir = (ux, uy)
            return (ux, uy)
        return self.last_dir

    def _face_lr_from_edges(self, frame, box):
        x1,y1,x2,y2 = map(int, box)
        if x2-x1 < 10 or y2-y1 < 10:
            return 0.0
        crop = frame[y1:y2, x1:x2]
        H, W = crop.shape[:2]
        head_band = crop[: max(8, H//2), :]
        gray = cv2.cvtColor(head_band, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 60, 180)
        left_edges  = edges[:, :W//2].sum()
        right_edges = edges[:, W//2:].sum()
        if abs(int(left_edges) - int(right_edges)) < 500:
            return 0.0
        return 1.0 if right_edges > left_edges else -1.0


if __name__ == "__main__":
    # CLI to select source (camera index or video file)
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="0",
                   help="Camera index (e.g., 0) or path to a video file")
    p.add_argument("--tau", type=float, default=None,
                   help="Override tau* (0..1); bypass tau.txt")
    p.add_argument("--record", action="store_true",
                   help="Save annotated video to OUT_DIR as MP4")
    p.add_argument("--aim", choices=["auto","left","right"], default="auto",
               help="Override aim side for butt region")
    args = p.parse_args()
    src = parse_src(args.src)
    NereusWatch(
        src,
        tau_override=args.tau,
        record=args.record,
        aim_override=args.aim,
    ).process()
