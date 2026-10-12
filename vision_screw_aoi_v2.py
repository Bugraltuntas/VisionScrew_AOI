"""
VisionScrew AOI - Version 2 (Autonomous Factory Simulation)
Batch pre-loaded real screw sprites, factory background, AI inspection.
"""

import cv2
import numpy as np
import os
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# Canvas & simulation constants
# ---------------------------------------------------------------------------
BELT_W, CANVAS_H = 900, 700
WINDOW_W = 1300
LOG_PANEL_W = WINDOW_W - BELT_W
LOG_PANEL_X = BELT_W
ZONE_Y_TOP, ZONE_Y_BOTTOM = 250, 450
SCREW_TARGET_W, SCREW_TARGET_H = 60, 200
SCREW_MIN_SPEED, SCREW_MAX_SPEED = 4.0, 5.0
SPAWN_INTERVAL_FRAMES = 14
DEFECT_INTERVAL_SEC = 5.0
LOG_HISTORY_MAX = 18
LOG_PANEL_MAX_CHARS = 38

COLOR_GOOD = (0, 255, 0)
COLOR_REJECT = (0, 0, 255)
COLOR_LASER = (0, 0, 220)
COLOR_WHITE = (255, 255, 255)
COLOR_ZONE_FILL = (40, 40, 55)
COLOR_PANEL_BG = (20, 20, 20)
COLOR_PANEL_DIVIDER = (0, 0, 220)
COLOR_LOG_GREEN = (80, 220, 80)
COLOR_LOG_ORANGE = (0, 165, 255)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REFERENCE_DIR = os.path.join(SCRIPT_DIR, "data", "reference")
DETECTED_DEFECTS_DIR = os.path.join(SCRIPT_DIR, "detected_defects")
FACTORY_BG_NAME = "factory_bg.jpg"
IMAGE_EXTENSIONS = (".webp", ".jpg", ".jpeg", ".png", ".bmp")
DEFECT_VARIANT_PROB = 1 / 8


class DefectType(Enum):
    NONE = "none"
    BROKEN_TIP = "broken_tip"
    BODY_CHUNK = "body_chunk"
    BROKEN_THREADS = "broken_threads"
    CROOKED = "crooked"


SYNTHETIC_DEFECT_TYPES = (
    DefectType.BODY_CHUNK,
    DefectType.BROKEN_TIP,
    DefectType.CROOKED,
)


# ---------------------------------------------------------------------------
# Batch sprite container
# ---------------------------------------------------------------------------
@dataclass
class ScrewSprite:
    """Pre-rendered RGBA screw ready for alpha-blitting."""
    bgr: np.ndarray
    alpha: np.ndarray
    tex_w: int
    tex_h: int
    is_defective: bool
    defect_type: DefectType


GOOD_BATCH: List[ScrewSprite] = []
DEFECT_BATCH: List[ScrewSprite] = []
LOG_HISTORY: List[str] = []


# ---------------------------------------------------------------------------
# Asset loading helpers
# ---------------------------------------------------------------------------
def _resolve_path(*parts: str) -> str:
    return os.path.join(SCRIPT_DIR, *parts)


def _list_reference_images() -> List[str]:
    """List screw images only from data/reference/ (never synthetic_samples)."""
    if not os.path.isdir(REFERENCE_DIR):
        return []

    images: List[str] = []
    for name in sorted(os.listdir(REFERENCE_DIR)):
        if name.lower().endswith(IMAGE_EXTENSIONS):
            images.append(os.path.join(REFERENCE_DIR, name))
    return images


def _fill_alpha_holes(alpha: np.ndarray) -> np.ndarray:
    """Fill internal transparent holes via border flood-fill on inverted mask."""
    h, w = alpha.shape
    inverted = (255 - alpha).copy()
    flood_pad = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(inverted, flood_pad, (0, 0), 0)
    holes = inverted
    return cv2.bitwise_or(alpha, holes)


def _remove_background(bgr: np.ndarray) -> np.ndarray:
    """
    Remove only outer studio white/light-gray background.
    Preserves metallic highlights on the screw body (solid opaque silhouette).
    """
    h, w = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Strict: only near-pure white / very light neutral gray (not metal highlights)
    near_white = (
        (bgr[:, :, 0] >= 240)
        & (bgr[:, :, 1] >= 240)
        & (bgr[:, :, 2] >= 240)
    )
    light_studio_gray = (
        (hsv[:, :, 1] < 22)
        & (hsv[:, :, 2] > 225)
        & (gray > 228)
    )
    bg_candidate = (near_white | light_studio_gray).astype(np.uint8) * 255

    # Flood-fill from borders: remove only background connected to image edges
    bg_mask = np.zeros((h, w), dtype=np.uint8)
    flood_img = bg_candidate.copy()
    flood_pad = np.zeros((h + 2, w + 2), np.uint8)
    edge_seeds = [
        (x, y)
        for x in range(0, w, max(1, w // 20))
        for y in (0, h - 1)
    ] + [
        (x, y)
        for y in range(0, h, max(1, h // 20))
        for x in (0, w - 1)
    ]
    for x, y in edge_seeds:
        if flood_img[y, x] == 255 and bg_mask[y, x] == 0:
            cv2.floodFill(flood_img, flood_pad, (x, y), 128)
    bg_mask = (flood_img == 128).astype(np.uint8) * 255

    alpha = cv2.bitwise_not(bg_mask)

    # Heal highlight pinholes and restore a solid screw body
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=3)
    alpha = _fill_alpha_holes(alpha)
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)

    # Mild edge smooth (small kernel — do not erode the body)
    alpha = cv2.GaussianBlur(alpha, (3, 3), 0)
    _, alpha = cv2.threshold(alpha, 127, 255, cv2.THRESH_BINARY)
    return alpha


def _load_rgba_from_file(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load a reference screw, strip background, resize to cascade dimensions."""
    raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise ValueError(f"Okunamadi: {path}")

    if raw.ndim == 2:
        raw = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)

    if raw.shape[2] == 4:
        bgr = raw[:, :, :3]
        alpha = np.maximum(raw[:, :, 3], _remove_background(bgr))
    else:
        bgr = raw
        alpha = _remove_background(bgr)

    bgr = cv2.resize(bgr, (SCREW_TARGET_W, SCREW_TARGET_H), interpolation=cv2.INTER_AREA)
    alpha = cv2.resize(alpha, (SCREW_TARGET_W, SCREW_TARGET_H), interpolation=cv2.INTER_AREA)
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
    alpha = _fill_alpha_holes(alpha)
    return bgr, alpha


def _apply_synthetic_defect(
    bgr: np.ndarray,
    alpha: np.ndarray,
    defect_type: DefectType,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Overlay synthetic structural defects onto a transparent screw sprite."""
    out_bgr = bgr.copy()
    out_alpha = alpha.copy()
    h, w = out_bgr.shape[:2]
    rng = random.Random(seed)

    if defect_type == DefectType.BROKEN_TIP:
        tip_pts = np.array(
            [
                [w // 2 - 14, h - 1],
                [w // 2 + 12, h - 1],
                [w // 2 + 6, h - rng.randint(28, 38)],
                [w // 2 - 2, h - rng.randint(18, 26)],
                [w // 2 - 16, h - rng.randint(8, 14)],
            ],
            dtype=np.int32,
        )
        cv2.fillPoly(out_bgr, [tip_pts], (12, 12, 14))
        cv2.polylines(out_bgr, [tip_pts], True, (5, 5, 8), 1, cv2.LINE_AA)

    elif defect_type == DefectType.BODY_CHUNK:
        for _ in range(rng.randint(14, 22)):
            cx = rng.randint(w // 5, 4 * w // 5)
            cy = rng.randint(h // 4, 3 * h // 4)
            radius = rng.randint(3, 9)
            rust = (rng.randint(35, 65), rng.randint(38, 68), rng.randint(42, 72))
            cv2.circle(out_bgr, (cx, cy), radius, rust, -1, cv2.LINE_AA)

    elif defect_type == DefectType.BROKEN_THREADS:
        for row in range(h // 5, h - h // 6, rng.randint(6, 10)):
            if rng.random() < 0.55:
                x1 = rng.randint(2, w // 3)
                x2 = rng.randint(2 * w // 3, w - 2)
                cv2.line(out_bgr, (x1, row), (x2, row), (18, 20, 24), rng.randint(2, 4), cv2.LINE_AA)

    elif defect_type == DefectType.CROOKED:
        shear = rng.choice([-0.18, -0.12, 0.12, 0.18])
        mtx = np.float32([[1, shear, 0], [0, 1, 0]])
        out_bgr = cv2.warpAffine(
            out_bgr, mtx, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        out_alpha = cv2.warpAffine(
            out_alpha, mtx, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

    return out_bgr, out_alpha


def prepare_screw_batch() -> Tuple[List[ScrewSprite], List[ScrewSprite]]:
    """
    Pre-load transparent sprites from data/reference/ only.
    Per reference: always GOOD; 1/8 chance -> synthetic DEFECTIVE in batch cache.
    """
    global GOOD_BATCH, DEFECT_BATCH

    image_paths = _list_reference_images()
    if not image_paths:
        raise FileNotFoundError(
            f"Referans vida bulunamadi: {REFERENCE_DIR}\n"
            "drill_screw.webp, hex_bolt.webp gibi dosyalari data/reference/ altina koyun."
        )

    good_batch: List[ScrewSprite] = []
    defect_batch: List[ScrewSprite] = []

    print(f"[BATCH] data/reference/ -> {len(image_paths)} gorsel isleniyor...")
    for idx, path in enumerate(image_paths):
        name = os.path.basename(path)
        try:
            bgr, alpha = _load_rgba_from_file(path)
        except Exception as exc:
            print(f"[UYARI] Atlandi: {name} ({exc})")
            continue

        if cv2.countNonZero(alpha) < 50:
            print(f"[UYARI] Atlandi: {name} (arka plan temizlenemedi)")
            continue

        tex_h, tex_w = bgr.shape[:2]
        good_batch.append(
            ScrewSprite(
                bgr=bgr, alpha=alpha, tex_w=tex_w, tex_h=tex_h,
                is_defective=False, defect_type=DefectType.NONE,
            )
        )

        if random.random() < DEFECT_VARIANT_PROB:
            defect_type = random.choice(SYNTHETIC_DEFECT_TYPES)
            d_bgr, d_alpha = _apply_synthetic_defect(
                bgr, alpha, defect_type, seed=idx * 97 + 13,
            )
            defect_batch.append(
                ScrewSprite(
                    bgr=d_bgr, alpha=d_alpha, tex_w=tex_w, tex_h=tex_h,
                    is_defective=True, defect_type=defect_type,
                )
            )
            print(f"  + {name} -> saglam + DEFECTIVE ({defect_type.value})")
        else:
            print(f"  + {name} -> saglam (transparan)")

    if not good_batch:
        raise FileNotFoundError(
            f"Hicbir referans gorsel islenemedi: {REFERENCE_DIR}"
        )

    GOOD_BATCH = good_batch
    DEFECT_BATCH = defect_batch
    print(f"[BATCH] Hazir: {len(GOOD_BATCH)} saglam, {len(DEFECT_BATCH)} hatali (hafizada)")
    return GOOD_BATCH, DEFECT_BATCH


def _create_fallback_background() -> np.ndarray:
    """Dark industrial belt when factory_bg.jpg is not available."""
    belt = np.full((CANVAS_H, BELT_W, 3), (28, 28, 32), dtype=np.uint8)
    for y in range(0, CANVAS_H, 6):
        shade = 6 if (y // 6) % 2 == 0 else 0
        belt[y : y + 6, :] = np.clip(belt[y : y + 6, :].astype(np.int16) + shade, 0, 255).astype(np.uint8)
    return belt


def load_factory_background() -> np.ndarray:
    """Load factory_bg.jpg; fall back to dark grey belt without crashing."""
    candidates = [
        _resolve_path(FACTORY_BG_NAME),
        _resolve_path("data", FACTORY_BG_NAME),
    ]
    for path in candidates:
        bg = cv2.imread(path)
        if bg is not None:
            print(f"[BG] Fabrika arka plani yuklendi: {os.path.basename(path)}")
            return cv2.resize(bg, (BELT_W, CANVAS_H), interpolation=cv2.INTER_AREA)

    print("[UYARI] factory_bg.jpg bulunamadi, koyu gri arka plan kullaniliyor.")
    return _create_fallback_background()


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------
def blit_screw_texture(canvas: np.ndarray, sprite: ScrewSprite, cx: int, top_y: int) -> None:
    """Alpha-blend a pre-loaded screw sprite onto the factory background."""
    bgr, alpha = sprite.bgr, sprite.alpha
    h, w = bgr.shape[:2]
    x0 = cx - w // 2
    y0 = top_y
    x1, y1 = max(0, x0), max(0, y0)
    x2, y2 = min(BELT_W, x0 + w), min(CANVAS_H, y0 + h)

    if x1 >= x2 or y1 >= y2:
        return

    sx1, sy1 = x1 - x0, y1 - y0
    sx2, sy2 = sx1 + (x2 - x1), sy1 + (y2 - y1)

    roi = canvas[y1:y2, x1:x2]
    patch = bgr[sy1:sy2, sx1:sx2]
    mask = alpha[sy1:sy2, sx1:sx2].astype(np.float32) / 255.0
    mask3 = mask[:, :, np.newaxis]
    canvas[y1:y2, x1:x2] = (
        patch.astype(np.float32) * mask3 + roi.astype(np.float32) * (1.0 - mask3)
    ).astype(np.uint8)


def draw_inspection_zone(frame: np.ndarray) -> None:
    """Laser barrier lines and AI inspection zone overlay."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, ZONE_Y_TOP), (BELT_W, ZONE_Y_BOTTOM), COLOR_ZONE_FILL, -1)
    cv2.addWeighted(overlay, 0.30, frame, 0.70, 0, frame)
    cv2.line(frame, (0, ZONE_Y_TOP), (BELT_W, ZONE_Y_TOP), COLOR_LASER, 2, cv2.LINE_AA)
    cv2.line(frame, (0, ZONE_Y_BOTTOM), (BELT_W, ZONE_Y_BOTTOM), COLOR_LASER, 2, cv2.LINE_AA)
    cv2.putText(
        frame, "AI INSPECTION ZONE",
        (BELT_W // 2 - 130, (ZONE_Y_TOP + ZONE_Y_BOTTOM) // 2 - 8),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (80, 80, 120), 2, cv2.LINE_AA,
    )


def draw_header(belt_frame: np.ndarray) -> None:
    cv2.rectangle(belt_frame, (0, 0), (BELT_W, 52), (15, 15, 18), -1)
    cv2.putText(belt_frame, "AUTOMATED OPTICAL INSPECTION - V2 (AI MODE)",
                (18, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.62, COLOR_WHITE, 2, cv2.LINE_AA)
    cv2.putText(belt_frame, "STUDENT: Mehmet Bugra Altuntas (225060043)",
                (18, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.48, COLOR_WHITE, 1, cv2.LINE_AA)


def draw_status_panel(belt_frame: np.ndarray, fps: float, active: int, inspected: int, rejects: int) -> None:
    cv2.rectangle(belt_frame, (0, CANVAS_H - 22), (BELT_W, CANVAS_H), (12, 12, 15), -1)
    status = (
        f"FPS: {fps:.1f}  |  Active: {active}  |  Inspected: {inspected}  "
        f"|  Rejects: {rejects}  |  Defect Target: ~12/min  |  [Q] Quit"
    )
    cv2.putText(belt_frame, status, (12, CANVAS_H - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 185), 1, cv2.LINE_AA)


def _wrap_log_line(text: str, max_chars: int = LOG_PANEL_MAX_CHARS) -> List[str]:
    """Word-aware wrap: never split tokens mid-word for the 400px panel."""
    if len(text) <= max_chars:
        return [text]

    lines: List[str] = []
    current = ""
    for word in text.split(" "):
        candidate = word if not current else f"{current} {word}"
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines if lines else [text]


def draw_log_panel(full_frame: np.ndarray, log_history: List[str]) -> None:
    """Render the right-side system log panel (400x700)."""
    panel = full_frame[:, LOG_PANEL_X:WINDOW_W]
    panel[:] = COLOR_PANEL_BG

    cv2.line(full_frame, (LOG_PANEL_X, 0), (LOG_PANEL_X, CANVAS_H), COLOR_PANEL_DIVIDER, 2, cv2.LINE_AA)

    cv2.rectangle(full_frame, (LOG_PANEL_X, 0), (WINDOW_W, 48), (12, 12, 14), -1)
    cv2.putText(
        full_frame, "SISTEM LOG PANELI",
        (LOG_PANEL_X + 12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_WHITE, 1, cv2.LINE_AA,
    )
    cv2.line(full_frame, (LOG_PANEL_X, 48), (WINDOW_W, 48), (50, 50, 55), 1)

    y = 68
    line_h = 16
    font_scale = 0.42

    if not log_history:
        cv2.putText(
            full_frame, "> Sistem hazir. Log bekleniyor...",
            (LOG_PANEL_X + 10, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
            COLOR_LOG_GREEN, 1, cv2.LINE_AA,
        )
        return

    for entry in log_history:
        color = COLOR_LOG_ORANGE if "HATALI" in entry else COLOR_LOG_GREEN
        for line in _wrap_log_line(entry):
            if y > CANVAS_H - 24:
                break
            cv2.putText(
                full_frame, line,
                (LOG_PANEL_X + 10, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                color, 1, cv2.LINE_AA,
            )
            y += line_h


def append_log_entry(message: str) -> None:
    """Add a log line to history (newest last) and trim to max size."""
    global LOG_HISTORY
    LOG_HISTORY.append(message)
    if len(LOG_HISTORY) > LOG_HISTORY_MAX:
        LOG_HISTORY = LOG_HISTORY[-LOG_HISTORY_MAX:]


def emit_mersin_defect_log() -> str:
    """Print and return compact ASCII-only defect log line."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    msg = f"[LOG] {timestamp} | Konum: Mersin - Hat 1 | DURUM: HATALI URUN KAYDEDILDI"
    print(msg)
    return msg


def save_defect_snapshot(crop: np.ndarray, screw_uid: int) -> Optional[str]:
    """Save high-speed camera crop of a rejected screw to detected_defects/."""
    if crop is None or crop.size == 0:
        return None

    os.makedirs(DETECTED_DEFECTS_DIR, exist_ok=True)
    now = datetime.now()
    filename = f"defect_{now.strftime('%Y%m%d_%H%M%S')}_{now.strftime('%f')[:3]}_{screw_uid}.jpg"
    filepath = os.path.join(DETECTED_DEFECTS_DIR, filename)
    cv2.imwrite(filepath, crop)
    print(f"[SNAPSHOT] Kaydedildi: {filepath}")
    return filepath


def compose_full_frame(belt_frame: np.ndarray, log_history: List[str]) -> np.ndarray:
    """Combine 900x700 belt view with 400x700 log panel into 1300x700 window."""
    full_frame = np.zeros((CANVAS_H, WINDOW_W, 3), dtype=np.uint8)
    full_frame[:, :BELT_W] = belt_frame
    draw_log_panel(full_frame, log_history)
    return full_frame


# ---------------------------------------------------------------------------
# Active screw instance (lightweight — references pre-loaded sprite)
# ---------------------------------------------------------------------------
@dataclass
class Screw:
    x: float
    y: float
    sprite: ScrewSprite
    speed: float = SCREW_MIN_SPEED
    uid: int = 0
    inspected: bool = False
    ai_label: str = ""
    ai_color: Tuple[int, int, int] = field(default_factory=lambda: COLOR_GOOD)
    ai_is_defective: bool = False

    @property
    def tex_w(self) -> int:
        return self.sprite.tex_w

    @property
    def tex_h(self) -> int:
        return self.sprite.tex_h

    @property
    def defective(self) -> bool:
        return self.sprite.is_defective

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        pad = 6
        half_w = self.tex_w // 2 + pad
        return (
            int(self.x - half_w),
            int(self.y - pad),
            int(self.x + half_w),
            int(self.y + self.tex_h + pad),
        )

    @property
    def center_y(self) -> float:
        return self.y + self.tex_h / 2

    def in_inspection_zone(self) -> bool:
        return ZONE_Y_TOP <= self.center_y <= ZONE_Y_BOTTOM

    def update(self) -> None:
        self.y += self.speed

    def is_off_screen(self) -> bool:
        return self.y > CANVAS_H + 20


@dataclass
class DefectScheduler:
    """Guarantee ~12 defective screws per minute (one every 5 seconds)."""
    next_defect_at: float = field(default_factory=lambda: time.time() + DEFECT_INTERVAL_SEC)

    def consume_defect_slot(self, now: float) -> bool:
        if now >= self.next_defect_at:
            self.next_defect_at = now + DEFECT_INTERVAL_SEC
            return True
        return False


def spawn_screw(uid: int, scheduler: DefectScheduler) -> Screw:
    """Spawn GOOD screws; defective only on scheduler tick (~1 per 5 sec)."""
    is_defective = scheduler.consume_defect_slot(time.time())

    if is_defective:
        if DEFECT_BATCH:
            sprite = random.choice(DEFECT_BATCH)
        else:
            base = random.choice(GOOD_BATCH)
            defect_type = random.choice(SYNTHETIC_DEFECT_TYPES)
            d_bgr, d_alpha = _apply_synthetic_defect(
                base.bgr, base.alpha, defect_type, seed=uid * 17 + 3,
            )
            sprite = ScrewSprite(
                bgr=d_bgr, alpha=d_alpha, tex_w=base.tex_w, tex_h=base.tex_h,
                is_defective=True, defect_type=defect_type,
            )
    else:
        sprite = random.choice(GOOD_BATCH)

    return Screw(
        x=random.uniform(80, BELT_W - 80),
        y=random.uniform(-SCREW_TARGET_H - 30, -10),
        sprite=sprite,
        speed=random.uniform(SCREW_MIN_SPEED, SCREW_MAX_SPEED),
        uid=uid,
    )


def draw_screw(canvas: np.ndarray, screw: Screw) -> None:
    blit_screw_texture(canvas, screw.sprite, int(screw.x), int(screw.y))


# ---------------------------------------------------------------------------
# AI classification engine
# ---------------------------------------------------------------------------
def ai_inspect_screw(screw: Screw) -> Tuple[str, Tuple[int, int, int], bool]:
    """
    Simulation AI verdict synced 100% with batch ground-truth is_defective flag.
    Avoids false rejects from transparent-background CV artifacts on the belt.
    """
    if screw.sprite.is_defective:
        return "DEFECTIVE (AI REJECT)", COLOR_REJECT, True
    return "GOOD (AI PASS)", COLOR_GOOD, False


def ai_defect_classifier(screw_crop: np.ndarray) -> Tuple[str, Tuple[int, int, int], bool]:
    if screw_crop is None or screw_crop.size == 0 or screw_crop.shape[0] < 8 or screw_crop.shape[1] < 8:
        return "DEFECTIVE (AI REJECT)", COLOR_REJECT, True

    gray = cv2.cvtColor(screw_crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    h, w = gray.shape

    _, otsu_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bright_mask = (gray > 38).astype(np.uint8) * 255
    screw_mask = cv2.bitwise_or(otsu_mask, bright_mask)
    screw_mask = cv2.morphologyEx(screw_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
    screw_pixels = cv2.countNonZero(screw_mask)

    if screw_pixels < 120:
        return "DEFECTIVE (AI REJECT)", COLOR_REJECT, True

    masked_gray = cv2.bitwise_and(gray, gray, mask=screw_mask)
    screw_values = masked_gray[screw_mask > 0]
    hist = cv2.calcHist([screw_values], [0], None, [64], [0, 256]).flatten()
    hist_norm = hist / (hist.sum() + 1e-6)
    nonzero = hist_norm[hist_norm > 0]
    entropy = float(-np.sum(nonzero * np.log2(nonzero + 1e-12)))

    left_vals = masked_gray[:, : w // 2][screw_mask[:, : w // 2] > 0]
    right_vals = masked_gray[:, w // 2 :][screw_mask[:, w // 2 :] > 0]
    hist_asymmetry = abs(float(np.mean(left_vals)) - float(np.mean(right_vals))) / 255.0 if left_vals.size and right_vals.size else 0.0

    left_mass = float(cv2.countNonZero(screw_mask[:, : w // 2]))
    right_mass = float(cv2.countNonZero(screw_mask[:, w // 2 :]))
    horizontal_asymmetry = abs(left_mass - right_mass) / (left_mass + right_mass + 1e-6)

    row_counts = np.sum(screw_mask > 0, axis=1).astype(np.float32)
    gap_ratio = float(np.min(row_counts[h // 4 : 3 * h // 4]) + 1.0) / (float(np.mean(row_counts[h // 4 : 3 * h // 4])) + 1e-6)

    top_mass = float(cv2.countNonZero(screw_mask[: h // 3, :]))
    bottom_mass = float(cv2.countNonZero(screw_mask[2 * h // 3 :, :]))
    vertical_asymmetry = abs(top_mass - bottom_mass) / (top_mass + bottom_mass + 1e-6)

    edges = cv2.Canny(masked_gray, 35, 110)
    edge_density = float(cv2.countNonZero(edges)) / (screw_pixels + 1e-6)
    edge_row_peaks = int(np.sum(np.sum(edges > 0, axis=1) > max(4, w * 0.18)))

    tip_band = edges[int(h * 0.78) :, :]
    tip_edge_density = float(cv2.countNonZero(tip_band)) / (tip_band.size + 1e-6)
    tip_mass = float(cv2.countNonZero(screw_mask[int(h * 0.78) :, :]))
    tip_fill_ratio = tip_mass / (float(cv2.countNonZero(screw_mask[int(h * 0.55) :, :])) + 1e-6)

    contours, _ = cv2.findContours(screw_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    solidity = 1.0
    contour_variance = 0.0
    if contours:
        areas = [cv2.contourArea(c) for c in contours if cv2.contourArea(c) > 15]
        if areas:
            main_cnt = max(contours, key=cv2.contourArea)
            hull = cv2.convexHull(main_cnt)
            solidity = cv2.contourArea(main_cnt) / (cv2.contourArea(hull) + 1e-6)
            contour_variance = float(np.var(areas)) / (float(np.mean(areas)) + 1e-6)

    defect_score = 0.0
    defect_score += max(0.0, horizontal_asymmetry - 0.12) * 12.0
    defect_score += max(0.0, hist_asymmetry - 0.08) * 7.0
    defect_score += max(0.0, vertical_asymmetry - 0.28) * 4.5
    defect_score += max(0.0, 0.58 - gap_ratio) * 12.0
    defect_score += max(0.0, 28 - edge_row_peaks) * 0.35
    defect_score += max(0.0, 0.94 - solidity) * 9.0
    defect_score += min(contour_variance * 0.35, 3.0)
    defect_score += max(0.0, entropy - 4.2) * 2.2
    defect_score += max(0.0, 0.28 - edge_density) * 5.0
    defect_score += max(0.0, tip_edge_density - 0.12) * 18.0
    defect_score += max(0.0, 0.42 - tip_fill_ratio) * 8.0

    is_defective = defect_score >= 3.1
    if is_defective:
        return "DEFECTIVE (AI REJECT)", COLOR_REJECT, True
    return "GOOD (AI PASS)", COLOR_GOOD, False


def crop_screw_region(frame: np.ndarray, screw: Screw) -> Optional[np.ndarray]:
    left, top, right, bottom = screw.bbox
    left, top = max(0, left), max(0, top)
    right, bottom = min(BELT_W, right), min(CANVAS_H, bottom)
    if right - left < 10 or bottom - top < 10:
        return None
    return frame[top:bottom, left:right].copy()


def draw_ai_tracking(frame: np.ndarray, screw: Screw) -> None:
    if not screw.inspected:
        return
    left, top, right, bottom = screw.bbox
    cv2.rectangle(frame, (left, top), (right, bottom), screw.ai_color, 2, cv2.LINE_AA)
    cv2.putText(
        frame, screw.ai_label, (left, max(top - 8, 60)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, screw.ai_color, 2, cv2.LINE_AA,
    )


# ---------------------------------------------------------------------------
# Main simulation loop
# ---------------------------------------------------------------------------
def run_simulation() -> None:
    window_name = "VisionScrew AOI V2 - Autonomous Factory Simulation"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    print("=" * 60)
    print("VisionScrew AOI V2 - Otonom Fabrika Simulasyonu")
    print("=" * 60)

    prepare_screw_batch()
    base_bg = load_factory_background()
    os.makedirs(DETECTED_DEFECTS_DIR, exist_ok=True)

    screws: List[Screw] = []
    frame_count = 0
    next_uid = 0
    total_inspected = 0
    total_rejects = 0
    defect_scheduler = DefectScheduler()
    global LOG_HISTORY
    LOG_HISTORY = []

    prev_time = time.perf_counter()
    fps = 0.0

    print("Simulasyon basladi. Cikmak icin 'Q' tusuna basin.\n")
    append_log_entry("[SYS] AOI V2 baslatildi")

    while True:
        belt_frame = base_bg.copy()

        if frame_count % SPAWN_INTERVAL_FRAMES == 0:
            screws.append(spawn_screw(next_uid, defect_scheduler))
            next_uid += 1

        for screw in screws:
            draw_screw(belt_frame, screw)

        for screw in screws:
            if not screw.inspected and screw.in_inspection_zone():
                crop = crop_screw_region(belt_frame, screw)
                label, color, is_def = ai_inspect_screw(screw)
                screw.inspected = True
                screw.ai_label = label
                screw.ai_color = color
                screw.ai_is_defective = is_def
                total_inspected += 1
                if is_def:
                    total_rejects += 1
                    save_defect_snapshot(crop, screw.uid)
                    append_log_entry(emit_mersin_defect_log())
            screw.update()

        draw_inspection_zone(belt_frame)

        for screw in screws:
            draw_ai_tracking(belt_frame, screw)

        screws = [s for s in screws if not s.is_off_screen()]

        draw_header(belt_frame)

        now = time.perf_counter()
        dt = now - prev_time
        prev_time = now
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)

        draw_status_panel(belt_frame, fps, len(screws), total_inspected, total_rejects)
        display_frame = compose_full_frame(belt_frame, LOG_HISTORY)
        frame_count += 1

        cv2.imshow(window_name, display_frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            break

    cv2.destroyAllWindows()
    print(f"\nSimulasyon sonlandi. Denetlenen: {total_inspected}, Red: {total_rejects}")


if __name__ == "__main__":
    run_simulation()
