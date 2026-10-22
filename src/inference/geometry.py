"""Canvas and timing arithmetic shared by the CLI and lightweight WebUI."""
import math

NATIVE_FPS = 24
CANVAS_MULTIPLE = 32
DEFAULT_WIDTH, DEFAULT_HEIGHT = 1344, 768
EXPORT_FPS = (12, 24, 25, 30, 48, 60)


def validate_canvas(width: int, height: int) -> None:
    if any(type(side) is not int or side < 32 or side > 4096
           or side % CANVAS_MULTIPLE for side in (width, height)):
        raise ValueError("width and height must be multiples of 32 between 32 and 4096")
    if not 0.25 <= width / height <= 4:
        raise ValueError("canvas aspect ratio must be between 1:4 and 4:1")


def frames_for_duration(seconds: float) -> int:
    """Nearest valid 17*n+5 native frame count; half-way values round upward."""
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("duration must be finite and positive")
    return max(22, 5 + 17 * math.floor((seconds * NATIVE_FPS - 5) / 17 + 0.5))


def export_frame_count(native_frames: int, fps: int) -> int:
    """Resample the native timeline, with at most half an output-frame rounding error."""
    return max(1, math.floor(native_frames * fps / NATIVE_FPS + 0.5))
