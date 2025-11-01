import overlay_lib
from overlay_lib import Vector2D, RgbaColor, SkDrawCircle

def callback():
    return [SkDrawCircle(Vector2D(960, 540), 10, RgbaColor(255, 255, 255, 255), 1)]

overlay = overlay_lib.Overlay(
    drawlistCallback=callback,
    refreshTimeout=1
)

overlay.spawn()