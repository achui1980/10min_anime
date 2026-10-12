"""截图匹配的纯函数：计算 dHash、汉明距离及最佳帧。

本模块不读取配置、不执行 IO；未来的图片读取与视频取帧层负责提供
9×8 灰度像素，并在匹配结果上应用阈值，与这里的计算逻辑分离。
"""

from collections.abc import Sequence

import numpy as np

HASH_WIDTH = 9
HASH_HEIGHT = 8
HASH_BYTES = 72
HASH_BITS = 64
MAX_DISTANCE = 12


def dhash(gray: bytes | Sequence[int] | np.ndarray) -> int:
    """将 9×8 灰度像素的左右比较结果按行优先编码为 64 位整数。"""
    if isinstance(gray, (bytes, bytearray)):
        pixels = np.frombuffer(gray, dtype=np.uint8)
    else:
        pixels = np.asarray(gray)
    if pixels.size != HASH_BYTES:
        raise ValueError(f"dHash 需要 {HASH_BYTES} 个灰度像素，收到 {pixels.size} 个")
    grid = pixels.reshape(HASH_HEIGHT, HASH_WIDTH).astype(np.int16)
    comparisons = grid[:, :-1] > grid[:, 1:]
    result = 0
    for bit in comparisons.ravel():
        result = (result << 1) | int(bit)
    return result


def hamming(a: int, b: int) -> int:
    """返回两个哈希之间不同位的数量。"""
    return (a ^ b).bit_count()


def best_frame(hashes: Sequence[int], ref: int) -> tuple[int, int]:
    """返回最小距离的帧索引及距离；同距取首帧，不应用匹配阈值。"""
    if len(hashes) == 0:
        raise ValueError("没有可比较的帧")
    return min(
        ((index, hamming(value, ref)) for index, value in enumerate(hashes)),
        key=lambda match: match[1],
    )
