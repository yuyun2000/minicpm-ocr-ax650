"""MiniCPM-OCR 官方图像切片逻辑的独立复刻（纯 numpy + PIL，不依赖 torch/transformers）。

来源：``transformers/models/minicpm_ocr/image_processing_minicpm_ocr.py``
（transformers 5.22.2，``MiniCPMOcrImageProcessor``）。本模块把其切片相关函数逐行复刻，
供 AX650 板端（aarch64 / Python 3.10 / numpy + PIL）在没有 torch 的环境下生成与官方
processor **逐像素对齐的输入**（全局图 + 最多 ``max_slice_nums`` 个切片）。

复刻范围（与官方实现一一对应）
------------------------------------------------------------------
- ``ensure_divide``                         官方同名函数
- ``find_best_resize``                      官方同名方法（height/width 顺序一致）
- ``get_refine_size``                       官方同名方法
- ``get_sliced_grid``                       官方同名方法
- ``reshape_by_patch``                      官方同名方法（numpy 版，替代 torch.unfold）
- ``divide_to_patches``                     官方 image_transforms.divide_to_patches
- ``adaptive_upsize_resolution``            官方同名函数（纯 numpy 版，替代 cv2）
- ``_resolve_tile_size``                    官方同名方法（small_image_upsize="adaptive"）
- ``_preprocess`` 的 slice 分支             本模块 ``slice_image`` / ``preprocess``

与官方实现的已知差异（都已量化，见 notes/minicpm-ocr/slicing-verification.md）
------------------------------------------------------------------
1. **resize 后端**：官方走 torchvision ``tvF.resize(..., BICUBIC, antialias=True)``；
   本模块用 PIL ``Image.resize(..., BICUBIC)``。二者都是带抗锯齿的 bicubic，
   实测 uint8 最大差 1 LSB、仅 0.1%~0.3% 像素不同（见验证笔记）。
   归一化后最大差 ≈ 2/255 ≈ 0.0078，不影响 NPU 输入。
2. **adaptive_upsize_resolution**：官方依赖 OpenCV（``cv2.threshold`` OTSU +
   ``cv2.connectedComponentsWithStats``）。本模块用纯 numpy 实现：
   - Otsu：与 OpenCV 相同的直方图/类间方差最大化算法，argmax 平局取第一个阈值；
   - 灰度：逐位复刻官方 ITU-R 601-2 整数公式（等价 ``PIL.Image.convert("L")``）；
   - 8 连通域：行程（run）+ 并查集两遍算法，统计量（left/top/width/height/area）与
     ``connectedComponentsWithStats(connectivity=8)`` 完全一致。
   实测在两页真实文档 + 6 类合成图上返回值与官方 **0 差异**，代价是慢 2~5 倍
   （448² 约 8 ms vs cv2 2 ms；整页 2000×3083 约 0.17 s vs 0.04 s，见笔记）。
   该路径仅当 ``small_image_upsize="adaptive"`` 时启用；官方默认是 ``"always"``。
3. **不涉及的部分**：``<image_id>`` 占位符、``<image>`` token 拼接属于 processor 层
   （``MiniCPMOcrProcessor``），本模块只负责“图像 + 切片几何 + 归一化后的 patch 张量”。
   板端拼 prompt 时按 slice 顺序插入占位符即可。
4. **输入约定**：本模块接收 PIL Image（与官方 ``do_convert_rgb=True`` 行为一致，内部
   统一 ``convert("RGB")``）。EXIF 方向校正由调用方负责（官方 processor 也不做）。

用法
------------------------------------------------------------------
>>> from PIL import Image
>>> from slicing import slice_image
>>> plan = slice_image(Image.open("page.jpg"))
>>> plan.grid, plan.num_slices, plan.total_tokens
([4, 2], 9, 2276)
>>> for s in plan.slices:
...     print(s.index, s.is_global, s.size, s.grid_h, s.grid_w, s.num_tokens)

>>> out = preprocess(Image.open("page.jpg"))   # 板端可直接喂 numpy
>>> out["pixel_values"].shape                 # (1, 3, 14, 127456)，与官方一致
>>> out["target_sizes"], out["grids"], out["num_patches_per_image"]
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from PIL import Image

__all__ = [
    "ensure_divide",
    "find_best_resize",
    "get_refine_size",
    "get_sliced_grid",
    "divide_to_patches",
    "reshape_by_patch",
    "adaptive_upsize_resolution",
    "merge_factor",
    "token_divisor",
    "tokens_for_grid",
    "Slice",
    "SlicePlan",
    "slice_image",
    "preprocess",
]

# ---------------------------------------------------------------------------
# 与官方 processor 默认值一致的常量
# ---------------------------------------------------------------------------
DEFAULT_SCALE_RESOLUTION = 448
DEFAULT_MAX_SLICE_NUMS = 9
DEFAULT_PATCH_SIZE = 14
DEFAULT_DOWNSAMPLE_MODE = "4x"
DEFAULT_SMALL_IMAGE_UPSIZE = "always"
VALID_SMALL_IMAGE_UPSIZE = ("always", "never", "adaptive")

# 官方 adaptive 字形分类器的标定常量（逐字照抄）
SMALL_FONT_PX = 12.0
DENSE_GLYPHS_PER_KPX = 2.0
TINY_PIXELS = 6272
TINY_RESOLUTION = 84

# 官方 resample = PILImageResampling.BICUBIC
_RESAMPLE = Image.BICUBIC


# ---------------------------------------------------------------------------
# 基础几何函数（与官方逐行一致）
# ---------------------------------------------------------------------------
def merge_factor(downsample_mode: str) -> int:
    """官方 ``find_best_resize`` 里的 merge_factor：4x→2，16x→4。

    含义：patch 网格对齐到 ``patch_size * merge_factor``；"4x" 只做一次 2×2 合并，
    "16x" 做两次（ViT insert merger + downsample MLP）。
    """
    if downsample_mode == "4x":
        return 2
    if downsample_mode == "16x":
        return 4
    raise ValueError(f"unsupported downsample_mode: {downsample_mode!r} (expected '4x' or '16x')")


def token_divisor(downsample_mode: str) -> int:
    """image token 数 = patch 数 // token_divisor。

    "4x"  → 2×2 = 4 个 patch 合成 1 个 image token（divisor=4）；
    "16x" → 两次 2×2 → divisor=16。
    """
    mf = merge_factor(downsample_mode)
    return mf * mf


def tokens_for_grid(grid_h: int, grid_w: int, downsample_mode: str = DEFAULT_DOWNSAMPLE_MODE) -> int:
    """把 patch 网格 (gh, gw) 换算成 image token 数（整除）。"""
    return (grid_h * grid_w) // token_divisor(downsample_mode)


def ensure_divide(length: int, divisor: int) -> int:
    """官方 ``ensure_divide``：四舍五入到 divisor 的整数倍，且至少为 divisor。"""
    return max(round(length / divisor) * divisor, divisor)


def find_best_resize(
    image_size: tuple[float, float],
    scale_resolution: int,
    patch_size: int = DEFAULT_PATCH_SIZE,
    downsample_mode: str = DEFAULT_DOWNSAMPLE_MODE,
    allow_upscale: bool = False,
) -> tuple[int, int]:
    """官方 ``find_best_resize``：返回 resize 后的 ``(height, width)``。

    与官方逐行一致（参数顺序也一致：image_size 是 (height, width)）。
    注意 ``height`` 先算、``width = int(height * aspect_ratio)`` 用的是**截断后**的
    height，这个顺序会影响 1~2 px，必须照抄。
    """
    height, width = image_size
    if (height * width > scale_resolution * scale_resolution) or allow_upscale:
        aspect_ratio = width / height
        height = int(scale_resolution / math.sqrt(aspect_ratio))
        width = int(height * aspect_ratio)
    mf = merge_factor(downsample_mode)
    best_width = ensure_divide(width, patch_size * mf)
    best_height = ensure_divide(height, patch_size * mf)
    return best_height, best_width


def get_refine_size(
    image_size: tuple[float, float],
    grid: Sequence[int],
    scale_resolution: int,
    patch_size: int = DEFAULT_PATCH_SIZE,
    downsample_mode: str = DEFAULT_DOWNSAMPLE_MODE,
    allow_upscale: bool = False,
) -> tuple[int, int]:
    """官方 ``get_refine_size``：切片前把原图 resize 到 ``(h, w)``，随后按 grid 均分。

    返回 ``(refine_height, refine_width)``；切开后每片恰好是
    ``(refine_height // grid_y, refine_width // grid_x)``。
    """
    height, width = image_size
    grid_y, grid_x = grid
    refine_width = ensure_divide(width, grid_x)
    refine_height = ensure_divide(height, grid_y)

    best_height, best_width = find_best_resize(
        image_size=(refine_height / grid_y, refine_width / grid_x),
        scale_resolution=scale_resolution,
        patch_size=patch_size,
        downsample_mode=downsample_mode,
        allow_upscale=allow_upscale,
    )
    return best_height * grid_y, best_width * grid_x


def get_sliced_grid(
    image_size: tuple[int, int],
    max_slice_nums: int = DEFAULT_MAX_SLICE_NUMS,
    scale_resolution: int = DEFAULT_SCALE_RESOLUTION,
) -> list[int] | None:
    """官方 ``get_sliced_grid``：返回 ``[grid_y(行数), grid_x(列数)]``，或 None（不切）。

    在 ``multiple-1 / multiple / multiple+1`` 个切片里，枚举所有行列因子组合，
    选让 ``log(w/h)`` 与 ``log(cols/rows)`` 误差最小的；平局取行数更多的。
    """
    original_height, original_width = image_size
    log_ratio = math.log(original_width / original_height)
    ratio = original_width * original_height / (scale_resolution * scale_resolution)
    multiple = min(math.ceil(ratio), max_slice_nums)
    if multiple <= 1:
        return None

    best_grid = [1, 1]
    min_error = float("inf")
    for num_slices in [multiple - 1, multiple, multiple + 1]:
        if num_slices == 1 or num_slices > max_slice_nums:
            continue
        for num_rows in range(1, num_slices + 1):
            if num_slices % num_rows == 0:
                num_cols = num_slices // num_rows
                error = abs(log_ratio - math.log(num_cols / num_rows))
                if error < min_error:
                    best_grid = [num_rows, num_cols]
                    min_error = error
                elif error == min_error and num_rows > best_grid[0]:
                    best_grid = [num_rows, num_cols]
    return best_grid


# ---------------------------------------------------------------------------
# 切图 / patchify
# ---------------------------------------------------------------------------
def divide_to_patches(image: Image.Image, patch_height: int, patch_width: int) -> list[Image.Image]:
    """官方 ``divide_to_patches``：按行优先顺序把图切成固定大小的片段。

    与官方一致：最后一行/列允许出现不足 patch 的残余片段（正常路径下
    refine 尺寸一定能被 grid 整除，不会发生）。
    """
    width, height = image.size
    patches: list[Image.Image] = []
    for i in range(0, height, patch_height):
        for j in range(0, width, patch_width):
            patches.append(image.crop((j, i, min(j + patch_width, width), min(i + patch_height, height))))
    return patches


def reshape_by_patch(pixels: np.ndarray, patch_size: int = DEFAULT_PATCH_SIZE) -> np.ndarray:
    """官方 ``reshape_by_patch`` 的 numpy 版：``[C, H, W]`` → ``[C, patch, H*W/patch]``。

    等价于 torch ``unfold`` + ``reshape(C, p, p, -1).permute(0,1,3,2)``：
    对每个 patch 行 kh，按 patch 编号 l 行优先依次拼接，末尾排 kw。

    输出语义（与官方一致）::

        out[c, kh, (bh * gw + bw) * patch + kw] == pixels[c, bh*patch + kh, bw*patch + kw]
    """
    c, h, w = pixels.shape
    if h % patch_size or w % patch_size:
        raise ValueError(f"shape {(h, w)} not divisible by patch_size={patch_size}")
    gh, gw = h // patch_size, w // patch_size
    # (c, bh, kh, bw, kw) → (c, kh, bh, bw, kw) → (c, kh, (bh*gw+bw)*kw)
    return (
        pixels.reshape(c, gh, patch_size, gw, patch_size)
        .transpose(0, 2, 1, 3, 4)
        .reshape(c, patch_size, gh * gw * patch_size)
    )


# ---------------------------------------------------------------------------
# adaptive_upsize_resolution：纯 numpy 复刻（官方依赖 cv2）
# ---------------------------------------------------------------------------
# 官方口径：小图/切片是否值得放大，取决于字形高度与密度；用 OTSU 二值化 +
# 8 连通域统计 “字” 的包围盒。下面三个函数分别对应 cv2 的三步：
#   cv2.threshold(THRESH_OTSU) → _otsu_threshold
#   cv2.connectedComponentsWithStats(connectivity=8) → _connected_components_8
#   ITU-R 601-2 整数灰度 → _luma_itu601
# 实测（见验证笔记）与 OpenCV 结果完全一致。
def _luma_itu601(chw: np.ndarray) -> np.ndarray:
    """官方逐字复刻的 ITU-R 601-2 整数亮度公式（与 ``PIL.Image.convert("L")`` 一致）。"""
    image = chw.astype(np.int32)
    gray = ((image[0] * 19595 + image[1] * 38470 + image[2] * 7471 + 32768) >> 16).astype(np.uint8)
    return gray


def _otsu_threshold(gray: np.ndarray) -> int:
    """Otsu 阈值，等价 ``cv2.threshold(gray, 0, 255, THRESH_BINARY|THRESH_OTSU)`` 返回的阈值。

    与 OpenCV 的 ``getThreshVal_Otsu_8u`` 同为“类间方差最大化、严格大于才更新”，
    因此平局取第一个候选阈值，结果逐位一致。
    """
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    total = float(hist.sum())
    if total == 0:
        return 0
    sum_all = float(np.dot(np.arange(256, dtype=np.float64), hist))
    sum_b = 0.0
    weight_b = 0.0
    max_sigma = 0.0
    threshold = 0
    for t in range(256):
        weight_b += hist[t]
        if weight_b == 0.0:
            continue
        weight_f = total - weight_b
        if weight_f == 0.0:
            break
        sum_b += t * hist[t]
        mean_b = sum_b / weight_b
        mean_f = (sum_all - sum_b) / weight_f
        sigma = weight_b * weight_f * (mean_b - mean_f) ** 2
        if sigma > max_sigma:
            max_sigma = sigma
            threshold = t
    return threshold


def _connected_components_8(mask: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    """8 连通域统计，等价格 ``cv2.connectedComponentsWithStats(mask, connectivity=8)``。

    返回 ``[(left, top, width, height, area), ...]``（不含背景）。
    算法：逐行求前景行程（run），用并查集把与上一行 8 连通重叠的行程合并，
    再把同一连通域的包围盒/面积累加起来。行程数远小于像素数，448² 图约 8 ms。
    """
    height, width = mask.shape
    parent: list[int] = []
    stats: list[list[int]] = []  # 每个行程: [left, top, right, bottom, area]

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    prev_runs: list[tuple[int, int, int]] = []  # (start, end_exclusive, node)
    for row in range(height):
        padded = np.zeros(width + 2, dtype=np.int8)
        padded[1:-1] = mask[row]
        diff = np.diff(padded)
        starts = np.flatnonzero(diff == 1).tolist()
        ends = np.flatnonzero(diff == -1).tolist()

        cur_runs: list[tuple[int, int, int]] = []
        cursor = 0
        for start, end in zip(starts, ends):
            node = len(parent)
            parent.append(node)
            stats.append([start, row, end - 1, row, end - start])
            # 8 连通：上一行行程 [ps, pe) 与当前 [start, end) 相邻当且仅当
            # ps <= end 且 start <= pe（像素距离 <= 1）。
            while cursor < len(prev_runs) and prev_runs[cursor][1] < start:
                cursor += 1
            k = cursor
            while k < len(prev_runs) and prev_runs[k][0] <= end:
                if prev_runs[k][1] >= start:
                    union(node, prev_runs[k][2])
                k += 1
            cur_runs.append((start, end, node))
        prev_runs = cur_runs

    merged: dict[int, list[int]] = {}
    for node in range(len(parent)):
        root = find(node)
        st = stats[node]
        box = merged.get(root)
        if box is None:
            merged[root] = list(st)
        else:
            box[0] = min(box[0], st[0])
            box[1] = min(box[1], st[1])
            box[2] = max(box[2], st[2])
            box[3] = max(box[3], st[3])
            box[4] += st[4]
    return [(left, top, right - left + 1, bottom - top + 1, area) for (left, top, right, bottom, area) in merged.values()]


def adaptive_upsize_resolution(image: "np.ndarray | Image.Image", scale_resolution: int = DEFAULT_SCALE_RESOLUTION) -> int | None:
    """官方 ``adaptive_upsize_resolution`` 的纯 numpy 复刻。

    参数
    ----
    image: ``uint8`` 图像，支持 ``(C, H, W)``（官方形式）、``(H, W, C)``（PIL 形式）
        或 ``(H, W)``（灰度）。也可直接传 PIL Image。
    scale_resolution: 放大目标上限。

    返回
    ----
    ``scale_resolution``（小字密排需要放大）、``TINY_RESOLUTION``（极小图）或 ``None``（保持原样）。
    """
    if isinstance(image, Image.Image):
        arr = _pil_to_chw(image)
    else:
        arr = np.asarray(image)
        if arr.ndim == 3 and arr.shape[0] not in (1, 3, 4):
            # 官方函数的 numpy 形式是 (C, H, W)；HWC 请先传 PIL Image 或自行 transpose。
            raise ValueError(
                f"numpy 输入须为 (C, H, W)（CHW），收到 shape={arr.shape}；"
                "HWC 图像请传 PIL Image（内部会转 RGB/CHW）或自行 transpose(2,0,1)。"
            )
    arr = np.ascontiguousarray(arr)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)

    height, width = arr.shape[-2:]
    if height <= 0 or width <= 0:
        return None

    if arr.ndim == 3 and arr.shape[0] >= 3:
        gray = _luma_itu601(arr)
    else:
        gray = arr.reshape(height, width)

    threshold = _otsu_threshold(gray)
    foreground = gray <= threshold  # cv2 THRESH_BINARY 的补集：暗色字形为前景
    if foreground.mean() > 0.5:
        # 官方：假设字形覆盖不到半张图，否则认为浅色字/深色底，取反
        foreground = ~foreground

    glyph_heights: list[int] = []
    for _left, _top, comp_width, comp_height, area in _connected_components_8(foreground):
        if area < 3 or comp_height < 2:
            continue
        if comp_width >= 0.9 * width or comp_height >= 0.9 * height:
            continue
        glyph_heights.append(comp_height)

    font_px = float(np.median(glyph_heights)) if glyph_heights else 0.0
    density = len(glyph_heights) * 1000.0 / (width * height)
    if 0 < font_px <= SMALL_FONT_PX and density >= DENSE_GLYPHS_PER_KPX:
        return scale_resolution
    if height * width < TINY_PIXELS:
        return TINY_RESOLUTION
    return None


# ---------------------------------------------------------------------------
# 内部辅助
# ---------------------------------------------------------------------------
def _pil_to_chw(image: Image.Image) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(image.convert("RGB")).transpose(2, 0, 1))


def _ensure_rgb(image: Image.Image) -> Image.Image:
    """官方 ``do_convert_rgb=True``：统一转 RGB。"""
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def _resize(image: Image.Image, height: int, width: int, resample: int = _RESAMPLE) -> Image.Image:
    """PIL bicubic resize；尺寸相同时直接返回（PIL 同尺寸 bicubic 为恒等变换）。"""
    if image.size == (width, height):
        return image
    return image.resize((width, height), resample)


# ---------------------------------------------------------------------------
# 对外数据结构
# ---------------------------------------------------------------------------
@dataclass
class Slice:
    """一个切片（``is_global=True`` 为全局缩略图，其余为按阅读顺序切出的局部块）。"""

    index: int
    image: Image.Image
    height: int
    width: int
    patch_size: int
    token_divisor: int
    is_global: bool = False

    @property
    def size(self) -> tuple[int, int]:
        """``(width, height)``，与 PIL 的 ``Image.size`` 一致。"""
        return (self.width, self.height)

    @property
    def grid_h(self) -> int:
        """patch 网格行数（= height // patch_size）。"""
        return self.height // self.patch_size

    @property
    def grid_w(self) -> int:
        """patch 网格列数（= width // patch_size）。"""
        return self.width // self.patch_size

    @property
    def num_patches(self) -> int:
        return self.grid_h * self.grid_w

    @property
    def num_tokens(self) -> int:
        """该切片在官方 processor 下占用的 image token 数 = patch 数 / divisor。"""
        return self.num_patches // self.token_divisor

    def summary(self) -> dict:
        return {
            "index": self.index,
            "is_global": self.is_global,
            "height": self.height,
            "width": self.width,
            "grid_h": self.grid_h,
            "grid_w": self.grid_w,
            "num_patches": self.num_patches,
            "num_tokens": self.num_tokens,
        }

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        kind = "global" if self.is_global else "slice"
        return (
            f"Slice(index={self.index}, {kind}, {self.width}x{self.height}, "
            f"grid={self.grid_h}x{self.grid_w}, tokens={self.num_tokens})"
        )


@dataclass
class SlicePlan:
    """一张图的完整切片方案（对应官方 processor 的一张图）。"""

    slices: list[Slice]
    grid: list[int] | None
    source_resolution: int
    allow_upscale: bool
    source_height: int
    source_width: int
    scale_resolution: int = DEFAULT_SCALE_RESOLUTION
    max_slice_nums: int = DEFAULT_MAX_SLICE_NUMS
    patch_size: int = DEFAULT_PATCH_SIZE
    downsample_mode: str = DEFAULT_DOWNSAMPLE_MODE
    small_image_upsize: str = DEFAULT_SMALL_IMAGE_UPSIZE

    @property
    def is_sliced(self) -> bool:
        return self.grid is not None

    @property
    def num_slices(self) -> int:
        """切片总数（含全局图），无切片时为 1。"""
        return len(self.slices)

    @property
    def num_local_slices(self) -> int:
        """局部切片数（不含全局图）。"""
        return sum(1 for s in self.slices if not s.is_global)

    @property
    def total_tokens(self) -> int:
        return sum(s.num_tokens for s in self.slices)

    @property
    def total_patches(self) -> int:
        return sum(s.num_patches for s in self.slices)

    def summary(self) -> dict:
        """可直接 JSON 序列化的摘要（板端 /api/slices 之类的接口用）。"""
        return {
            "source_size": [self.source_width, self.source_height],
            "grid": self.grid,
            "scale_resolution": self.scale_resolution,
            "max_slice_nums": self.max_slice_nums,
            "patch_size": self.patch_size,
            "downsample_mode": self.downsample_mode,
            "small_image_upsize": self.small_image_upsize,
            "source_resolution": self.source_resolution,
            "allow_upscale": self.allow_upscale,
            "num_slices": self.num_slices,
            "num_local_slices": self.num_local_slices,
            "total_patches": self.total_patches,
            "total_tokens": self.total_tokens,
            "slices": [s.summary() for s in self.slices],
        }


# ---------------------------------------------------------------------------
# 主流程：复刻官方 _preprocess 的 slice 分支
# ---------------------------------------------------------------------------
def _resolve_tile(
    image: Image.Image,
    grid: Sequence[int],
    row: int,
    column: int,
    base_tile: Image.Image,
    scale_resolution: int,
    patch_size: int,
    downsample_mode: str,
) -> Image.Image:
    """官方 ``_resolve_tile_size``：对某个切片单元格挑自适应分辨率。

    只有当“自适应后的 token 数严格少于基准切片”时才换；否则沿用基准切片。
    """
    width, height = image.size
    grid_rows, grid_columns = grid
    left = round(width * column / grid_columns)
    right = round(width * (column + 1) / grid_columns)
    top = round(height * row / grid_rows)
    bottom = round(height * (row + 1) / grid_rows)
    cell = image.crop((left, top, right, bottom))

    cell_width, cell_height = cell.size
    if cell_height < 1 or cell_width < 1 or cell_height * cell_width > scale_resolution**2:
        return base_tile

    resolution = adaptive_upsize_resolution(_pil_to_chw(cell), scale_resolution)
    if resolution == scale_resolution:
        return base_tile
    target_height, target_width = find_best_resize(
        (cell_height, cell_width),
        resolution if resolution is not None else scale_resolution,
        patch_size,
        downsample_mode,
        allow_upscale=resolution is not None,
    )
    if target_height * target_width >= base_tile.size[1] * base_tile.size[0]:
        return base_tile
    if (target_height, target_width) == (cell_height, cell_width):
        return cell
    return _resize(cell, target_height, target_width, _RESAMPLE)


def slice_image(
    image: Image.Image,
    *,
    scale_resolution: int = DEFAULT_SCALE_RESOLUTION,
    max_slice_nums: int = DEFAULT_MAX_SLICE_NUMS,
    patch_size: int = DEFAULT_PATCH_SIZE,
    slice_mode: bool = True,
    downsample_mode: str = DEFAULT_DOWNSAMPLE_MODE,
    small_image_upsize: str = DEFAULT_SMALL_IMAGE_UPSIZE,
    adaptive_fn: Callable[[np.ndarray, int], int | None] | None = None,
) -> SlicePlan:
    """按官方 ``MiniCPMOcrImageProcessor._preprocess`` 的逻辑切图。

    参数
    ----
    image: PIL Image（任意模式，内部转 RGB）。
    scale_resolution / max_slice_nums / patch_size / downsample_mode: 与官方同名配置项。
    slice_mode: False 时强制不切（只有全局图）。
    small_image_upsize: ``"always"``（官方默认：小图放大到 scale_resolution）、
        ``"never"``（保持原分辨率，只做 patch 对齐）、``"adaptive"``（按字形大小/密度自适应，
        需要 ``adaptive_upsize_resolution``，即上面的 numpy 实现）。
    adaptive_fn: 可注入替换 ``adaptive_upsize_resolution`` 的实现（对比测试用）。

    返回
    ----
    :class:`SlicePlan`：``slices[0]`` 恒为全局图，其后为按阅读顺序（行优先）的局部切片。
    每片带 ``(height, width)``、patch 网格 ``(grid_h, grid_w)`` 与 image token 数。
    """
    if small_image_upsize not in VALID_SMALL_IMAGE_UPSIZE:
        raise ValueError(
            f"small_image_upsize must be one of {VALID_SMALL_IMAGE_UPSIZE}, got {small_image_upsize!r}"
        )
    if adaptive_fn is None:
        adaptive_fn = adaptive_upsize_resolution

    image = _ensure_rgb(image)
    source_width, source_height = image.size
    image_size = (source_height, source_width)

    best_grid = (
        get_sliced_grid(image_size, max_slice_nums, scale_resolution) if slice_mode else None
    )

    # 未切图的小图可以保留或抬高分辨率；切图时全局缩略图固定用 scale_resolution
    source_resolution = scale_resolution
    allow_upscale = best_grid is None and small_image_upsize == "always"
    if best_grid is None and small_image_upsize == "adaptive":
        resolution = adaptive_fn(_pil_to_chw(image), scale_resolution)
        source_resolution = resolution if resolution is not None else scale_resolution
        allow_upscale = resolution is not None

    source_h, source_w = find_best_resize(
        image_size,
        source_resolution,
        patch_size,
        downsample_mode,
        allow_upscale=allow_upscale,
    )
    source_img = _resize(image, source_h, source_w)
    divisor = token_divisor(downsample_mode)
    slices: list[Slice] = [
        Slice(
            index=0,
            image=source_img,
            height=source_h,
            width=source_w,
            patch_size=patch_size,
            token_divisor=divisor,
            is_global=True,
        )
    ]

    if best_grid is not None:
        refine_h, refine_w = get_refine_size(
            image_size,
            best_grid,
            scale_resolution,
            patch_size,
            downsample_mode,
            allow_upscale=True,
        )
        refine_img = _resize(image, refine_h, refine_w)
        grid_y, grid_x = best_grid
        patch_height = refine_img.size[1] // grid_y
        patch_width = refine_img.size[0] // grid_x
        tiles = divide_to_patches(refine_img, patch_height, patch_width)
        if small_image_upsize == "adaptive":
            tiles = [
                _resolve_tile(
                    image,
                    best_grid,
                    i // grid_x,
                    i % grid_x,
                    tile,
                    scale_resolution,
                    patch_size,
                    downsample_mode,
                )
                for i, tile in enumerate(tiles)
            ]
        for i, tile in enumerate(tiles):
            tile_width, tile_height = tile.size
            slices.append(
                Slice(
                    index=i + 1,
                    image=tile,
                    height=tile_height,
                    width=tile_width,
                    patch_size=patch_size,
                    token_divisor=divisor,
                    is_global=False,
                )
            )

    return SlicePlan(
        slices=slices,
        grid=best_grid,
        source_resolution=source_resolution,
        allow_upscale=allow_upscale,
        source_height=source_height,
        source_width=source_width,
        scale_resolution=scale_resolution,
        max_slice_nums=max_slice_nums,
        patch_size=patch_size,
        downsample_mode=downsample_mode,
        small_image_upsize=small_image_upsize,
    )


def preprocess(
    image: Image.Image,
    *,
    scale_resolution: int = DEFAULT_SCALE_RESOLUTION,
    max_slice_nums: int = DEFAULT_MAX_SLICE_NUMS,
    patch_size: int = DEFAULT_PATCH_SIZE,
    slice_mode: bool = True,
    downsample_mode: str = DEFAULT_DOWNSAMPLE_MODE,
    small_image_upsize: str = DEFAULT_SMALL_IMAGE_UPSIZE,
    image_mean: Sequence[float] = (0.5, 0.5, 0.5),
    image_std: Sequence[float] = (0.5, 0.5, 0.5),
    rescale_factor: float = 1.0 / 255.0,
    adaptive_fn: Callable[[np.ndarray, int], int | None] | None = None,
) -> dict:
    """复刻官方 ``_preprocess`` 的完整输出（除 processor 层的占位符拼接）。

    返回 dict（numpy，无 torch）：

    - ``pixel_values``: ``float32 [1, 3, patch_size, sum(gh*gw*patch_size)]``，
      与官方 processor 同形同义的 patchified 张量。
    - ``target_sizes``: ``int32 [N, 2]``，每片 ``[grid_h, grid_w]``。
    - ``grids``: ``[[grid_y, grid_x]]``（不切图时 ``[[0, 0]]``）。
    - ``num_patches_per_image``: ``[N]``。
    - ``plan``: 对应 :class:`SlicePlan`（含 PIL 切片图，便于调试/版面用）。
    """
    plan = slice_image(
        image,
        scale_resolution=scale_resolution,
        max_slice_nums=max_slice_nums,
        patch_size=patch_size,
        slice_mode=slice_mode,
        downsample_mode=downsample_mode,
        small_image_upsize=small_image_upsize,
        adaptive_fn=adaptive_fn,
    )
    mean = np.asarray(image_mean, dtype=np.float32).reshape(3, 1, 1)
    std = np.asarray(image_std, dtype=np.float32).reshape(3, 1, 1)

    patched: list[np.ndarray] = []
    target_sizes: list[list[int]] = []
    for s in plan.slices:
        # HWC uint8 → CHW float32 → rescale → normalize（顺序与官方一致）
        chw = np.asarray(s.image, dtype=np.float32).transpose(2, 0, 1)
        if rescale_factor is not None:
            chw = chw * float(rescale_factor)
        if mean is not None:
            chw = (chw - mean) / std
        patched.append(reshape_by_patch(chw, patch_size))
        target_sizes.append([s.grid_h, s.grid_w])

    pixel_values = np.concatenate(patched, axis=-1)[None, ...]
    return {
        "pixel_values": pixel_values,
        "target_sizes": np.asarray(target_sizes, dtype=np.int32),
        "grids": [plan.grid if plan.grid is not None else [0, 0]],
        "num_patches_per_image": [len(plan.slices)],
        "plan": plan,
    }


# ---------------------------------------------------------------------------
# CLI：打印一张图的切片方案（板端排查用）
# ---------------------------------------------------------------------------
def _cli() -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(description="MiniCPM-OCR 切片方案预演（不推理）")
    parser.add_argument("image", help="输入图片路径")
    parser.add_argument("--scale-resolution", type=int, default=DEFAULT_SCALE_RESOLUTION)
    parser.add_argument("--max-slice-nums", type=int, default=DEFAULT_MAX_SLICE_NUMS)
    parser.add_argument("--patch-size", type=int, default=DEFAULT_PATCH_SIZE)
    parser.add_argument("--downsample-mode", default=DEFAULT_DOWNSAMPLE_MODE, choices=("4x", "16x"))
    parser.add_argument("--small-image-upsize", default=DEFAULT_SMALL_IMAGE_UPSIZE, choices=VALID_SMALL_IMAGE_UPSIZE)
    parser.add_argument("--no-slice", action="store_true", help="等价 slice_mode=False")
    args = parser.parse_args()

    plan = slice_image(
        Image.open(args.image),
        scale_resolution=args.scale_resolution,
        max_slice_nums=args.max_slice_nums,
        patch_size=args.patch_size,
        slice_mode=not args.no_slice,
        downsample_mode=args.downsample_mode,
        small_image_upsize=args.small_image_upsize,
    )
    print(f"源图 {plan.source_width}x{plan.source_height}  grid={plan.grid}  "
          f"tiles={plan.num_slices}  patches={plan.total_patches}  tokens={plan.total_tokens}")
    print(f"{'#':>2} {'kind':<7} {'size':>11} {'patch grid':>10} {'tokens':>7}")
    for s in plan.slices:
        print(f"{s.index:>2} {'global' if s.is_global else 'slice':<7} "
              f"{s.width:>5}x{s.height:<5} {s.grid_h:>4}x{s.grid_w:<4} {s.num_tokens:>7}")


if __name__ == "__main__":  # pragma: no cover
    _cli()
