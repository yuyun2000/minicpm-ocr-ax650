#!/usr/bin/env python3
"""OTSL → Markdown 表格转换（纯标准库，板端 Python 3.10 + 无第三方依赖）。

背景
----
MiniCPM-OCR 的表格输出是 **OTSL**（OmniDocBench Table Structure Language），形如::

    <otsl><fcel>Name<fcel>Age<fcel>City<nl><fcel>Tom<fcel>18<fcel>SZ<nl></otsl>

官方 SDK 只提供 OTSL→HTML（``MiniCPM-OCR/sdk/minicpmvocr/postprocess/otsl``），
板端网页测试台需要直接渲染 Markdown，因此这里实现一个自包含的 OTSL→Markdown 转换器，
并把官方 SDK 的合并单元格语义（``lcel``/``ucel``/``xcel`` 是**续格**而不是独立单元格）
与“松散行宽修复”算法一并复刻。

OTSL 标签语义（本实现覆盖）
--------------------------
==========================  ============================================================
标签                         含义
==========================  ============================================================
``<otsl>`` / ``</otsl>``    表格体开始/结束（缺失时按整段文本处理）
``<nl>``                    行分隔符（新起一行）
``<fcel>``                  普通单元格（full cell），后面紧跟该格文本
``<ecel>``                  空单元格（empty cell），无文本
``<lcel>``                  左续格：与**左侧**锚点单元格合并（被 colspan 覆盖）
``<ucel>``                  上续格：与**上方**锚点单元格合并（被 rowspan 覆盖）
``<xcel>``                  交叉续格：2×2 合并的右下角（同时被 rowspan 与 colspan 覆盖）
``<ched>``                  列头单元格（column header），带文本，按普通单元格处理
``<rhed>``                  行头单元格（row header），带文本，按普通单元格处理
``<scel>``                  spanning cell（OTSL v1.0 扩展），按普通单元格处理
==========================  ============================================================

Markdown 输出规则（Markdown 语法本身不支持合并单元格）
------------------------------------------------------
1. ``<nl>`` 分行；行宽不一致时按官方 ``otsl_pad_to_sqr_v2`` 的代价最小化算法修成等宽
   （对超长行**截断**、对短行补 ``<ecel>``），列数 = 修出来的宽度。
2. 锚点单元格（``fcel``/``ecel``/``ched``/``rhed``/``scel``）决定合并范围：

   - ``colspan`` = 1 + 该格右侧连续的 ``lcel``/``xcel`` 数量；
   - ``rowspan`` = 1 + 该格下方同列连续的 ``ucel``/``xcel`` 数量。

3. Markdown 渲染：合并区域的文本放在**左上角格**，其余被覆盖的位置输出空串；
   这样表格保持矩形、不丢列，转到 HTML 视图时可用 ``to_html`` 输出真正的
   ``rowspan``/``colspan``。
4. 单元格转义：``|`` → ``\\|``（否则会截断 Markdown 表格）；换行 → ``<br>``；
   其它内容（含 ``<br>``/``<sup>`` 等内联标签、LaTeX）原样保留。
5. 表头：默认首行作 Markdown 表头（``header=True``）；``header=False`` 时输出一行空表头，
   所有行都作为表体（Markdown 必须有表头行）。
6. 输入若被 HTML 转义过（模型/API 偶尔输出 ``&lt;fcel&gt;``），会先 ``html.unescape``。

真实模型输出样例（单元测试使用）
--------------------------------
- ``notes/minicpm-ocr/perf-image.json``：AX650 板端两次 Table Recognition 的原始输出
  （6 行 × 8 列、全部 ``<fcel>``），本模块内嵌了一份拷贝，文件存在时以文件为准；
- ``demo/minicpm-ocr/baseline-result.md``：Mac CPU 基线输出的真实表格
  （6 行 × 4 列，含 ``<lcel>`` 列合并与 ``<ucel>`` 行合并）。

用法
----
>>> from otsl_to_md import otsl_to_markdown
>>> print(otsl_to_markdown("<otsl><fcel>A<fcel>B<nl><fcel>1<ecel><nl></otsl>"))
| A | B |
| --- | --- |
| 1 |  |

CLI::

    python otsl_to_md.py --test                    # 跑单元测试（真实模型输出样例）
    python otsl_to_md.py --file out.txt            # 从文件读（自动截取 <otsl>…</otsl>）
    echo '<otsl><fcel>A<nl></otsl>' | python otsl_to_md.py
    python otsl_to_md.py --otsl '<otsl>...' --format html
"""

from __future__ import annotations

import html
import pathlib
import re
import sys
from dataclasses import dataclass, field

__all__ = [
    "OTSL_FCEL",
    "OtslCell",
    "OtslTable",
    "extract_otsl",
    "parse_otsl",
    "to_markdown",
    "to_html",
    "otsl_to_markdown",
    "otsl_to_html",
]

# ---------------------------------------------------------------------------
# 标签常量
# ---------------------------------------------------------------------------
OTSL_FCEL = "<fcel>"
OTSL_ECEL = "<ecel>"
OTSL_LCEL = "<lcel>"
OTSL_UCEL = "<ucel>"
OTSL_XCEL = "<xcel>"
OTSL_NL = "<nl>"
OTSL_CHED = "<ched>"
OTSL_RHED = "<rhed>"
OTSL_SCEL = "<scel>"
OTSL_OPEN = "<otsl>"
OTSL_CLOSE = "</otsl>"

#: 会开启一个新单元格的标签
CELL_TAGS = (OTSL_FCEL, OTSL_ECEL, OTSL_LCEL, OTSL_UCEL, OTSL_XCEL, OTSL_CHED, OTSL_RHED, OTSL_SCEL)
#: 作为“锚点”的单元格标签（决定合并范围的左上角格）
ANCHOR_TAGS = (OTSL_FCEL, OTSL_ECEL, OTSL_CHED, OTSL_RHED, OTSL_SCEL)
#: 延续格（本身不携带文本）
CONTINUATION_TAGS = (OTSL_LCEL, OTSL_UCEL, OTSL_XCEL)
#: 可携带文本的标签
TEXT_TAGS = (OTSL_FCEL, OTSL_CHED, OTSL_RHED, OTSL_SCEL)

_TAG_RE = re.compile(r"</otsl>|<otsl>|<nl>|<fcel>|<ecel>|<lcel>|<ucel>|<xcel>|<ched>|<rhed>|<scel>")


# ---------------------------------------------------------------------------
# 解析
# ---------------------------------------------------------------------------
@dataclass
class OtslCell:
    """解析出来的一个**锚点**单元格（含合并跨度）。"""

    row: int
    col: int
    text: str
    row_span: int = 1
    col_span: int = 1
    tag: str = OTSL_FCEL

    @property
    def is_header(self) -> bool:
        """``<ched>`` / ``<rhed>`` 标记的单元格。"""
        return self.tag in (OTSL_CHED, OTSL_RHED)

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return (
            f"OtslCell(row={self.row}, col={self.col}, text={self.text!r}, "
            f"row_span={self.row_span}, col_span={self.col_span}, tag={self.tag})"
        )


@dataclass
class OtslTable:
    """OTSL 解析结果：显式网格 + 所有锚点单元格。"""

    num_rows: int
    num_cols: int
    cells: list[OtslCell] = field(default_factory=list)
    _grid: list[list[str]] = field(default_factory=list, repr=False)

    @property
    def grid(self) -> list[list[str]]:
        """二维文本网格：合并单元格的文本在左上角，被覆盖的位置是空串。"""
        return self._grid

    def to_markdown(self, header: bool = True) -> str:
        return to_markdown(self, header=header)

    def to_html(self, header: bool = True) -> str:
        return to_html(self, header=header)


def extract_otsl(text: str) -> str:
    """从任意文本中截出 OTSL 片段。

    - 有 ``<otsl>`` 时取到 ``</otsl>``（缺失则取到结尾）；
    - 没有 ``<otsl>`` 但含 OTSL 单元格标签时，把整段当作表体；
    - 都没有则抛 ``ValueError``。
    """
    if not isinstance(text, str):
        raise TypeError(f"OTSL 必须是 str，收到 {type(text).__name__}")
    text = html.unescape(text)
    start = text.find(OTSL_OPEN)
    if start >= 0:
        end = text.find(OTSL_CLOSE, start)
        body = text[start:] if end < 0 else text[start:end + len(OTSL_CLOSE)]
        return body
    if _TAG_RE.search(text):
        # 模型偶尔省略 <otsl> 包裹
        return text
    raise ValueError("输入里找不到 OTSL 标签（<otsl>/<fcel>/<nl> …）")


def _tokenize(otsl: str) -> list[list[dict]]:
    """把 OTSL 字符串切成 ``rows -> [{'tag':..., 'text':...}, ...]``。"""
    rows: list[list[dict]] = []
    row: list[dict] = []
    cells: list[dict] = []
    pending: dict | None = None
    pos = 0
    for match in _TAG_RE.finditer(otsl):
        chunk = otsl[pos:match.start()]
        if pending is not None and chunk.strip():
            # 文本归属当前单元格（可能跨多段，直接拼接）
            pending["text"] += chunk
        pos = match.end()
        tag = match.group(0)
        if tag == OTSL_OPEN:
            rows, row, cells, pending = [], [], [], None
        elif tag == OTSL_CLOSE:
            break
        elif tag == OTSL_NL:
            rows.append(row)
            row = []
            pending = None
        else:
            cell = {"tag": tag, "text": ""}
            cells.append(cell)
            row.append(cell)
            pending = cell
    # 结尾（可能没有 </otsl>）
    tail = otsl[pos:]
    if pending is not None and tail.strip():
        pending["text"] += tail
    if row:
        rows.append(row)
    # 丢弃空行（连续 <nl> 或纯尾随 <nl>），与官方 pad 逻辑一致
    rows = [r for r in rows if r]
    return rows


def _pad_rows(rows: list[list[dict]]) -> tuple[list[list[dict]], int]:
    """复刻官方 ``otsl_pad_to_sqr_v2``：把不等宽的行修成等宽。

    在 ``[最后 fcel 之后的列数, 最大行宽]`` 里选一个宽度，使所有行“截断/补齐”的
    总代价 ``Σ|行宽 - 宽度|`` 最小（平局取更小的宽度，与官方 ``<`` 严格比较一致）。
    """
    if not rows:
        return [], 0
    min_widths = []
    for row in rows:
        last = 0
        for i, cell in enumerate(row):
            if cell["tag"] == OTSL_FCEL:
                last = i + 1
        min_widths.append(last)
    global_min_width = max(min_widths)
    max_total_len = max(len(row) for row in rows)
    search_start = global_min_width
    search_end = max(global_min_width, max_total_len)
    best_width = search_end
    best_cost = float("inf")
    for width in range(search_start, search_end + 1):
        cost = sum(abs(len(row) - width) for row in rows)
        if cost < best_cost:
            best_cost = cost
            best_width = width
    if best_width <= 0:
        return [], 0
    padded: list[list[dict]] = []
    for row in rows:
        if len(row) > best_width:
            new_row = row[:best_width]
        else:
            new_row = row + [{"tag": OTSL_ECEL, "text": ""} for _ in range(best_width - len(row))]
        padded.append(new_row)
    return padded, best_width


def parse_otsl(text: str) -> OtslTable:
    """解析 OTSL 文本，返回带合并跨度的 :class:`OtslTable`。

    合并语义与官方 SDK 一致：``lcel`` 把左侧锚点的 ``col_span`` 加 1，
    ``ucel`` 把上方锚点的 ``row_span`` 加 1，``xcel`` 同时加给两者（2×2 合并的右下角）。
    """
    otsl = extract_otsl(text)
    rows = _tokenize(otsl)
    rows, width = _pad_rows(rows)
    if not rows:
        return OtslTable(num_rows=0, num_cols=0, cells=[], _grid=[])

    num_rows = len(rows)
    grid = [["" for _ in range(width)] for _ in range(num_rows)]
    occupied = [[False] * width for _ in range(num_rows)]
    cells: list[OtslCell] = []

    for r in range(num_rows):
        for c in range(width):
            if occupied[r][c]:
                continue
            cell = rows[r][c]
            tag = cell["tag"]
            occupied[r][c] = True
            if tag in CONTINUATION_TAGS:
                # 没有锚点的裸续格（模型输出不规范）：当空单元格处理
                continue
            col_span = 1
            while c + col_span < width and rows[r][c + col_span]["tag"] in (OTSL_LCEL, OTSL_XCEL):
                col_span += 1
            row_span = 1
            while r + row_span < num_rows and rows[r + row_span][c]["tag"] in (OTSL_UCEL, OTSL_XCEL):
                row_span += 1
            text = "" if tag == OTSL_ECEL else cell["text"].strip()
            for rr in range(r, min(num_rows, r + row_span)):
                for cc in range(c, min(width, c + col_span)):
                    occupied[rr][cc] = True
            grid[r][c] = text
            cells.append(
                OtslCell(row=r, col=c, text=text, row_span=row_span, col_span=col_span, tag=tag)
            )

    return OtslTable(num_rows=num_rows, num_cols=width, cells=cells, _grid=grid)


# ---------------------------------------------------------------------------
# 渲染
# ---------------------------------------------------------------------------
def _escape_cell(text: str) -> str:
    """Markdown 单元格转义：换行 → ``<br>``，``|`` → ``\\|``。"""
    text = text.strip()
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\n", "<br>")
    text = text.replace("|", "\\|")
    return text


def _md_row(cells: list[str]) -> str:
    return "| " + " | ".join(_escape_cell(c) for c in cells) + " |"


def to_markdown(table: OtslTable, header: bool = True) -> str:
    """把 :class:`OtslTable` 渲染成 Markdown 表格。

    ``header=True``（默认）用第一行做表头；``header=False`` 输出空表头 + 全表体。
    """
    if table.num_rows == 0 or table.num_cols == 0:
        return ""
    separator = "| " + " | ".join(["---"] * table.num_cols) + " |"
    lines: list[str] = []
    if header:
        body = table.grid[1:]
        lines.append(_md_row(table.grid[0]))
        lines.append(separator)
        for row in body:
            lines.append(_md_row(row))
    else:
        lines.append(_md_row([""] * table.num_cols))
        lines.append(separator)
        for row in table.grid:
            lines.append(_md_row(row))
    return "\n".join(lines)


def to_html(table: OtslTable, header: bool = True) -> str:
    """把 :class:`OtslTable` 渲染成带 ``rowspan``/``colspan`` 的 HTML 表格。"""
    if table.num_rows == 0 or table.num_cols == 0:
        return ""
    covered = [[False] * table.num_cols for _ in range(table.num_rows)]
    anchor = {(cell.row, cell.col): cell for cell in table.cells}
    out = ["<table>"]
    for r in range(table.num_rows):
        out.append("  <tr>")
        for c in range(table.num_cols):
            if covered[r][c]:
                continue
            cell = anchor.get((r, c))
            if cell is None:  # 裸续格（不规范输出）
                out.append("    <td></td>")
                continue
            for rr in range(cell.row, min(table.num_rows, cell.row + cell.row_span)):
                for cc in range(cell.col, min(table.num_cols, cell.col + cell.col_span)):
                    covered[rr][cc] = True
            attrs = ""
            if cell.row_span > 1:
                attrs += f' rowspan="{cell.row_span}"'
            if cell.col_span > 1:
                attrs += f' colspan="{cell.col_span}"'
            is_head = header and r == 0
            node = "th" if is_head else "td"
            text = html.escape(cell.text, quote=False) if cell.text else ""
            out.append(f"    <{node}{attrs}>{text}</{node}>")
        out.append("  </tr>")
    out.append("</table>")
    return "\n".join(out)


def otsl_to_markdown(text: str, header: bool = True) -> str:
    """便捷函数：任意含 OTSL 的文本 → Markdown 表格。"""
    return to_markdown(parse_otsl(text), header=header)


def otsl_to_html(text: str, header: bool = True) -> str:
    """便捷函数：任意含 OTSL 的文本 → HTML 表格。"""
    return to_html(parse_otsl(text), header=header)


# ---------------------------------------------------------------------------
# 真实模型输出样例（从仓库文件读取；下方内嵌拷贝用于文件缺失时的自包含测试）
# ---------------------------------------------------------------------------
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_PERF_IMAGE_JSON = _REPO_ROOT / "notes/minicpm-ocr/perf-image.json"
_BASELINE_RESULT_MD = _REPO_ROOT / "demo/minicpm-ocr/baseline-result.md"

# 内嵌拷贝：perf-image.json runs[0].content（与 runs[1] 相同）
_REAL_PERF_OTSL = '<otsl><fcel>Name<fcel>Characteristics<fcel>Sample Size (n)<fcel>Sample Type<fcel>Sample Source<fcel>Data Collection Method<fcel>Data Analysis Method<fcel>Notes<nl><fcel>C. albicauda<fcel>Non-infectious<fcel>10<fcel>Mammal<fcel>Field catch<fcel>Visual identification<fcel>Statistical analysis<fcel>Only in situ dataset<nl><fcel>C. albicauda<fcel>Infectious<fcel>10<fcel>Mammal<fcel>Field catch<fcel>Visual identification<fcel>Statistical analysis<fcel>Only in situ dataset<nl><fcel>C. albicauda<fcel>Non-infectious<fcel>10<fcel>Mammal<fcel>Field catch<fcel>Visual identification<fcel>Statistical analysis<fcel>Only in situ dataset<nl><fcel>C. albicauda<fcel>Infectious<fcel>10<fcel>Mammal<fcel>Field catch<fcel>Visual identification<fcel>Statistical analysis<fcel>Only in situ dataset<nl><fcel>C. albicauda<fcel>Non-infectious<fcel>10<fcel>Mammal<fcel>Field catch<fcel>Visual identification<fcel>Statistical analysis<fcel>Only in situ dataset<nl><fcel>C. albicauda<fcel>Infectious<fcel>10<fcel>Mammal<fcel>Field catch<fcel>Visual identification<fcel>Statistical analysis<fcel>Only in situ dataset<nl></otsl>'

# 内嵌拷贝：baseline-result.md 里的截断输出（有 <lcel>、无 </otsl>）
_REAL_BASELINE_OTSL = '<otsl><fcel>Task<fcel>Benchmark<fcel>MiniCPM-V4.7<lcel><nl><fcel>Size<lcel><fcel>1.3B<lcel><nl><fcel>Visual Token Compression Ratio<lcel><fcel>4x<fcel>16x<nl><fcel>General<fcel>MMBench<fcel>83.17<fcel>81.89<nl><ucel><fcel>MMStar<fcel>69.53<fcel>68.80<nl><ucel><fcel>MMT-Bench<fcel>61.69<fcel>61.43<nl></otsl>'


def load_real_samples() -> list[tuple[str, str]]:
    """读取真实模型输出样例：[(名称, OTSL 文本), ...]。文件缺失时退回内嵌拷贝。"""
    samples: list[tuple[str, str]] = []
    if _PERF_IMAGE_JSON.exists():
        import json

        data = json.loads(_PERF_IMAGE_JSON.read_text(encoding="utf-8"))
        for i, run in enumerate(data.get("runs", [])):
            content = run.get("content", "")
            if "<otsl>" in content:
                samples.append((f"perf-image.json runs[{i}]（AX650 板端 Table Recognition）", content))
    if not samples:
        samples.append(("内嵌 perf-image 样例", _REAL_PERF_OTSL))

    if _BASELINE_RESULT_MD.exists():
        text = _BASELINE_RESULT_MD.read_text(encoding="utf-8")
        idx = text.find("<otsl>")
        if idx >= 0:
            line = text[idx:].splitlines()[0]
            if line.strip():
                samples.append(("baseline-result.md（Mac CPU 基线，截断输出）", line.strip()))
    else:
        samples.append(("内嵌 baseline 样例", _REAL_BASELINE_OTSL))
    return samples


# ---------------------------------------------------------------------------
# 单元测试
# ---------------------------------------------------------------------------
def _run_tests() -> int:
    import unittest

    class TestOtslToMarkdown(unittest.TestCase):
        # ---- 真实模型输出样例 ----
        def test_real_samples_parse(self) -> None:
            samples = load_real_samples()
            # 仓库内 perf-image.json 有 2 次真实板端输出，baseline-result.md 有 1 段真实表格
            expected = 3 if (_PERF_IMAGE_JSON.exists() and _BASELINE_RESULT_MD.exists()) else 2
            self.assertGreaterEqual(len(samples), expected, "真实模型样例数量不足")
            for name, raw in samples:
                with self.subTest(sample=name):
                    table = parse_otsl(raw)
                    self.assertGreater(table.num_rows, 0, "解析后至少 1 行")
                    self.assertGreater(table.num_cols, 0, "解析后至少 1 列")
                    md = otsl_to_markdown(raw)
                    self.assertTrue(md.startswith("| "), "Markdown 应以 | 开头")
                    for tag in ("<otsl>", "</otsl>", "<fcel>", "<nl>", "<lcel>", "<ucel>", "<xcel>"):
                        self.assertNotIn(tag, md, f"输出里不应残留 {tag}")
                    # 行列数与 Markdown 行数一致（表头 + 分隔线 + 表体）
                    self.assertEqual(len(md.splitlines()), table.num_rows + 1)

        def test_real_perf_sample_shape(self) -> None:
            table = parse_otsl(_REAL_PERF_OTSL)
            self.assertEqual((table.num_rows, table.num_cols), (7, 8))  # 表头 + 6 行数据
            self.assertEqual(table.grid[0][0], "Name")
            self.assertEqual(table.grid[0][7], "Notes")
            self.assertEqual(table.grid[1][0], "C. albicauda")
            self.assertEqual(table.grid[6][4], "Field catch")
            md = otsl_to_markdown(_REAL_PERF_OTSL)
            self.assertIn("| Name | Characteristics |", md)
            self.assertIn("| C. albicauda |", md)
            self.assertEqual(len(md.splitlines()), 8)

        def test_real_baseline_sample_merges(self) -> None:
            """真实 CPU 基线输出：6 行 × 4 列，含 lcel 列合并与 ucel 行合并。"""
            table = parse_otsl(_REAL_BASELINE_OTSL)
            self.assertEqual((table.num_rows, table.num_cols), (6, 4))
            anchors = {(c.row, c.col): c for c in table.cells}
            # 列合并：MiniCPM-V4.7 在 (0,2) 横跨两列；Size / 1.3B 各自横跨两列
            self.assertEqual(anchors[(0, 2)].text, "MiniCPM-V4.7")
            self.assertEqual(anchors[(0, 2)].col_span, 2)
            self.assertEqual(anchors[(1, 0)].text, "Size")
            self.assertEqual(anchors[(1, 0)].col_span, 2)
            self.assertEqual(anchors[(1, 2)].text, "1.3B")
            self.assertEqual(anchors[(1, 2)].col_span, 2)
            # 行合并：General 纵跨 3 行
            self.assertEqual(anchors[(3, 0)].text, "General")
            self.assertEqual(anchors[(3, 0)].row_span, 3)
            # 被覆盖位置为空，右下角文本保留
            self.assertEqual(table.grid[4][0], "")
            self.assertEqual(table.grid[5][3], "61.43")
            md = otsl_to_markdown(_REAL_BASELINE_OTSL)
            self.assertEqual(len(md.splitlines()), 7)
            self.assertIn("| Task | Benchmark | MiniCPM-V4.7 |  |", md)

        # ---- 合并单元格语义 ----
        def test_colspan_lcel(self) -> None:
            table = parse_otsl("<otsl><fcel>A<lcel><lcel><fcel>B<nl><fcel>1<fcel>2<fcel>3<fcel>4<nl></otsl>")
            anchors = {(c.row, c.col): c for c in table.cells}
            self.assertEqual(anchors[(0, 0)].col_span, 3)
            self.assertEqual(anchors[(0, 0)].text, "A")
            self.assertEqual(anchors[(0, 3)].col_span, 1)
            self.assertEqual(table.grid[0], ["A", "", "", "B"])
            self.assertEqual(table.grid[1], ["1", "2", "3", "4"])

        def test_rowspan_ucel(self) -> None:
            table = parse_otsl("<otsl><fcel>A<fcel>B<nl><ucel><fcel>C<nl><ucel><fcel>D<nl></otsl>")
            anchors = {(c.row, c.col): c for c in table.cells}
            self.assertEqual(anchors[(0, 0)].row_span, 3)
            self.assertEqual(anchors[(0, 1)].text, "B")
            self.assertEqual(table.grid[0], ["A", "B"])
            self.assertEqual(table.grid[1], ["", "C"])
            self.assertEqual(table.grid[2], ["", "D"])

        def test_xcel_2x2(self) -> None:
            table = parse_otsl(
                "<otsl><fcel>A<fcel>B<fcel>C<fcel>E<nl>"
                "<fcel>D<fcel>X<lcel><fcel>F<nl>"
                "<fcel>G<ucel><xcel><fcel>I<nl></otsl>"
            )
            anchors = {(c.row, c.col): c for c in table.cells}
            merged = anchors[(1, 1)]
            self.assertEqual((merged.row_span, merged.col_span), (2, 2))
            self.assertEqual(table.grid[1], ["D", "X", "", "F"])
            self.assertEqual(table.grid[2], ["G", "", "", "I"])

        def test_empty_cell_ecel(self) -> None:
            table = parse_otsl("<otsl><fcel>A<ecel><nl><ecel><fcel>B<nl></otsl>")
            self.assertEqual(table.grid, [["A", ""], ["", "B"]])
            md = otsl_to_markdown("<otsl><fcel>A<ecel><nl><ecel><fcel>B<nl></otsl>")
            self.assertEqual(md, "| A |  |\n| --- | --- |\n|  | B |")

        def test_ragged_rows_repaired(self) -> None:
            # 行宽 3 / 2：官方 pad 算法补成 3 列
            table = parse_otsl("<otsl><fcel>A<fcel>B<fcel>C<nl><fcel>1<fcel>2<nl></otsl>")
            self.assertEqual((table.num_rows, table.num_cols), (2, 3))
            self.assertEqual(table.grid[1], ["1", "2", ""])

        # ---- 转义 / 文本 ----
        def test_pipe_escaped(self) -> None:
            md = otsl_to_markdown("<otsl><fcel>a|b<fcel>c<nl><fcel>d<fcel>e<nl></otsl>")
            self.assertIn("| a\\|b | c |", md)
            # 未被转义的竖线只应出现在表格分隔符里
            body = md.splitlines()[0]
            self.assertEqual(body.count("|"), 4)  # 4 个边框 + 1 个被转义的竖线字符

        def test_newline_in_cell_becomes_br(self) -> None:
            md = otsl_to_markdown("<otsl><fcel>A\nB<fcel>C<nl><fcel>D<fcel>E<nl></otsl>")
            self.assertIn("| A<br>B | C |", md)

        def test_header_tags(self) -> None:
            table = parse_otsl(
                "<otsl><ched>列1<ched>列2<nl><rhed>行1<fcel>v1<nl><rhed>行2<fcel>v2<nl></otsl>"
            )
            anchors = {(c.row, c.col): c for c in table.cells}
            self.assertTrue(anchors[(0, 0)].is_header)
            self.assertTrue(anchors[(1, 0)].is_header)
            self.assertEqual(anchors[(1, 1)].text, "v1")
            md = table.to_markdown()
            self.assertIn("| 列1 | 列2 |", md)
            html_out = to_html(table)
            self.assertIn("<th", html_out)
            self.assertIn(">列1</th>", html_out)

        def test_escaped_entities(self) -> None:
            table = parse_otsl("&lt;fcel&gt;A&lt;fcel&gt;B&lt;nl&gt;&lt;fcel&gt;C&lt;fcel&gt;D&lt;nl&gt;")
            self.assertEqual(table.grid, [["A", "B"], ["C", "D"]])

        def test_wrapped_in_markdown_fence(self) -> None:
            raw = "好的，结果如下：\n```markdown\n<otsl><fcel>A<fcel>B<nl><fcel>1<fcel>2<nl></otsl>\n```\n"
            table = parse_otsl(raw)
            self.assertEqual(table.grid, [["A", "B"], ["1", "2"]])

        def test_no_otsl_raises(self) -> None:
            with self.assertRaises(ValueError):
                parse_otsl("这是一段普通文本，没有表格结构。")

        def test_bare_continuation_cell(self) -> None:
            # 裸 <lcel> 没有锚点：不应抛异常，按空单元格处理
            table = parse_otsl("<otsl><lcel><fcel>A<nl><fcel>B<fcel>C<nl></otsl>")
            self.assertEqual(table.num_rows, 2)
            self.assertEqual(table.grid[1], ["B", "C"])

        def test_html_output_spans(self) -> None:
            table = parse_otsl("<otsl><fcel>A<lcel><fcel>B<nl><fcel>C<fcel>D<nl></otsl>")
            html_out = to_html(table)
            self.assertIn('colspan="2"', html_out)
            self.assertIn("<th", html_out)
            # 被 colspan 覆盖的格子不能重复输出
            self.assertEqual(html_out.count("<th"), 2)   # A(colspan=2) + B
            self.assertEqual(html_out.count("<td"), 3)   # C + D + 补齐的空 ecel
            self.assertEqual(html_out.count("</tr>"), 2)

        # ---- 性能兜底：真实样例转换耗时 ----
        def test_real_sample_conversion_is_fast(self) -> None:
            import time

            raw = load_real_samples()[0][1]
            t0 = time.perf_counter()
            for _ in range(100):
                otsl_to_markdown(raw)
            dt = (time.perf_counter() - t0) / 100
            self.assertLess(dt, 0.05, f"真实表格转换单次 {dt * 1000:.1f} ms，应 < 50 ms")

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestOtslToMarkdown)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="OTSL → Markdown/HTML 表格转换")
    parser.add_argument("--test", action="store_true", help="运行单元测试（含真实模型输出样例）")
    parser.add_argument("--otsl", default=None, help="直接给 OTSL 字符串")
    parser.add_argument("--file", default=None, help="从文件读取（自动截取 <otsl>…</otsl>）")
    parser.add_argument("--format", default="markdown", choices=("markdown", "html"))
    parser.add_argument("--no-header", action="store_true", help="首行不作表头")
    parser.add_argument("--demo", action="store_true", help="用内嵌真实样例演示转换")
    args = parser.parse_args(argv)

    if args.test:
        return _run_tests()

    if args.demo:
        name, raw = load_real_samples()[0]
        print(f"# 输入：{name}\n")
        print(raw, "\n")
        print("# Markdown\n")
        print(otsl_to_markdown(raw, header=not args.no_header))
        return 0

    if args.file:
        text = open(args.file, encoding="utf-8").read()
    elif args.otsl:
        text = args.otsl
    elif not sys.stdin.isatty():
        text = sys.stdin.read()
    else:
        parser.print_help()
        return 2

    if args.format == "html":
        print(otsl_to_html(text, header=not args.no_header))
    else:
        print(otsl_to_markdown(text, header=not args.no_header))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
