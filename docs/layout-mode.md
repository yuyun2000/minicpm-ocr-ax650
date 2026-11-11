# 版面模式：DocLayout-YOLO 转 AX8850 + 版面→区域 OCR 流水线（2026-11-11）

## 目标

补上原始仓库的「版面模型」环节：检测标题/正文/表格/公式区域 → 按类型逐区域识别 → 按阅读顺序拼接，
解决「整页文档里表格被网格切片切断、阅读顺序乱」的问题。

## 模型与转换（服务器 192.168.20.38 上用 Pulsar2 转）

来源：`wybxc/DocLayout-YOLO-DocStructBench-onnx`（75 MB，YOLOv10 系 NMS-free 头，10 类：
title / plain text / table / figure / figure_caption / table_caption / table_footnote /
isolate_formula / formula_caption 等），输入动态 `[batch,3,H,W]`，输出 `[1,300,6]`。

Pulsar2 直接转会失败，逐步修了 4 处（都在 ONNX 侧、可复现）：

| # | 问题 | 处理 |
|---|---|---|
| 1 | 动态输入/输出维度 → `AssertionError: illegal value_info images` | 固定输入 `1x3x1024x1024`、输出维度写死 |
| 2 | opset 19 → `Cast version must be <19` | `onnx.version_converter` 降到 **opset 18**（输出与原图 max diff 0） |
| 3 | `/model.23/Mod` 不支持 → `get opr failed` | 用 `Div/Mul/Sub` 替换 `Mod`（`a-(a/b)*b`，int 截断除法；随机输入验证 diff 0） |
| 4 | 图内 `TopK/GatherElements` 头 Pulsar2 解析失败（`check onnx parser e2e failed`） | **切掉 NMS-free 头**：用 `onnx.utils.extract_model` 输出 `Mul_3`（boxes [1,4,21504]）和 `Split_1`（score logits [1,10,21504]）两个张量，top-k/NMS 放到板端 Python |

量化：**INT8 精度不够**（logits cosine 0.9866，出现大量伪检测）；改 **U16 激活**（`layer_configs DEFAULT→DEFAULT U16` +
conv bias FP32）后：**boxes cosine 0.999999 / logits cosine 0.99995**，伪检测消失。

| 项 | 数值 |
|---|---|
| axmodel 体积 | **35 MB**（U16）/ 30.5 MB（INT8） |
| 板端单次推理（ax_run_model） | **212.6 ms**（U16，1024×1024）/ 120.5 ms（INT8） |
| 板端检测结果（用户表格图） | `table conf 0.814 box [236.3,0,1096.8,974.5]` = ONNX 参考 `0.828 [236.9,0,1096.6,975.0]` ✅ |

## 板端集成（`service/board_app.py`）

- `mode=layout`：`ax_run_model` 跑版面模型（进程内临时目录，212 ms）→ 后处理（sigmoid、conf≥0.35、
  面积≥0.2% 页面积、同类别 NMS IoU 0.5）→ 阅读顺序（按 y 聚类成行、行内按 x）→ 逐区域识别：
  - `table` → `Table Recognition:` → OTSL → Markdown（大表按行切分合并）
  - `isolate_formula` → `Formula Recognition:`；其余 → `Text Recognition:`
- **单表格裁片兜底**：若只检出一个 table 且覆盖 >35% 页面 → 整图按表格识别（避免检测框裁掉列）。
- `auto` 模式：有版面模型时，竖长文档页（h≥1.2w 且长边≥1200，或长边≥2200）自动走 **layout**；
  表格线明显的裁片仍走 table；宽条走 wide；其余 region。
- `/api/health` 返回 `layout_model` 字段，前端展示。

## 实测（板端，`http://192.168.10.144:8080/`）

| 输入 | 模式 | 结果 |
|---|---|---|
| 用户表格截图 1100×986 | layout（单表兜底） | **22.9 s**，6 列 × 10 行 Markdown 全对 |
| PPT 文档页（多区域） | layout | **19.8 s**，6 个区域（title + 5 plain text），按阅读顺序拼接可读 |
| 整页表格页 2000×3083 | layout | **53.8 s**，检出 table（0.916）+ table_footnote（0.906），输出表格 + 脚注 |
| 版面模型单独推理 | - | 212.6 ms/次 |

## 仍有限制

- 2000px 宽的整页表格缩到 448 后小字识别会掉字（输入分辨率物理上限），需要更细的版面/表格结构模型才行；
- 版面后处理（NMS/阅读顺序）目前是简单行聚类，复杂多栏页可再优化；
- 版面 axmodel 是可选组件：`LAYOUT_MODEL` 环境变量未设置时，`layout`/auto-layout 自动回退到 page 模式。

## 复现命令

```bash
# 服务器（192.168.20.38）
cd ~/work/doclayout
# 1) 静态化 + opset18 + 去 Mod + 切两输出（脚本在 Mac 上跑，产物 ~/work/doclayout/doclayout_boxlogit_static.onnx）
# 2) U16 build
python -c "...pulsar2 build --config buildu16/pulsar2_config.json..."   # 见 build_doclayout.py + buildu16/
# 板端
scp build/buildu16/doclayout_bl_u16.axmodel root@<board>:/root/minicpm-ocr-v2/
# 启动服务时带 LAYOUT_MODEL=.../doclayout_bl_u16.axmodel
```
