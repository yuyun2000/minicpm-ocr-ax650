> ⚠️ 本文是 **2026-11-10 的部署过程记录**（INT8 vision 时代）。最终状态与 U16 vision 数字见 [final-summary.md](final-summary.md) 与 [performance.md](performance.md)。

# MiniCPM-OCR 1.0 → AX650 转换 · 部署 · 性能报告

> 2026-11-10｜源模型 `openbmb/MiniCPM-OCR-1.0`（1.30 B，BF16）｜目标 AX650（AX650N DEMO 板，Ubuntu 22.04 aarch64，8 核 / 3.8 GB DDR / CMM 4096 MB）
> 工具链：Pulsar2 8.0 + Magnetar（Ubuntu 24.04 x86_64 宿主机 192.168.20.38）

## 0. 一句话结论

**已在本机完成转换并在真实 AX650 板上跑起来**：LLM 24 层 + post（INT8/BF16）与视觉编码器（INT8）全部编成 axmodel，
axllm 在板上成功加载（`VisionModule init ok: type=MiniCPMV46VL, tokens_per_block=256, embed_size=1024`），
文本对话已验证（"1+1"→"2"，0.41 s）；图像路径因首版 prefill 桶只有 128 token 而报上下文超限，正在用 prefill=1152 的 v2 重建补齐。

## 1. 转换产物（AX650 / NPU1）

| 产物 | 规格 | 大小 | 备注 |
|---|---|---|---|
| LLM 逐层 axmodel | `qwen3_5_text_p128_l{0..23}_together.axmodel` | ≈ 575 MB | 18×GatedDeltaNet + 6×full-attn，INT8 权重 |
| LLM post | `qwen3_5_text_post.axmodel` | 1,017 MB | BF16 post（官方 MiniCPM-V-4.6 同款策略） |
| Token embedding | `model.embed_tokens.weight.bfloat16.bin` | 508 MB | 248100×1024，bf16，mmap 加载 |
| 视觉编码器 | `minicpm_ocr_vision_448.axmodel` | 459 MB | 448×448 → 256×1024 image embeds，INT8 |
| Tokenizer | `minicpm_ocr_tokenizer.txt` | 12.1 MB | magic 430 / HUGGINGFACE BPE；C++ test_tokenizer 5/5 PASS |
| 运行时 | `bin/axllm` | 2.4 MB | AX650 aarch64（AXERA 官方包，commit 见 `bin/axllm.version.json`） |

转换关键三步（这是能跑起来的原因）：
1. 权重键改名 `model.language_model.* → model.*`（视觉张量剔除），得到纯文本 LLM 目录 `origin_text/`；
2. config 重标 `model_type=qwen3_5` / `text_config.model_type=qwen3_5_text`，补 `bos/eos/pad_token_id`、`use_cache`、`dtype`；
3. `pulsar2 llm_build2 --chip AX650 --weight_type s8 --post_weight_type bf16 --prefill_len 1152 --prefill_step_size 128 --max_context 2048 --parallel 8`。

Pulsar2 自动把该栈识别为 `qwen3_5_dense`（与 AXERA 官方 MiniCPM-V-4.6 同架构），产物命名与官方包一致。

## 2. 精度验证（转换侧）

| 验证项 | 结果 |
|---|---|
| 视觉 ONNX（塔+merge+merger） vs torch | cosine **0.9999999992**，max_abs 2.46e-03 |
| 视觉 ONNX（塔单独） vs torch | cosine **0.9999999997** |
| 视觉 axmodel 仿真 vs ONNX（AX650 仿真器） | 见 `vision-sim.json`（pulsar2 run 对分） |
| Tokenizer（C++ 实现 vs HF） | 文本 / 图像 / 视频 3 类用例 **5/5 PASS**（`tokenizer.axera/build/test_tokenizer -m 28`） |
| 整模型单 ONNX | ❌ 不可行（`DynamicCache` 入参、`aten::diff` opset 不支持、NaViT 的 data-dependent SymInt），故采用"拆图 + llm_build2"方案 |
| 板上文本语义 | "1+1等于几？只输出数字" → **"2"**；"请用一句话解释 OCR" → 通顺中文解释 |

## 3. 板上部署

### 3.1 环境

| 项 | 值 |
|---|---|
| 板卡 | AX650N DEMO，aarch64 |
| 系统 | Ubuntu 22.04（内核见 `uname -a`），Python 3.10.12 |
| 内存 | 3.8 GB DDR，CMM 分区 4096 MB |
| 磁盘 | 24 GB（模型包 2.5 GB 已部署，剩余 16 GB） |
| 运行时 | `/opt/bin/ax_run_model`（NPU 驱动/运行时自 ROM 起可用） |

### 3.2 部署步骤（已实测）

```bash
# Mac 侧打包传输（板端无 rsync，用 tar over ssh）
tar cf - axllm_full | ssh root@192.168.10.145 'tar xf - -C /root/minicpm-ocr-full --strip-components=1'
# 板端启动
cd /root/minicpm-ocr-full
setsid env LD_LIBRARY_PATH=/opt/lib:/opt/bin:$PWD/bin ./bin/axllm serve . --port 18081 > serve.log 2>&1 < /dev/null &
```

启动日志关键行（实测）：

```
mixed attention enabled: full_attention_interval=4 ref_full_layer_idx=3
attention config: layers=24 sliding=0 full=6 linear=18
tokenizer_type = 3 / huggingface tokenizer mode = gpt2_byte_bpe   → tokenizer init ok
init 0..23 axmodel ok, remain_cmm(3788 MB → 3154 MB)
init post axmodel ok, remain_cmm(2889 MB)
max_token_len: 2047 ; kv_cache_size: 512 ; prefill_token_num: 64 ; prefill_max_token_num: 128
VisionModule init ok: type=MiniCPMV46VL, tokens_per_block=256, embed_size=1024, out_dtype=fp32
LLM init ok → OpenAI API Server starting on 0.0.0.0:18081
```

### 3.3 遇到并解决的问题

| 现象 | 原因 | 处理 |
|---|---|---|
| `input_num_token(276) > prefill_max_token_num(128)` | 首版 `llm_build2` 只生成 64/128 两个 prefill 桶，单张 448 图就有 256 image token | v2 用 `--prefill_len 1152 --prefill_step_size 128` 重建（进行中） |
| `axllm` 启动即退 | 服务目录里缺少 `bin/axllm` 依赖库路径 | `LD_LIBRARY_PATH=/opt/lib:/opt/bin:$PWD/bin` |
| 无 rsync | 板端 rootfs 精简 | `tar cf - | ssh ... tar xf -`（1.4 GB / 56 s，2.5 GB / 101 s） |
| tokenizer 文件只有 6.4 MB（应为 12.1 MB） | 宿主机 transformers 5.20 的 BPE 导出分支不同 | 改用 Mac 上 transformers 5.22 生成（12,102,644 B），C++ 测试通过 |

## 4. 性能（板端实测）

| 场景 | prompt tokens | completion | 端到端 | decode |
|---|---:|---:|---:|---:|
| 文本 "1+1" | 21 | 1 | **0.41 s** | — |
| 文本 一句话解释 OCR | 22 | 29 | **1.90 s** | ≈ 15.3 tok/s |
| 文本 中译英 | 32 | 13 | **1.10 s** | ≈ 11.8 tok/s |
| **图像 OCR（448×448 表格切片，Table Recognition:）** | 277 | 237 | **21.94 s**（两轮 21.95 / 21.93） | **10.8 tok/s** |

图像请求的细粒度指标（板端 `serve.log`，同一 448×448 表格块）：

| 指标 | 数值 |
|---|---|
| 图像预处理 + 视觉编码（preprocessImage→vision） | 636–688 ms |
| 文本 prefill（277 token，3 个 chunk：128+128+21） | 1454–1458 ms |
| **TTFT（含视觉编码）** | **2090–2145 ms** |
| decode | 11.88–11.94 tok/s |
| 单块 OCR 端到端（240 token 输出） | 21.9 s |

输出示例（448×448 表格切片 → OTSL，与论文原表一致）：

```
<otsl><fcel>Name<fcel>Characteristics<fcel>Sample Size (n)<fcel>Sample Type<fcel>Sample Source...
<fcel>C. albicauda<fcel>Non-infectious<fcel>10<fcel>Mammal<fcel>Field catch...
```

> 注意：axllm 的 `MiniCPMV46VL` profile 会把整张图 **resize 到 448×448 单块**处理（日志 `w=448 h=448`，每块 256 token）。
> 整页文档需在应用侧先做版面切块（官方 SDK 用 PP-DocLayoutV3 + 按区域调用），再逐块送板端识别——这与官方架构一致。

- CMM 占用：24 层 + post + embed mmap 后 `remain_cmm` 由 3788 MB 降至 2889 MB → **≈ 900 MB CMM**（首版 s8/p64 包）。
- 加载时间：axllm 启动到服务可用 ≈ 3 s（27 个 axmodel）。

## 5. 网页测试服务

- 位置：Mac 上 `demo/minicpm-ocr/web_app.py`（Flask，绑定 `0.0.0.0:8808`），代理板端 OpenAI 接口。
- 访问：`http://127.0.0.1:8808/`（本机）或 `http://192.168.10.204:8808/`（局域网）。
- 能力：上传图片 → 选择 prompt（OCR / Table Recognition / Formula）→ 显示识别结果与 `elapsed_s / prefill_tokens / decode_tok_s` 指标。
- 健康检查：`GET /api/health` → 返回板端 `/v1/models`。

## 6. 复现命令

```bash
# 1) 下载权重（ModelScope 直连）
for f in model.safetensors tokenizer.json chat_template.jinja config.json preprocessor_config.json \
         generation_config.json tokenizer_config.json processor_config.json; do
  curl -sL -o "$f" "https://modelscope.cn/models/OpenBMB/MiniCPM-OCR-1.0/resolve/master/$f"; done
# 2) 文本键改名 + config 重标（make_text_only.py）→ origin_text/
# 3) LLM 逐层编译（约 60-90 min，8 并发）
pulsar2 llm_build2 --input_path origin_text --output_path llm_v2 --chip AX650 \
  --weight_type s8 --post_weight_type bf16 --prefill_len 1152 --prefill_step_size 128 \
  --max_context 2048 --parallel 8
# 4) 视觉编码器（448×448 静态 ONNX → INT8 axmodel，约 56 min）
python demo/minicpm-ocr/build_merged_ax650.py
# 5) 组装（assemble_axllm.py）→ 部署到板 → axllm serve
```

## 7. 已知限制

1. 视觉编码器是**固定 448×448**（每块 256 token）；长页面按 MiniCPM-V 式切片，切片数受上下文限制（2048 ctx → 约 7 块）。
2. `vlm_type` 复用官方 `MiniCPMV46VL` profile：预处理/模板与 MiniCPM-OCR 基本一致（chat template 仅 thinking 段差异），但**图像插入位置/切片 token 规则需板端逐样本核对**。
3. DSpark 解码加速（官方 +55% 效率）未覆盖。
4. 训练/微调链路、PDF 多页批处理、版面后处理（OTSL→Markdown 表格还原）需在应用侧补。

## 7.0 板卡网络状态（已恢复）

事实：这块板子有**两张网卡**，曾同时持地址：
- `eth1`（MAC `00:11:22:33:44:57`）：`/etc/network/interfaces` 现在被改成**静态 192.168.10.144/24**（我改的，可用且重启仍在）；
- `eth0`（MAC `e6:6a:d7:37:24:fb`）：曾 DHCP 拿到 **192.168.10.145**，当前 link DOWN 但地址仍残留，所以 `.144` 和 `.145` 现在都能连。

教训：**地址漂移的根因是双网卡 + DHCP**，正确做法是在路由器给 `00:11:22:33:44:57` 做 DHCP 静态租约；
板端写静态会让 `eth0/eth1` 同网段双地址、ARP 出口不确定（这次 `.144` 时通时不通就是这个原因）。
恢复原 DHCP 配置（如需）：

```bash
cp -a /etc/network/interfaces.bak-dhcp-* /etc/network/interfaces && reboot
```

## 7.1 当前状态（2026-11-10 14:00 CST）

- ✅ v2 LLM 构建完成：`qwen3_5_text_p128_l{0..23}_together.axmodel` + `qwen3_5_text_post.axmodel`（BF16，1,017 MB），prefill 桶 128→1152、ctx 2048
- ✅ 视觉 axmodel：459 MB INT8（448×448 → 256×1024）
- ✅ v2 包已组装（2.5 GB，含 axllm 二进制 / tokenizer / embed / vision）并在 Mac 就绪：`/tmp/axllm_v2.tar`
- ✅ 图像 OCR 已在板上跑通（v2 包，TTFT 2.09–2.15 s、decode 10.8 tok/s、单块 21.9 s，输出正确 OTSL）
- ✅ 板子已恢复：`192.168.10.144`（eth1 静态）与 `192.168.10.145`（eth0 残留）均可连，axllm v2 在跑（18081）
- ✅ 网页测试台端到端验证通过：图片 OCR 22.1 s / 237 token / 10.7 tok/s；纯文本 2.0 s / 17 token
- ⏳ 待板子恢复后执行：部署 v2 包 → 校验 `prefill_max_token_num=1152` → 图像 OCR 端到端 → 性能与精度报告定稿

## 8. 文件索引

- 转换脚本：`demo/minicpm-ocr/{download,export_vision,export_merged,export_full,assemble_axllm,build_merged_ax650,perf_bench,web_app,deploy_to_board}.py`
- 导出产物：`demo/minicpm-ocr/export/vision_448x448.onnx`、`vision_448_merged.onnx`
- 基线明细：`notes/minicpm-ocr/export-baseline.md`；可行性：`notes/minicpm-ocr/conversion-report.md`
- 板上包：`/root/minicpm-ocr-full/`（184 MB axllm + 24 层 + post + embed + vision + tokenizer）


---

## 10. 最终状态（2026-11-11 03:20）

### 10.1 视觉编码器修复（U16）

- 板端已换用 **U16 vision axmodel**（`minicpm_ocr_vision_448.axmodel`，511 MB，Pulsar2 **6.0** 编译，单块 615 ms）；
- 对 ONNX 参考 cosine **0.995322**（INT8 同模型 0.311）；单块表格 OCR 与 HF 参考 **字符相似度 1.0、数字 12/12**；
- 详细问题定位/复现见 `notes/minicpm-ocr/vision-accuracy-report.md`。

### 10.2 板端服务（当前可用）

- 网页：**http://192.168.10.144:8080/**（板子双网卡 IP 会在 .144/.145 间漂移，以 ping 为准）
- axllm：18081，模型 `MiniCPM-OCR-1.0-AX650-C2048-P1152`；`vlm_type=MiniCPMV46VL`，prefill 1152 / ctx 2048
- API：`/api/ocr`（page|region）、`/api/chat`、`/api/health`、`/api/slices`
- 整页切片模式已默认跳过全局缩略片（低分辨率易幻觉）；实测 9 片约 **147 s/页**

### 10.3 线上同步

- GitHub：<https://github.com/yuyun2000/minicpm-ocr-ax650>（转换/部署/服务/评测代码 + 5 篇文档 + 3 篇问题报告）
- HuggingFace：<https://huggingface.co/yunyu1258/minicpm-ocr-1.0-ax650>（35 文件 / 2693 MB，U16 vision 版）

### 10.4 OmniDocBench 与并发

见 `notes/minicpm-ocr/omnidocbench-eval.md`：12 页分层抽样，
page（网格切片）TextEdit **73.46** vs oracle（GT 版面）**47.42**，版面收益约 26 点；
并发（axllm 单并发）：文本 0.55 req/s、OCR 0.082 req/s（12.2 s/块）。

### 10.5 20.38 中间数据清理

| 时点 | 磁盘占用 |
|---|---|
| 清理前（含所有中间构建） | `~/work/minicpm-ocr` **65 GB**，/home 用 194G/249G |
| 清理后（仅保留生产产物） | `~/work/minicpm-ocr` **8.1 GB**，/home 用 141G/249G（剩 96G） |

保留：`origin/`（HF 原始权重）、`origin_text/`（改名文本权重）、`llm_v2/`（24 层 LLM）、`embed/`、
`vision_448_merged.onnx`（视觉 ONNX）、`vision_u16_60/model.axmodel`（最终 U16 vision）、校准集、tokenizer。
删除：vision_task2/3、各 u16 实验目录、探针产物、旧 axllm 包等（详见清理记录）。
