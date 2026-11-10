# MiniCPM-OCR 1.0 on AX650

把 [MiniCPM-OCR 1.0](https://huggingface.co/openbmb/MiniCPM-OCR-1.0)（1.30 B 端到端 OCR 视觉语言模型）完整跑在
**AX650 / AX650N** NPU 上的转换流程 + 可运行模型包 + 板端网页测试台。

- **视觉编码器**：27 层 SigLIP 类塔 → 静态 ONNX → axmodel（**必须 U16**，INT8 会崩）
- **文本 LLM**：Qwen3.5 hybrid（24 层 = 18 线性注意力 + 6 full attention）→ `pulsar2 llm_build2` 逐层 axmodel
- **运行时**：AXERA `axllm`（OpenAI 兼容接口），prefill 桶 1152 / ctx 2048
- **应用侧**：MiniCPM-V 式切片（9 片 / 2276 token，与官方 processor 逐片一致）+ OTSL→Markdown + 浏览器测试台

> ⚠️ 这是**社区转换产物**，不是 AXERA / OpenBMB 官方发布。模型仓：
> 🤗 [`yunyu1258/minicpm-ocr-1.0-ax650`](https://huggingface.co/yunyu1258/minicpm-ocr-1.0-ax650)（2.5 GB，U16 vision 版）

---

## 实测数字（2026-11-10，真实 AX650N 板）

| 场景 | 结果 |
|---|---|
| 文本 "1+1"（21 token prompt → 1 token） | **0.41 s** |
| 文本 "一句话解释 OCR"（22 → 29） | **1.90 s** / ≈15.3 tok/s |
| **图像 OCR**（448×448 表格块，277 → 237 token） | **21.94 s** / **10.8 tok/s** |
| TTFT（含视觉编码） | **2.09–2.15 s** |
| 视觉编码器对拍（U16，2 层探针） | cosine **0.997895** |
| 视觉编码器对拍（INT8，27 层） | cosine **0.311** ❌（所以必须 U16） |
| 切片对齐官方 processor | **9 片 / 2276 image token**，逐片尺寸/网格/token 完全一致 |
| 模型包 / CMM 占用 | ≈2.5 GB / ≈900 MB |

细节：[docs/performance.md](docs/performance.md)、[docs/accuracy.md](docs/accuracy.md)。

---

## 上手：三步

### 1. 下模型包（HF）

```bash
export HF_ENDPOINT=https://hf-mirror.com   # 国内
huggingface-cli download yunyu1258/minicpm-ocr-1.0-ax650 \
  --local-dir ./axllm_pkg --local-dir-use-symlinks False
```

期望得到 **26 个 `.axmodel`**（24 层 + post + vision）、embed bin、tokenizer txt、`config.json`、`bin/axllm`。
文件清单与校验方式见 [models/README.md](models/README.md)。

### 2. 传到板子并启动

```bash
# 转换机 / Mac 上（需要能 ssh 到板子）：tar over ssh，板端没有 rsync
./scripts/deploy_to_board.sh \
  --pkg ./axllm_pkg --host root@192.168.10.144 \
  --dst /root/minicpm-ocr-v2 --port 18081
# 想同时用网页测试台：把本仓库 service/ 一起拷到模型包目录下（或直接在板上 git clone 本仓库）
```

板端手工启动（与脚本等价）：

```bash
cd /root/minicpm-ocr-v2
chmod +x bin/axllm
setsid env LD_LIBRARY_PATH=/opt/lib:/opt/bin:$PWD/bin \
  ./bin/axllm serve . --port 18081 > serve.log 2>&1 < /dev/null &

# 自检：应返回 model id = MiniCPM-OCR-1.0-AX650-C2048-P1152
curl -s http://127.0.0.1:18081/v1/models | head -c 300
```

### 3. 浏览器访问

```bash
cd /root/minicpm-ocr-v2
./service/start.sh start          # axllm(18081) + Flask 测试台(8080) 一起起
./service/start.sh status
```

打开 **`http://<板子 IP>:8080/`**：上传图片 → 选任务词（Text / Table / Formula / Seal / Chart）
→ 整页或单区域 → 看 Markdown 结果 + TTFT / decode / 切片指标。

![板端网页测试台（界面演示数据，非板端实测）](service/web/preview.png)

> 截图是界面演示（指标为占位数据，不是实测）；真实板端数字见上面的「实测数字」表。

接口（不走网页也能用）：

```bash
# 单块图像
curl -s http://<board-ip>:18081/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "MiniCPM-OCR-1.0-AX650-C2048-P1152",
  "messages": [{"role":"user","content":[
    {"type":"text","text":"Table Recognition:"},
    {"type":"image_url","image_url":{"url":"data:image/png;base64,...."}}]}],
  "max_tokens": 768, "temperature": 0.0}'
```

---

## 目录

```
minicpm-ocr-ax650/
├── README.md                  # 本文件
├── docs/
│   ├── conversion.md          # HF → 纯文本 LLM → llm_build2 → vision ONNX → U16 axmodel → 组装（含踩坑）
│   ├── deployment.md          # 板端部署：tar 传输、axllm 配置、端口、CMM、服务化
│   ├── accuracy.md            # 精度：vision cosine 表、U16 结论、切片 2276 token、tokenizer
│   ├── performance.md         # 实测性能：TTFT / decode / 内存 / 与 CPU 基线对比
│   └── troubleshooting.md     # 已知问题：INT8 崩、prefill 桶、CMM、pkill 自杀…
├── scripts/
│   ├── prep_text_model.py     # 权重改名 model.language_model.* → model.* + config 重标 qwen3_5
│   ├── build_llm.sh           # llm_build2（s8/bf16, prefill 1152, ctx 2048）
│   ├── export_vision_onnx.py  # HF → vision_448_merged.onnx（NaViT 打包，1×3×14×14336 → 1×256×1024）
│   ├── build_vision.py        # pulsar2 build vision（官方 U16 配置 + 64 张真实校准）
│   ├── assemble_package.py    # 组装 axllm 模型包（config.json / 24 层 / embed / post / vision / bin）
│   ├── deploy_to_board.sh     # tar over ssh 传包 + 板端拉起 + 就绪检查
│   └── compare_ocr_output.py  # 板端 vs 参考输出的字符相似度 / 表格 cell F1 / 数字准确率
├── service/
│   ├── start.sh               # 板端一键起停（axllm 18081 + Flask 8080）
│   ├── board_app.py           # Flask：/api/ocr(page|region)、/api/chat、/api/health、/api/slices
│   ├── slicing.py             # 官方切片逻辑复刻（纯 numpy+PIL，与 processor 逐片对齐）
│   ├── otsl_to_md.py          # OTSL → Markdown/HTML（板端渲染表格）
│   ├── samples/               # 真实模型输出样例（otsl 单元测试用）
│   └── web/                   # M5Stack 风格静态前端（零依赖、无构建）
└── models/README.md           # HF 模型包文件清单 / 校验 / 许可
```

---

## 进阶：从零转换

不想用打包好的模型、想自己转一遍：看 **[docs/conversion.md](docs/conversion.md)**，最短路径是

```bash
# 1) 下载 HF 原始权重（ModelScope 直连更快）
# 2) 拆纯文本 LLM + config 重标
python3 scripts/prep_text_model.py --src origin --out origin_text
# 3) 24 层文本 LLM
INPUT=origin_text OUT_DIR=llm_v2 bash scripts/build_llm.sh
# 4) vision 静态 ONNX
python3 scripts/export_vision_onnx.py --model-dir origin --out-dir .
# 5) vision axmodel（U16！）
python3 scripts/build_vision.py --onnx vision_448_merged.onnx --calib calib/pixel_values.tar.gz \
    --out-dir vision_task
# 6) 组装 + 推板
python3 scripts/assemble_package.py --work-dir . --dest axllm_pkg \
    --tokenizer minicpm_ocr_tokenizer.txt --axllm-bin /path/to/axllm --ctx 2048
./scripts/deploy_to_board.sh --pkg axllm_pkg --host root@<board> --port 18081
```

**硬前置**：编译必须在 **Linux x86_64** 上做（Pulsar2 只发 Linux x86_64 独立包，macOS 最多导 ONNX）；
需要 Pulsar2 8.0；vision 必须 **U16**；tokenizer.txt 用 transformers 5.22 生成。

---

## 已知限制

1. **vision 固定 448×448 单块**（每块 256 image token）。整页靠应用侧切片，板端没有官方 SDK 的版面模型（PP-DocLayoutV3）。
2. `max_context=2048`：整页 9 片 ≈2276 image token 会超上下文，整页模式要限片数。
3. decode 10.8 tok/s 是当前瓶颈（单块 237 token 输出 ≈20 s）；DSpark 解码加速未覆盖。
4. `vlm_type` 复用官方 `MiniCPMV46VL` profile；图像插入位置/切片 token 规则需逐样本核对。
5. 训练/微调、PDF 多页批处理、版面后处理未包含。
6. 网页测试台 `slice_times` 单位（秒 vs 毫秒）已知不一致。

---

## 待补充（TODO）

- [ ] **U16 全量 vision axmodel 的板端对拍数值 / 体积 / 耗时**（2 层探针 0.997895 已达标，全量结果未定稿）
- [ ] HF 模型仓定稿：U16 vision 版本、`sha256sum.txt`、模型卡
- [ ] v2（prefill 1152）包完整启动日志与 `max_token_len/预填充桶` 实测行
- [ ] 板端 vs HF CPU 基线的表格 cell F1 / 字符相似度（`scripts/compare_ocr_output.py` 已就绪）
- [ ] 整页多切片的端到端质量与建议片数上限
- [ ] `FLOAT_MATMUL_USE_CONV_EU=1` 对构建的影响（官方脚本有，本项目未记录）
- [ ] 板端开机自启（systemd）、Flask 上传体积/并发上限
- [ ] LICENSE / NOTICE（含 AXERA axllm BSD-3 与 OpenBMB Apache-2.0 声明）

---

## 致谢

- 模型：[OpenBMB / MiniCPM-OCR](https://github.com/OpenBMB/MiniCPM-OCR)（Apache-2.0）
- 工具链：[AXERA-TECH Pulsar2](https://huggingface.co/AXERA-TECH) 与 [ax-llm](https://github.com/AXERA-TECH/ax-llm)（BSD-3-Clause）
- 视觉转换配置参考：[AXERA-TECH/MiniCPM-V-4.6.axera](https://github.com/AXERA-TECH/MiniCPM-V-4.6.axera)
- 转换流程编排：[AXERA-TECH/Magnetar](https://github.com/AXERA-TECH/Magnetar)


---

## 最终验证结果（2026-11-11）

| 项 | 结果 |
|---|---|
| 视觉编码器 U16 axmodel 体积 / 板端耗时 | **511 MB / 615 ms**（448×448 → [1,256,1024]） |
| 视觉 U16 对 ONNX 参考 cosine（全量 27 层 + merger） | **0.995322** ✅（INT8 同模型仅 **0.311** ❌） |
| 单块表格 OCR（HF 参考对拍） | 字符相似度 **1.0**、数字准确率 **12/12** |
| 整页切片模式（9 片，跳过全局缩略片） | **147 s/页**，1515 completion token |
| 文本对话 | "1+1"→"2" 约 0.6 s |
| 并发（axllm 最大并发 1） | 文本并发 1：0.55 req/s（1.82 s/req）；OCR 并发 1：0.082 req/s（12.2 s/块） |

**关键结论**：视觉编码器必须 **U16 激活**，且必须用 **Pulsar2 6.0** 编译全量图——7.0 / 8.0 都会在
`build op serially` 阶段原生崩溃（`new_ddr_tensor` 后进程被杀），6.0 可稳定编出 511 MB axmodel。
详见 [docs/accuracy.md](docs/accuracy.md) 与转换仓库的问题报告。


## OmniDocBench 评测（2026-11-11）

见 [docs/omnidocbench.md](docs/omnidocbench.md)。摘要（12 页分层抽样）：

| 模式 | 平均文本编辑距离 | 平均耗时 |
|---|---|---|
| page（网格切片 + 统一 prompt，现状） | **0.7346（TextEdit 73.46）** | 91.8 s/页 |
| oracle（GT 版面框 + 按类型 prompt，上限） | 见文档（运行中） | — |

并发：文本并发 1 → 0.55 req/s（1.82 s）；OCR 并发 1 → 0.082 req/s（12.2 s/块）；axllm 单并发，请求排队。
官方 1.2 pages/s 是 vLLM/L40S 口径，端侧 AX650 差 2~3 个数量级属预期。
