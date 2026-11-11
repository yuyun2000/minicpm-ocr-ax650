# 模型包（HuggingFace）

🤗 **<https://huggingface.co/yunyu1258/minicpm-ocr-1.0-ax650>**

35 个文件 / **2,693 MB**（含 U16 vision axmodel）。下载：

```bash
export HF_ENDPOINT=https://hf-mirror.com   # 国内
huggingface-cli download yunyu1258/minicpm-ocr-1.0-ax650 \
  --local-dir ./axllm_pkg --local-dir-use-symlinks False
```

完整性校验（模型仓里也带了 `sha256sum.txt`）：

```bash
cd axllm_pkg && shasum -a 256 -c sha256sum.txt
```

## 文件清单（实测大小）

| 文件 | 大小（字节） | sha256（前 16 位） | 说明 |
|---|---:|---|---|
| `config.json` | 1,411 | `71ed288047b24002` | axllm 配置（vlm_type=MiniCPMV46VL，prefill 桶 1152 / ctx 2048） |
| `post_config.json` | 279 | `21ecd0699e5ba6a1` | 采样配置（greedy 默认） |
| `qwen3_5_text_p128_l{0..23}_together.axmodel`（24 个） | 每个 22.6–26.8 M | 见 `sha256sum.txt` | 24 层 Qwen3.5-hybrid LLM 逐层 axmodel |
| `qwen3_5_text_post.axmodel` | 1,016,919,008 | `d9d59c1ee746be94` | final norm + lm_head |
| `model.embed_tokens.weight.bfloat16.bin` | 508,108,800 | `d2ecb95a65466693` | 词表 embedding（248100 × 1024, bf16） |
| `minicpm_ocr_tokenizer.txt` | 12,102,644 | `96476e1f29264b6d` | HF tokenizer → axllm txt（magic 430 / BPE） |
| `minicpm_ocr_vision_448.axmodel` | **535,663,750** | `1039487da423f4de` | **U16 vision**（448×448 → [1,256,1024]），Pulsar2 6.0 编译 |
| `doclayout_bl_u16.axmodel` | 37,173,317 | 见 `sha256sum.txt` | **可选**：DocLayout-YOLO 版面检测 axmodel（U16，1024×1024，212 ms） |
| `bin/axllm` | 2,496,048 | `5c0ac7775b9dab05` | AX650 aarch64 运行时（AXERA 官方包） |
| `client_ocr.py` / `run.sh` | 1,874 / 291 | 见清单 | 示例客户端与启动脚本 |
| `README.md` / `sha256sum.txt` | — | — | 模型卡与校验清单 |

> **注意**：vision 必须是本清单里的 **U16 版（535,663,750 B）**。早期 INT8 版（459,308,822 B）对 ONNX 参考
> cosine 只有 **0.311**，会让模型"看不到图"，不要使用（`models/` 里也没有它）。

## 版本对照

| 项 | 数值 |
|---|---|
| 视觉编码器 | U16，511 MB，板端单块 448×448 **615 ms**，对 ONNX cosine **0.995322** |
| LLM（24 层 + post） | INT8 层权重 + BF16 输出；decode ≈ **11.6 tok/s** |
| 包体积 / CMM | 2.5 GB / 加载后 CMM 剩余 **2130 MB** |
| 精度（单块表格 vs HF 参考） | 字符相似度 **1.0**、数字 **12/12** |

## 运行

```bash
tar xf axllm_pkg.tar -C /root/minicpm-ocr-v2   # 或直接 scp -r
cd /root/minicpm-ocr-v2 && chmod +x bin/axllm
LD_LIBRARY_PATH=/opt/lib:/opt/bin:$PWD/bin ./bin/axllm serve . --port 18081
```

## 许可

- 底座模型：`openbmb/MiniCPM-OCR-1.0`（Apache-2.0）
- 运行时 `axllm`：AXERA（BSD-3-Clause，二进制随包分发，保留 LICENSE）
- 本转换产物（axmodel / 配置 / 脚本）：Apache-2.0
