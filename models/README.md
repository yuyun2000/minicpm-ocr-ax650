# 模型包清单（HF 权重仓）

模型体积较大，**不放 GitHub**，放在 HuggingFace：

> 🤗 **`yunyu1258/minicpm-ocr-1.0-ax650`** ← 本仓库对应的转换产物（TODO：首次上传后补确认链接与 revision）

原始模型：`openbmb/MiniCPM-OCR-1.0`（1.30 B，Apache-2.0，ModelScope 直连可下）。

---

## 1. 文件清单

| 文件 | 大小 | 必需 | 说明 |
|---|---:|:---:|---|
| `config.json` | ~3 KB | ✅ | axllm 模型配置（见 `docs/conversion.md` §10 字段表） |
| `qwen3_5_text_p128_l{0..23}_together.axmodel` | 共 ≈575 MB | ✅ | 24 层文本 LLM，INT8，每层 20.6–26.6 MB |
| `qwen3_5_text_post.axmodel` | 1017 MB | ✅ | post（lm_head / final norm），BF16 |
| `model.embed_tokens.weight.bfloat16.bin` | 508 MB | ✅ | 248100×1024 bf16 embedding，mmap 加载 |
| `minicpm_ocr_tokenizer.txt` | 12.1 MB | ✅ | axllm 本地分词器（magic 430 / type 3 / HUGGINGFACE BPE） |
| `post_config.json` | <1 KB | ✅ | 采样配置（greedy 默认；内容 TODO 补齐） |
| `minicpm_ocr_vision_448.axmodel` | INT8 459 MB | ✅ | vision encoder（448×448 → 256×1024 image embeds） |
| `bin/axllm` | 2.4 MB | ✅ | AX650 aarch64 运行时（取自 AXERA 官方 AX650 包） |
| `bin/axllm.version.json` | <1 KB | ➖ | 运行时版本/commit 记录 |
| `service/` | ~500 KB | ➖ | Flask 测试台 + 前端 + 切片/OTSL 模块（也可以从本仓库拷） |

合计 ≈ **2.5 GB**。

> vision axmodel 目前清单里是 **INT8 版**（459 MB）。**必须换成 U16 版**才是可用精度
> （INT8 板端对拍 cosine 0.311，见 `docs/accuracy.md`）。U16 全量重编完成后：
> TODO —— 更新 HF 仓的 `minicpm_ocr_vision_448.axmodel`（体积会变大）、重新算 sha256、在下方表格补数值。

---

## 2. 下载

```bash
# 方式 A：huggingface-cli（国内加镜像）
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download yunyu1258/minicpm-ocr-1.0-ax650 \
  --local-dir ./axllm_pkg --local-dir-use-symlinks False

# 方式 B：git-lfs
GIT_LFS_SKIP_SMUDGE=1 git clone https://hf-mirror.com/yunyu1258/minicpm-ocr-1.0-ax650
cd minicpm-ocr-1.0-ax650 && git lfs pull
```

下载后应该 **26 个 `.axmodel`**（24 层 + post + vision）+ embed bin + tokenizer txt + `config.json` + `bin/`。

---

## 3. 校验

```bash
# 文件齐全性
ls -1 qwen3_5_text_p128_l*_together.axmodel | wc -l     # 期望 24
ls -lh qwen3_5_text_post.axmodel model.embed_tokens.weight.bfloat16.bin \
       minicpm_ocr_vision_448.axmodel minicpm_ocr_tokenizer.txt bin/axllm

# sha256（TODO：把首次上传后的实测值填进下表，并附 sha256sum.txt 文件）
sha256sum -c sha256sum.txt
```

| 文件 | 大小（实测） | sha256 |
|---|---:|---|
| `qwen3_5_text_p128_l0..23_together.axmodel`（24 个合计） | ≈575 MB | TODO |
| `qwen3_5_text_post.axmodel` | 1,017 MB | TODO |
| `model.embed_tokens.weight.bfloat16.bin` | 508 MB | TODO |
| `minicpm_ocr_tokenizer.txt` | 12,102,644 B | TODO |
| `minicpm_ocr_vision_448.axmodel`（INT8 现状） | 459 MB | TODO |
| `bin/axllm` | 2.4 MB | TODO |
| `model.safetensors`（上游原始权重，转选用） | 2,600,867,654 B | TODO |

> 大小是实测的（v2 包），hf 上以实际上传为准；sha256 需要在包定稿后生成一次 `sha256sum.txt` 随仓发布。

---

## 4. 包完整性自检（板端）

板端解包后，用仓库 `service/start.sh` 起服务，看这几行：

```
tokenizer_type = 3 / huggingface tokenizer mode = gpt2_byte_bpe   → tokenizer init ok
init 0..23 axmodel ok, remain_cmm(...)
init post axmodel ok, remain_cmm(...)
VisionModule init ok: type=MiniCPMV46VL, tokens_per_block=256, embed_size=1024, out_dtype=fp32
LLM init ok → OpenAI API Server starting on 0.0.0.0:18081
```

任一行缺失/报错 → `docs/troubleshooting.md` §3。

---

## 5. 许可证与再分发

| 组件 | 许可 | 说明 |
|---|---|---|
| MiniCPM-OCR 原始权重/代码 | **Apache-2.0**（`openbmb/MiniCPM-OCR-1.0`） | 转换产物遵守上游许可，需保留版权与许可声明 |
| `bin/axllm`（AXERA 官方运行时） | **BSD 3-Clause**（AXERA `ax-llm`） | 二进制再分发需保留 LICENSE；TODO：确认官方包对「随模型包一起分发」的具体要求 |
| 本仓库脚本/服务代码 | TODO（建议 MIT 或 Apache-2.0） | 发布前补 LICENSE |

**发布前必须做**（TODO）：

1. 在本仓库加 `LICENSE` 与 `NOTICE`，保留 AXERA / OpenBMB 的版权声明；
2. HF 模型仓 README 写明「非 AXERA / OpenBMB 官方发布，为社区转换产物」；
3. 确认 `bin/axllm` 的再分发许可（联系 AXERA 或引用其公开发布条款）。

---

## 6. TODO 汇总

1. HF 仓文件清单定稿（尤其 U16 vision axmodel）。
2. `sha256sum.txt` 生成与校验值填写。
3. `post_config.json` 实际内容（当前仓库脚本生成空对象）。
4. `bin/axllm.version.json` 的 commit 信息抄进本文。
5. HF 仓 README（模型卡：来源、转换方法、精度、性能、限制）。
