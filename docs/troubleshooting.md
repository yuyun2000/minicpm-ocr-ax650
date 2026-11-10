# 已知问题与排查（Troubleshooting）

> 按「现象 → 原因 → 处理」组织。所有条目都来自 2026-11-10 的实测；未复现/未确认的标注 `TODO`。
> 排障原则：**先分层**（文本链路 / 视觉链路 / 应用侧），别一上来就怀疑整条链。

---

## 0. 先分层定位（最重要）

```
整个服务不动？
├─ axllm 没起来         → §2 / §3（LD_LIBRARY_PATH、端口、pkill 自杀、CMM）
├─ 纯文本能回、图不行   → §1（vision 精度）→ 先跑 docs/accuracy.md §2 的 ax_run_model 对拍
├─ 单块图正常、整页乱   → §4（切片 / token 占位符 / 上下文）
└─ 命令行 OK、网页不行  → §6（Flask / 静态目录 / 端口）
```

一句话判据：**给纯白图也能吐出一张像样表格** = vision 编码器坏了（不是 prompt 问题）。

---

## 1. 视觉链路：INT8 量化导致精度崩塌（本项目最大的坑）

**现象**

| 项 | 表现 |
|---|---|
| 文本 | 正常（"1+1" → "2"） |
| 图像 | 输出内容与图无关；给纯白图也编表格；同一图多次请求结果不同（temperature=0） |
| 板端对拍 | vision axmodel 输出 vs ONNX 参考 cosine **0.311**（应 ≥0.99） |

**原因**：vision 是 27 层深度塔，INT8 激活的逐层误差连乘。2 层探针就掉到 0.8606，27 层连乘到 0.311。
不是输入归一化、不是 ONNX 导出、不是 Pulsar2 版本、也不是板端运行时太老（官方 V4.6 包在同一块板上看图正常）。

**处理**：vision 编译改用 **U16** + smooth_quant + FP32 conv_bias + disable_auto_refine_scale
（`scripts/build_vision.py` 已内置官方配置），并用 **64 张真实切片**校准。
2 层探针实测 0.8606 → **0.997895**。

**验证**：别只看端到端输出，按 `docs/accuracy.md` §2 用 `ax_run_model` 单独对拍 vision 输出。

---

## 2. 上下文：prefill 桶太小 → `input_num_token > prefill_max_token_num`

**现象**

```
input_num_token(276) > prefill_max_token_num(128)
```
请求直接失败；纯文本短句却正常（因为文本 prompt 短）。

**原因**：首版 `llm_build2` 只生成 64/128 的 prefill 桶，而**单张 448 图就有 256 image token**，
加上 prompt 一定超过 128。

**处理**：用 prefill 桶 1152 重建：

```bash
PREFILL_LEN=1152 PREFILL_STEP=128 MAX_CONTEXT=2048 bash scripts/build_llm.sh
# 板端重启后日志应显示 prefill_max_token_num: 1152
grep prefill_max_token_num /root/minicpm-ocr-v2/serve.log
```

**注意**：`config.json` 里没有 prefill 字段，桶大小由**编译产物**决定，`template_filename_axmodel`
里的 `p128` 只是桶步长。换桶必须重新编译 + 重推模型包。

---

## 3. axllm 启动与进程管理

### 3.1 启动即退：缺 `LD_LIBRARY_PATH`

**现象**：`./bin/axllm serve .` 立刻退出/无日志。

**处理**：

```bash
cd /root/minicpm-ocr-v2
setsid env LD_LIBRARY_PATH=/opt/lib:/opt/bin:$PWD/bin ./bin/axllm serve . --port 18081 \
  > serve.log 2>&1 < /dev/null &
```

`service/start.sh` 已经把这个路径拼好。

### 3.2 `pkill -f 'axllm serve'` 会"自杀"

**现象**：远程执行「杀掉旧 axllm 再重启」的一行命令时，SSH 会话直接断开/命令没跑完，
旧进程杀了但新进程没起来。

**原因**：`pkill -f` 匹配**完整命令行**。你通过 SSH 执行的远端 shell 命令行里本身就含有
`axllm serve` 这个字符串，于是 `pkill` 把当前 shell（及整条 SSH 命令）一起杀了。

**处理**（`scripts/` 与 `service/start.sh` 都按这个来）：

```bash
pkill -x axllm      # 精确匹配进程名，不匹配命令行
# 或：用 pidfile —— kill "$(cat logs/axllm.pid)"
```

同理，别用 `pkill -f board_app` / `pkill -f flask` 在 SSH 一行命令里重启服务。

### 3.3 只能跑一个实例（NPU/CMM 独占）

**现象**：新实例启动后推理报错，或加载到一半失败。

**处理**：先确认旧进程死了再起：

```bash
pgrep -x axllm && pkill -x axllm && sleep 3
cat /proc/ax_proc/mem_cmm_info   # 看 CMM 余量
```

CMM 分区 4096 MB，单包占 ≈900 MB（首版）。U16 vision 包更大，加载前先看余量。

### 3.4 服务起来了但端口不通

```bash
curl -s http://127.0.0.1:18081/v1/models   # 板内自测
ss -lntp | grep -E '18081|8080'            # 端口占用
tail -50 /root/minicpm-ocr-v2/serve.log
```

确认 axllm 监听的是 `0.0.0.0`（日志 `OpenAI API Server starting on 0.0.0.0:18081`），
再从局域网访问 `http://<board-ip>:18081`。

---

## 4. 应用侧：切片 / 整页模式

### 4.1 整页结果错位、表格对不上

**原因分三种**：

1. 切片方案与官方不一致（token 数对不上，prompt 占位符数量错）→ 用 `service/slicing.py`，
   不要自己写 resize/grid 逻辑；它的切片数/尺寸/patch 网格/token 数与官方 processor **逐片一致**
   （真实页 9 片 / 2276 token）。
2. 单块送板前没有 `letterbox_448`（板端 vision 固定 448×448 输入）→ `board_app.letterbox_448`。
3. 整页 9 片 ≈2276 image token 超过 `max_context=2048` → 必须限片数；
   实际使用建议按版面切块，一页 ≤7 片（`TODO` 待定量验证）。

### 4.2 空白片导致输出重复/退化

`board_app.py` 用 `ink_ratio < 0.003` 跳过空白片。如果出现整段重复，先确认空白片真的被跳过了
（`metrics.slice_info` 里 `skipped: true`）。

### 4.3 `slice_times` 显示差 1000 倍

前端按**毫秒**解释 `metrics.slice_times`，而 `board_app.py` 目前填的是**秒**（每片 `elapsed_s`）。
已知不一致：改 `board_app.py` 里 `slice_times.append(r["elapsed_s"] * 1000)`，或前端读取
`slice_times_unit` 字段。

---

## 5. 转换/编译阶段

### 5.1 `llm_build2` 报 `'Config' object has no attribute 'linear_num_key_heads'`

**原因**：权重键没改名（`model.language_model.*` 还是原样），builder 取不到张量。
**处理**：`python3 scripts/prep_text_model.py --src origin --out origin_text`。

### 5.2 `llm_build2` 报 `get_norm_attrs AssertionError`

**原因**：改了键名但没重标 `config.json`（`model_type=qwen3_5` / `text_config.model_type=qwen3_5_text`）。
**处理**：同上，脚本会一起写 config。

### 5.3 vision ONNX 导出/quantize 各种错

| 现象 | 原因 | 处理 |
|---|---|---|
| ORT 喂 `target_sizes` 报 `Invalid input name` | `target_sizes` 被常量折叠进图里 | 图只有 `pixel_values` 一个输入；换分辨率重导 |
| `Type parameter (T) of Optype (Concat) bound to different types` | `target_sizes` int32 与常量 int64 冲突 | merger 侧用 int64 |
| NPU 编译器不接受 `Unsqueezeimage_embeds_dim_1` | `unsqueeze(0)` 留了符号维 | 改静态 `reshape(1,256,1024)` |
| `expected scalar type Float but found BFloat16` | config dtype=bfloat16，processor 给 float32 | 显式 `dtype=torch.float32` 加载 |

### 5.4 tokenizer.txt 只有 6.4 MB（应为 12.1 MB）

**原因**：宿主机 transformers 5.20 的 BPE 导出分支与 5.22 不同。
**处理**：用 transformers 5.22 重新生成（12,102,644 B），并跑官方 `tokenizer.axera/build/test_tokenizer -m 28`（5/5 PASS）。
转换机与导出机版本要一致。

### 5.5 整模型 ONNX 走不通（不是 bug，是设计约束）

5 条路径的原始报错（详见 `docs/conversion.md` §8）：

| 路径 | 报错 |
|---|---|
| `export(dynamo=False)` | `unsupported type: DynamicCache` |
| wrapper + `dynamo=False` | `aten::diff` not supported in opset 18 |
| `export(dynamo=True)` | `No module named 'onnxscript'` |
| `torch.export.export` | `GuardOnDataDependentSymNode`（NaViT host int） |
| 仅 LLM backbone | 探针签名冲突 `got multiple values for argument 'use_cache'` |

**处理**：拆图（vision 静态 ONNX + LLM `llm_build2`），不要硬怼单图导出。

---

## 6. 网页测试台

| 现象 | 原因 | 处理 |
|---|---|---|
| 页面没样式 / `styles.css` 404 | Flask 用了默认 `static_url_path="/static"` | `Flask(static_folder=WEB_DIR, static_url_path="")` |
| `/api/*` 404 | 静态目录里有同名 `api/` 抢占，或 Flask 没起 | 静态目录不要有 `api/`；`curl :8080/api/health` |
| 打开页面但提交报错 | axllm 没起 / URL 不对 | `AXLLM_URL` 默认 `http://127.0.0.1:18081` |
| `ModuleNotFoundError: flask/PIL/numpy` | 板端缺依赖 | `pip3 install flask pillow numpy` |
| 图片上传即报错 | 文件过大 / 非图片 | 板端内存紧，建议 Flask 设 `MAX_CONTENT_LENGTH=20MB` |
| 局域网打不开 | axllm/Flask 只监听 127.0.0.1 或防火墙 | `board_app.py` 默认 `0.0.0.0`；检查板端 iptables |

---

## 7. 板卡网络：双网卡地址漂移

**现象**：`192.168.10.144` 时通时不通；板上有两个同网段地址。

**原因**：这块 DEMO 板有两张网卡（`eth1` MAC `00:11:22:33:44:57` 静态 192.168.10.144，
`eth0` MAC `e6:6a:d7:37:24:fb` 曾 DHCP 到 192.168.10.145，link DOWN 但地址残留），
同网段双地址让 ARP 出口不确定。

**处理**：不要手改板端静态 IP。在路由器给 `00:11:22:33:44:57` 做 **DHCP 静态租约**；
需要恢复原 DHCP 配置时：

```bash
cp -a /etc/network/interfaces.bak-dhcp-* /etc/network/interfaces && reboot
```

---

## 8. 传包：板端没有 rsync

**现象**：`rsync: command not found`。

**处理**：`tar over ssh`（实测 1.4 GB / 56 s，2.5 GB / 101 s）：

```bash
tar cf - -C <pkg 父目录> <pkg 名> | ssh root@<board> \
  'tar xf - -C /root/minicpm-ocr-v2 --strip-components=1'
```

---

## 9. 排障 Checklist（照着走一遍）

```
[ ] 板端：pgrep -x axllm 有进程、curl :18081/v1/models 返回 model id
[ ] 日志：prefill_max_token_num ≥ 1000；VisionModule init ok: type=MiniCPMV46VL, tokens_per_block=256
[ ] 纯文本请求：curl /v1/chat/completions 只发 text，能正常回答
[ ] 单块图像：448×448 图 + "Table Recognition:"，输出 OTSL 且与图内容一致
[ ] vision 对拍：ax_run_model 输出 vs ONNX 参考 cosine ≥0.99（见 docs/accuracy.md §2）
[ ] 网页：curl :8080/api/health ok，浏览器能上传并出结果
[ ] CMM：cat /proc/ax_proc/mem_cmm_info 有余量
```

---

## 10. 待补充（TODO）

1. U16 vision 全量包加载后的 `remain_cmm`/启动日志（如果内存不足，需要写 `dynamic_load_enable` 之类的降级说明）。
2. 板端并发请求下的报错形态（未压测）。
3. `mem_guard_*` 配置在**本项目使用的 axllm 二进制版本**里是否支持（ax-llm 新版文档有，官方包版本未核对）。
4. 整页模式实际建议片数（ctx 2048 下 6 片 / 7 片的端到端质量对比）。
