# 板端部署：模型包上传 → axllm 推理服务 → 网页测试台

> 目标板：AX8850（AX650N 板型）aarch64，Ubuntu 22.04，kernel 5.15.73，Python 3.10.12，8 核 / 3.8 GB DDR / CMM 4096 MB / 磁盘 24 GB）
> 板端推理：`axllm serve`（OpenAI 兼容，端口 **18081**）｜网页测试台：Flask（端口 **8080**）
> 本文数字来自 2026-11-10 ~ 2026-11-11 在真实板上的实测记录（最终 U16 包）。

---

## 0. 两条路：只用推理接口 / 带网页测试台

| 场景 | 需要的东西 | 端口 |
|---|---|---|
| 只要 OCR 推理接口 | 模型包（含 `bin/axllm`）+ `scripts/deploy_to_board.sh` | 18081 |
| 要浏览器里点点就用 | 模型包 + `service/`（Flask + `web/` + 切片/OTSL 模块） | 18081 + 8080 |

模型包 ≈2.5 GB，板端磁盘 24 GB（部署后剩 16 GB 左右），够放两个版本做回滚。

---

## 1. 板端前置检查

```bash
# 板端（root）
uname -m                     # 期望 aarch64
cat /etc/os-release          # 期望 Ubuntu 22.04
python3 -V                   # 3.10.12
free -m                      # 3.8 GB DDR
df -h /                      # 24 GB
ls /opt/bin/ax_run_model     # NPU 运行时/驱动随出厂 rootfs 就位
cat /proc/ax_proc/mem_cmm_info   # CMM 分区（4096 MB）
```

网页测试台的 Python 依赖（板端离线，先确认已装）：

```bash
python3 -c "import flask, PIL, numpy; print('ok')"
# 缺的话（有网时）：
pip3 install flask pillow numpy
```

> 板端没有 rsync、没有 cmake；rootfs 精简。**不要指望 rsync/apt 大包**。

---

## 2. 上传模型包：tar over ssh（板端没有 rsync）

实测：1.4 GB / 56 s，**2.5 GB / 101 s**。用仓库里的部署脚本：

```bash
# 转换机 / Mac 上执行
./scripts/deploy_to_board.sh \
  --pkg ./work/minicpm-ocr/axllm_pkg \
  --host root@192.168.10.144 \
  --dst /root/minicpm-ocr-v2 \
  --port 18081
# 板子用密码登录时加 --password '***'（走 SSH_ASKPASS，不落盘）
# 只传不启动：--no-start
```

脚本内部就是实测可用的那条命令：

```bash
tar cf - -C <pkg 的父目录> <pkg 名> \
  | ssh root@<board> 'mkdir -p /root/minicpm-ocr-v2 && tar xf - -C /root/minicpm-ocr-v2 --strip-components=1'
```

手工版（不想用脚本时）：

```bash
cd work/minicpm-ocr
tar cf - axllm_pkg | ssh root@192.168.10.144 \
  'tar xf - -C /root/minicpm-ocr-v2 --strip-components=1'
```

---

## 3. 启动 axllm

```bash
cd /root/minicpm-ocr-v2
chmod +x bin/axllm
setsid env LD_LIBRARY_PATH=/opt/lib:/opt/bin:$PWD/bin \
  ./bin/axllm serve . --port 18081 > serve.log 2>&1 < /dev/null &
```

- **`LD_LIBRARY_PATH=/opt/lib:/opt/bin:$PWD/bin` 不能省**：少了它 `axllm` 启动即退（板端缺依赖库路径）。
- `setsid ... < /dev/null`：脱离 SSH 会话，断线不退出；`serve.log` 里能看到全部加载过程。

### 3.1 启动成功的标志（日志，实测）

```
mixed attention enabled: full_attention_interval=4 ref_full_layer_idx=3
attention config: layers=24 sliding=0 full=6 linear=18
tokenizer_type = 3 / huggingface tokenizer mode = gpt2_byte_bpe   → tokenizer init ok
init 0..23 axmodel ok, remain_cmm(3788 MB → 3154 MB)
init post axmodel ok, remain_cmm(2889 MB)
VisionModule init ok: type=MiniCPMV46VL, tokens_per_block=256, embed_size=1024, out_dtype=fp32
LLM init ok → OpenAI API Server starting on 0.0.0.0:18081
```

> 上面这组是**首版 s8/p64 包**的实测日志（`max_token_len: 2047`、`prefill_token_num: 64`、
> `prefill_max_token_num: 128`）。v2 包（prefill 桶 1152 / ctx 2048）应看到
> **`prefill_max_token_num: 1152`**；`max_token_len` 的确切值以实际日志为准
> 启动日志关键行（v2 + U16 包实测）：
> `max_token_len : 2047`、`current prefill_max_token_num:1152`、
> `VisionModule init ok: type=MiniCPMV46VL, tokens_per_block=256, embed_size=1024, out_dtype=fp32`、
> `init post axmodel ok, remain_cmm(2130 MB)`、`LLM init ok`。
> 版面模型（可选）：`LAYOUT_MODEL=/root/minicpm-ocr-v2/doclayout_bl_u16.axmodel`，`/api/health` 会返回 `layout_model`。

### 3.2 自检

```bash
curl -s http://127.0.0.1:18081/v1/models | head -c 300
# 期望返回 model id: MiniCPM-OCR-1.0-AX650-C2048-P1152
```

磁盘上的模型包长这样：

```
/root/minicpm-ocr-v2/
├── config.json
├── qwen3_5_text_p128_l0..23_together.axmodel
├── qwen3_5_text_post.axmodel
├── model.embed_tokens.weight.bfloat16.bin
├── minicpm_ocr_tokenizer.txt
├── post_config.json
├── minicpm_ocr_vision_448.axmodel
└── bin/axllm  (+ axllm.version.json)
```

加载时间：启动到服务可用 ≈ **3 s**（24 层 + post + vision，全部 mmap 加载）。

---

## 4. CMM 与内存

| 项 | 实测值 |
|---|---|
| 物理内存 | 3.8 GB DDR |
| CMM 分区 | 4096 MB |
| 加载 24 层后 `remain_cmm` | 3788 MB → **3154 MB** |
| 加载 post 后 `remain_cmm` | **2889 MB** |
| 合计占用 | **≈900 MB CMM**（首版 s8/p64 包） |

要点：

1. **只能跑一个 axllm 实例**。NPU + CMM 是独占资源，重启前必须杀干净旧进程（见 `docs/troubleshooting.md` §3）。
2. 新包（U16 vision）体积/内存会比 INT8 大，加载前先 `cat /proc/ax_proc/mem_cmm_info` 看余量。
3. 别在板上同时跑别的 NPU 模型（会抢 CMM，表现为加载到一半失败或推理报错）。

---

## 5. 网页测试台（Flask + 静态前端）

模型包同级的 `service/` 一起推上板即可：

```
/root/minicpm-ocr-v2/service/
├── start.sh           # 一键起/停/状态（axllm 18081 + Flask 8080）
├── board_app.py       # Flask：/api/ocr、/api/chat、/api/health、/api/slices
├── slicing.py         # MiniCPM-V 式切片复刻（纯 numpy+PIL，与官方逐片对齐）
├── otsl_to_md.py      # OTSL → Markdown/HTML（板端渲染表格用）
├── samples/           # 真实模型输出样例（otsl 单元测试用）
└── web/               # 静态前端（M5Stack 风格，零依赖、无构建）
```

启动：

```bash
cd /root/minicpm-ocr-v2
./service/start.sh start     # 先 axllm 再 Flask；已经有实例会复用
./service/start.sh status
./service/start.sh stop
```

或手工分开起：

```bash
# 终端 1：axllm（上面的启动命令）
# 终端 2：Flask
cd /root/minicpm-ocr-v2/service
AXLLM_URL=http://127.0.0.1:18081 PORT=8080 python3 board_app.py
```

浏览器访问 `http://<板子 IP>:8080/`。

接口（详见 `service/web/README.md`）：

| 方法/路径 | 作用 |
|---|---|
| `GET /` | 网页测试台（`web/` 静态目录） |
| `POST /api/ocr` | multipart：`image` + `prompt` + `mode=page\|region` + `max_tokens` |
| `POST /api/chat` | 纯文本：`{"prompt": "...", "max_tokens": 512}` |
| `GET /api/health` | 透传板端 `/v1/models` |
| `GET /api/slices?w=&h=` | 预演切片方案（不推理，排障用） |

板端服务行为要点：

1. **整页模式**：`slicing.py` 复刻官方切片（全局缩略 + 网格切片），逐片 `letterbox_448` 后送 NPU
   （板端 vision encoder 固定 448×448 输入），再按阅读顺序拼接 + OTSL→Markdown。
2. **空白片跳过**：非白像素比例 < 0.003 的切片直接跳过，避免模型对空白图退化重复。
3. `MAX_SLICES_PER_PAGE`（默认 9）控制每页最多几片；整页模式默认跳过全局缩略片（8 片 ≈2016 image token）。
   整页混合版面建议用 `layout` 模式（版面检测后逐区域识别），避免网格切片把表格/多栏切开。
4. Flask 用 `static_folder=web, static_url_path=""`；前端全部走同源相对路径 `/api/*`，不需要 CORS。
5. 前端 `slice_times` 与 `board_app.py` 均已统一为**毫秒**（`metrics.slice_time_unit="ms"`）。

---

## 6. 板卡网络（双网卡，地址会漂）

这块 DEMO 板有**两张网卡**，曾经同时持地址：

| 网卡 | MAC | 地址 |
|---|---|---|
| `eth1` | `00:11:22:33:44:57` | 静态 192.168.10.144/24（改过 `/etc/network/interfaces`） |
| `eth0` | `e6:6a:d7:37:24:fb` | 曾 DHCP 拿到 192.168.10.145，link DOWN 但地址残留 |

教训：**地址漂移的根因是双网卡 + DHCP**。正确做法是在路由器给 `00:11:22:33:44:57` 做 **DHCP 静态租约**，
不要手改板端静态 IP（同网段双地址会让 ARP 出口不确定，表现为 `.144` 时通时不通）。

恢复原 DHCP 配置：

```bash
cp -a /etc/network/interfaces.bak-dhcp-* /etc/network/interfaces && reboot
```

---

## 7. 升级 / 回滚

```bash
# 板上先备份旧包（模型包 2.5 GB，注意磁盘）
mv /root/minicpm-ocr-v2 /root/minicpm-ocr-v2.bak-$(date +%Y%m%d)

# 只换 vision axmodel（调精度时最常用）：
scp minicpm_ocr_vision_448.axmodel root@<board>:/root/minicpm-ocr-v2/vision_new.axmodel
ssh root@<board> 'cd /root/minicpm-ocr-v2 && \
  mv minicpm_ocr_vision_448.axmodel minicpm_ocr_vision_448.axmodel.bak && \
  mv vision_new.axmodel minicpm_ocr_vision_448.axmodel'
# 然后 ./service/start.sh restart
```

---

## 已验证设备矩阵

| 设备 | 芯片 / 板型 | CMM | 运行时 | 网页端口 | 备注 |
|---|---|---|---|---|---|
| 192.168.10.144 | AX8850（AX650N 板型）/ M5Stack 8G | 4096 MB | libax_engine V1.45.0 (2024-09) | 8080 | 首发部署板 |
| 192.168.10.105 | **AX8850C**（AX650C 板型）/ `AX650N_M5stack_8G` | **6144 MB** | libax_engine V3.6.4 (2025-08) | **8088** | 预装 M5Stack ModuleLLM 占用 80/8000/8080；同包直接可跑；**systemd 自启**（ocr-axllm / ocr-web） |

两台设备均验证：vision U16 cosine 0.9953、表格/版面模式正常、流式 SSE 正常。
105 的部署记录见 [deploy-105.md](deploy-105.md)。

## 8. 后续可继续优化（非阻塞）

已完成：v2 启动日志（`max_token_len=2047` / `prefill_max_token_num=1152` / VisionModule 256 token）、
U16 vision 体积 511 MB、加载后 `remain_cmm 2130 MB`、单块 615 ms。

可选增强：
1. 板端开机自启（systemd unit 或 `/etc/rc.local`）——当前用 `service/start.sh` 手工拉起；
2. 板端 Flask 的 `MAX_CONTENT_LENGTH`（建议 20 MB）与上行并发上限；
3. 版面/表格模式在更大页面上的并行化。


## 服务 API 的识别模式

`POST /api/ocr`（multipart）：`image`、`prompt`、`mode`、`max_tokens`

- `mode=auto`（默认）：表格线检测 → 表格模式；竖长大图 → 整页切片；宽条 → 列分块；其余 → 单区域；
- `mode=table`：整表识别（大表按行切分后合并）→ 一个 Markdown 表格；
- `mode=layout`：DocLayout-YOLO 版面检测（U16 axmodel，212 ms）→ 按区域类型识别 → 阅读顺序拼接；
  需要 `LAYOUT_MODEL=/path/doclayout_bl_u16.axmodel` 环境变量，未设置时自动回退；
- `mode=page`：官方 9 片切片逐片识别后拼接；
- `mode=wide`：宽条文字按列分块后顺序拼接；
- `mode=region`：整图单块识别。

表格截图用 `auto`/`table`；`page` 会把表格沿列切开导致乱序（见 [table-mode.md](table-mode.md)）。
