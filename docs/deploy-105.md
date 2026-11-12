# 新设备部署：192.168.10.105（AX650C / M5stack 8G）— 2026-11-11

## 设备信息

| 项 | 值 |
|---|---|
| 芯片 | `AX650C_CHIP`（板型 `AX650N_M5stack_8G`） |
| CMM | 分区 **6144 MB**（`ax_cmm V3.6.4_20250822020158`） |
| Linux 可见内存 | 1.9 GB（其余为 CMM 保留），8 核 aarch64，kernel 5.15.73 |
| 运行时 | `/soc/lib/libax_engine.so`（新于 144 的 V1.45.0） |
| 系统预装 | M5Stack ModuleLLM：`api_server.py`(8000)、`tokenizer_qwen2.5-0.5B`(127.0.0.1:8080)、nginx(80) |

## 兼容性结论

- 为 AX650 编译的 axmodel（vision U16、DocLayout-YOLO U16）在 AX650C 上**可直接运行**：
  版面模型 213 ms（与 144 的 212.6 ms 基本相同）；
- `axllm` 官方 AX8850（AX650 目标）aarch64 二进制可运行，加载后 `remain_cmm(3469 MB)`，
  `VisionModule tokens_per_block=256`、`prefill_max_token_num=1152`、`max_token_len=2047` 均正常。

## 部署步骤（与 144 相同，端口不同）

```bash
# 1) 包（2.6 GB / 36 文件）
scp -r /tmp/hf_pkg root@192.168.10.105:/root/minicpm-ocr-v2      # 或 tar over ssh
# 2) Python 依赖（105 自带 python3.10）
pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple flask pillow numpy
# 3) 服务文件
scp board_app.py slicing.py otsl_to_md.py root@105:/root/minicpm-ocr-v2/webapp/
scp -r web root@105:/root/minicpm-ocr-v2/webapp/
# 4) 启动
sh /root/minicpm-ocr-v2/start_axllm.sh     # 18081
sh /root/minicpm-ocr-v2/start_flask.sh     # 8088（8080 被预装 tokenizer 占用，8000 被 ModuleLLM 占用）
```

- 访问：**http://192.168.10.105:8088/**（axllm 仍为 18081）
- 启动脚本已放在包目录：`start_axllm.sh` / `start_flask.sh`。

## 实测

| 项 | 结果 |
|---|---|
| 文本 "1+1" | 0.62 s → "2" |
| 表格示例（auto→table，233 token） | **23.2 s**，decode 10.1 tok/s，Markdown 表格正确（含 83.17） |
| 版面模式 文档示例（6 区域） | **21.2 s**，按阅读顺序输出 |
| 流式 OCR 首字 | **0.37 s**（预填缓存命中时）；SSE 正常 |
| 版面模型单次 | **213 ms** |
| CMM | 加载后剩余 3469 MB（预装的 ModuleLLM 另占部分 CMM） |
| Linux 内存 | 1.9 GB 总量（CMM 之外），Flask + 页面常驻约 100 MB |

## 注意

1. **端口冲突**：105 预装 M5Stack ModuleLLM 服务占 8000（OpenAI 插件）、127.0.0.1:8080（tokenizer）、80（nginx），
   所以 OCR 网页放在 **8088**；如需 8080 需先停预装服务（不建议）。
2. Linux 可见内存只有 1.9 GB，但 CMM 独立 6 GB，模型加载没问题；不要同时再起其它占 CMM 的大模型。
3. **服务已于 2026-11-12 测试结束后关闭**（用户确认测试完成）：axllm(18081) 与 Flask(8088) 均停止，
   对应的 systemd 自启也已删除。设备上模型包与 `start_axllm.sh` / `start_flask.sh` 保留，
   需要时执行两个脚本即可恢复（axllm 加载约 25 s）。
