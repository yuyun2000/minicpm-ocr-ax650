/*!
 * MiniCPM-OCR 板端测试台 · 前端逻辑
 * ---------------------------------------------------------------------------
 * 零依赖、无构建。所有请求走同源相对路径：/api/ocr、/api/chat、/api/health。
 *
 * 接口约定（与 README 一致，实现时对旧字段做了容错）：
 *   POST /api/ocr   multipart: image, prompt, mode=page|region, max_tokens
 *                   -> { content, markdown?, metrics{...}, error }
 *   POST /api/chat  json: { prompt, max_tokens }
 *                   -> { content, metrics{...}, error }
 *   GET  /api/health-> { ok, board }
 *
 * 模型输出按不可信文本处理：先 HTML 转义，再套最小 Markdown 渲染；
 * OTSL 不在前端解析（转 Markdown 由服务端 markdown 字段负责）。
 */
(function () {
  'use strict';

  var THEME_KEY = 'moc-theme';
  var THEME_LABEL = { auto: '主题：跟随系统', light: '主题：浅色', dark: '主题：深色' };
  var THEME_ORDER = ['auto', 'light', 'dark'];
  var TASK_CUSTOM = '__custom__';
  var DEFAULT_IMAGE_PROMPT = 'Text Recognition:';

  var MODE_HINT = {
    auto: '自动：检测到表格线走表格模式；竖长文档页（h≥1.2w 且长边≥1200，或长边≥2200）走整页切片；宽条走列分块；其余整图单区域。',
    table: '表格模式：整表识别（大表按行切分后合并），输出 Markdown 表格；忽略上方 prompt，固定用 Table Recognition。',
    layout: '版面模式：DocLayout-YOLO 检测标题/正文/表格/公式区域，按类型逐区域识别后按阅读顺序拼接。',
    page: '整页模式：服务端按官方 9 片切片（跳过全局缩略片），逐片识别后拼接，返回每片耗时。',
    region: '单区域模式：整图作为单个区域送入，不切片；适合已裁好的单栏文本或单张表格。'
  };

  // ---------------------------------------------------------------- DOM 引用
  function $(id) { return document.getElementById(id); }

  var el = {
    // 主题 / 健康
    themeToggle: $('theme-toggle'),
    healthTag: $('health-tag'),
    healthText: $('health-text'),
    healthRefresh: $('health-refresh'),
    healthService: $('health-service'),
    healthModel: $('health-model'),
    healthBoard: $('health-board'),
    healthTime: $('health-time'),
    healthHint: $('health-hint'),
    boardMeta: $('board-meta'),

    // 输入
    form: $('input-form'),
    tabImage: $('tab-image'),
    tabText: $('tab-text'),
    paneImage: $('pane-image'),
    paneText: $('pane-text'),
    dropzone: $('dropzone'),
    fileInput: $('file-input'),
    preview: $('preview'),
    previewImg: $('preview-img'),
    previewName: $('preview-name'),
    previewClear: $('preview-clear'),
    taskSelect: $('task-select'),
    customPromptField: $('custom-prompt-field'),
    customPrompt: $('custom-prompt'),
    promptEcho: $('prompt-echo'),
    modeHint: $('mode-hint'),
    maxTokens: $('max-tokens'),
    submitBtn: $('submit-btn'),
    cancelBtn: $('cancel-btn'),
    chatPrompt: $('chat-prompt'),
    chatMaxTokens: $('chat-max-tokens'),
    chatSubmitBtn: $('chat-submit-btn'),
    chatCancelBtn: $('chat-cancel-btn'),

    // 结果
    resultState: $('result-state'),
    copyResult: $('copy-result'),
    resultEmpty: $('result-empty'),
    resultLoading: $('result-loading'),
    loadingText: $('loading-text'),
    resultError: $('result-error'),
    resultErrorText: $('result-error-text'),
    resultContent: $('result-content'),
    markdownOut: $('markdown-out'),
    rawDetails: $('raw-details'),
    rawOut: $('raw-out'),

    // 指标
    kpiTtft: $('kpi-ttft'),
    kpiDecode: $('kpi-decode'),
    kpiElapsed: $('kpi-elapsed'),
    kpiPromptTokens: $('kpi-prompt-tokens'),
    kpiCompletionTokens: $('kpi-completion-tokens'),
    kpiSlices: $('kpi-slices'),
    metricsNote: $('metrics-note'),
    slicesPanel: $('slices-panel'),
    slicesCount: $('slices-count'),
    sliceList: $('slice-list'),
    slicesHelp: $('slices-help')
  };

  // ---------------------------------------------------------------- 运行时状态
  var state = {
    tab: 'image',
    file: null,
    objectUrl: null,
    dims: '',
    loading: false,
    controller: null,
    timer: null,
    startedAt: 0,
    raw: '',
    lastMode: 'page',
    lastKind: 'image'
  };

  var mql = window.matchMedia ? window.matchMedia('(prefers-color-scheme: dark)') : null;

  // ================================================================ 小工具
  function setText(node, text) {
    if (node) node.textContent = text == null ? '' : String(text);
  }

  function on(node, type, handler) {
    if (node) node.addEventListener(type, handler);
  }

  function clampTokens(value, fallback) {
    var n = parseInt(value, 10);
    if (!isFinite(n)) n = fallback;
    return Math.min(4096, Math.max(16, n));
  }

  function formatBytes(n) {
    if (typeof n !== 'number' || !isFinite(n)) return '';
    if (n < 1024) return n + ' B';
    if (n < 1024 * 1024) return (n / 1024).toFixed(1) + ' KB';
    return (n / 1024 / 1024).toFixed(2) + ' MB';
  }

  function formatClock(date) {
    try {
      return date.toLocaleTimeString('zh-CN', { hour12: false });
    } catch (e) {
      return date.toISOString().slice(11, 19);
    }
  }

  function escapeHtml(value) {
    return String(value == null ? '' : value)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  // ================================================================ 主题
  function currentThemePref() {
    try {
      return localStorage.getItem(THEME_KEY) || 'auto';
    } catch (e) {
      return 'auto';
    }
  }

  function applyTheme(pref) {
    if (THEME_ORDER.indexOf(pref) === -1) pref = 'auto';
    var dark = pref === 'dark' || (pref === 'auto' && mql && mql.matches);
    var root = document.documentElement;
    root.setAttribute('data-theme', dark ? 'dark' : 'light');
    root.setAttribute('data-theme-pref', pref);
    if (el.themeToggle) {
      el.themeToggle.textContent = THEME_LABEL[pref];
      el.themeToggle.setAttribute('aria-label', THEME_LABEL[pref]);
    }
    try {
      localStorage.setItem(THEME_KEY, pref);
    } catch (e) { /* 隐私模式下忽略 */ }
  }

  on(el.themeToggle, 'click', function () {
    var pref = currentThemePref();
    var next = THEME_ORDER[(THEME_ORDER.indexOf(pref) + 1) % THEME_ORDER.length];
    applyTheme(next);
  });

  if (mql) {
    var onSchemeChange = function () {
      if (currentThemePref() === 'auto') applyTheme('auto');
    };
    if (mql.addEventListener) mql.addEventListener('change', onSchemeChange);
    else if (mql.addListener) mql.addListener(onSchemeChange);
  }

  // ================================================================ 标签页
  function setTab(tab) {
    state.tab = tab === 'text' ? 'text' : 'image';
    var isImage = state.tab === 'image';
    if (el.tabImage) {
      el.tabImage.classList.toggle('is-active', isImage);
      el.tabImage.setAttribute('aria-selected', String(isImage));
    }
    if (el.tabText) {
      el.tabText.classList.toggle('is-active', !isImage);
      el.tabText.setAttribute('aria-selected', String(!isImage));
    }
    if (el.paneImage) el.paneImage.hidden = !isImage;
    if (el.paneText) el.paneText.hidden = isImage;
  }

  on(el.tabImage, 'click', function () { setTab('image'); });
  on(el.tabText, 'click', function () { setTab('text'); });

  // ================================================================ 图片选择
  function clearFile() {
    state.file = null;
    state.dims = '';
    if (state.objectUrl) {
      URL.revokeObjectURL(state.objectUrl);
      state.objectUrl = null;
    }
    if (el.previewImg) el.previewImg.removeAttribute('src');
    if (el.preview) el.preview.hidden = true;
    if (el.previewName) el.previewName.textContent = '';
    if (el.fileInput) el.fileInput.value = '';
  }

  function setFile(file) {
    if (!file) return;
    if (!/^image\//.test(file.type || '')) {
      showError('只支持图片文件（PNG / JPEG / WebP）。');
      return;
    }
    clearFile();
    state.file = file;
    state.objectUrl = URL.createObjectURL(file);
    if (el.previewImg) {
      el.previewImg.onload = function () {
        state.dims = el.previewImg.naturalWidth + ' × ' + el.previewImg.naturalHeight + ' px';
        setText(el.previewName, describeFile());
      };
      el.previewImg.src = state.objectUrl;
    }
    if (el.preview) el.preview.hidden = false;
    setText(el.previewName, describeFile());
  }

  function describeFile() {
    if (!state.file) return '';
    var parts = [state.file.name, formatBytes(state.file.size)];
    if (state.dims) parts.push(state.dims);
    return parts.join(' · ');
  }

  on(el.fileInput, 'change', function (event) {
    var file = event.target.files && event.target.files[0];
    if (file) setFile(file);
  });

  on(el.dropzone, 'click', function () {
    if (el.fileInput) el.fileInput.click();
  });

  on(el.dropzone, 'keydown', function (event) {
    if (event.key === 'Enter' || event.key === ' ' || event.key === 'Spacebar') {
      event.preventDefault();
      if (el.fileInput) el.fileInput.click();
    }
  });

  ['dragenter', 'dragover'].forEach(function (type) {
    on(el.dropzone, type, function (event) {
      event.preventDefault();
      event.stopPropagation();
      el.dropzone.classList.add('is-over');
    });
  });

  ['dragleave', 'dragend'].forEach(function (type) {
    on(el.dropzone, type, function (event) {
      event.preventDefault();
      event.stopPropagation();
      el.dropzone.classList.remove('is-over');
    });
  });

  on(el.dropzone, 'drop', function (event) {
    event.preventDefault();
    event.stopPropagation();
    el.dropzone.classList.remove('is-over');
    var dt = event.dataTransfer;
    if (!dt) return;
    if (dt.files && dt.files[0]) {
      setFile(dt.files[0]);
      return;
    }
    // 从网页拖来的图片没有 File，只有 URL；不静默失败，提示用户
    var url = dt.getData && dt.getData('text/uri-list');
    if (url) showError('不支持直接拖入网页图片链接，请先把图片保存到本地再拖入。');
  });

  // 阻止浏览器在页面其他位置打开被拖入的文件
  ['dragover', 'drop'].forEach(function (type) {
    document.addEventListener(type, function (event) {
      if (el.dropzone && el.dropzone.contains(event.target)) return;
      event.preventDefault();
    });
  });

  on(el.previewClear, 'click', function () {
    clearFile();
  });

  // Ctrl/Cmd+V 直接粘贴截图
  document.addEventListener('paste', function (event) {
    if (state.tab !== 'image') return;
    var items = event.clipboardData && event.clipboardData.items;
    if (!items) return;
    for (var i = 0; i < items.length; i++) {
      if (items[i].type && items[i].type.indexOf('image/') === 0) {
        var file = items[i].getAsFile();
        if (file) {
          event.preventDefault();
          setFile(file);
        }
        return;
      }
    }
  });

  // ================================================================ 任务类型 / 模式
  function currentPrompt() {
    var value = el.taskSelect ? el.taskSelect.value : DEFAULT_IMAGE_PROMPT;
    if (value === TASK_CUSTOM) {
      return el.customPrompt ? (el.customPrompt.value || '').trim() : '';
    }
    return value;
  }

  function updatePromptEcho() {
    var isCustom = el.taskSelect && el.taskSelect.value === TASK_CUSTOM;
    if (el.customPromptField) el.customPromptField.hidden = !isCustom;
    var text = isCustom ? (el.customPrompt ? (el.customPrompt.value || '').trim() : '') : (el.taskSelect ? el.taskSelect.value : '');
    if (el.promptEcho) {
      el.promptEcho.innerHTML = '将发送：' + (text ? '<code>' + escapeHtml(text) + '</code>' : '<span class="m5-help">（自定义 prompt 为空）</span>');
    }
  }

  on(el.taskSelect, 'change', function () {
    updatePromptEcho();
    if (el.taskSelect.value === TASK_CUSTOM && el.customPrompt) el.customPrompt.focus();
  });

  on(el.customPrompt, 'input', updatePromptEcho);

  function currentMode() {
    if (!el.form) return 'page';
    var checked = el.form.querySelector('input[name="mode"]:checked');
    return checked ? checked.value : 'page';
  }

  function updateModeHint() {
    if (el.modeHint) el.modeHint.textContent = MODE_HINT[currentMode()] || '';
  }

  Array.prototype.forEach.call(document.querySelectorAll('input[name="mode"]'), function (radio) {
    radio.addEventListener('change', updateModeHint);
  });

  // ================================================================ 请求封装
  function request(url, options) {
    var controller = new AbortController();
    state.controller = controller;
    var init = options || {};
    init.signal = controller.signal;
    init.cache = 'no-store';
    init.headers = Object.assign({ Accept: 'application/json' }, init.headers || {});

    return fetch(url, init).then(function (response) {
      var contentType = response.headers.get('content-type') || '';
      if (contentType.indexOf('application/json') === -1) {
        return response.text().then(function (text) {
          var hint = text ? '（响应前 120 字符：' + text.replace(/\s+/g, ' ').slice(0, 120) + '）' : '';
          throw new Error('服务端返回了非 JSON 响应，HTTP ' + response.status + hint);
        });
      }
      return response.json().then(function (data) {
        if (!response.ok) throw new Error((data && data.error) || ('HTTP ' + response.status));
        if (data && data.error) throw new Error(String(data.error));
        return data || {};
      });
    });
  }

  // ================================================================ 结果区状态
  function setResultState(kind, text) {
    if (!el.resultState) return;
    var cls = 'm5-tag';
    if (kind === 'error') cls += ' m5-tag--danger';
    else if (kind !== 'done') cls += ' m5-tag--neutral';
    el.resultState.className = cls;
    el.resultState.textContent = text;
  }

  function beginLoading(kind) {
    state.loading = true;
    state.startedAt = Date.now();
    state.lastKind = kind;
    setResultState('running', kind === 'chat' ? '对话中' : '识别中');
    if (el.resultEmpty) el.resultEmpty.hidden = true;
    if (el.resultError) el.resultError.hidden = true;
    if (el.resultContent) el.resultContent.hidden = true;
    if (el.resultLoading) el.resultLoading.hidden = false;
    if (el.copyResult) el.copyResult.hidden = true;
    if (el.submitBtn) {
      el.submitBtn.classList.add('is-loading');
      el.submitBtn.setAttribute('aria-busy', 'true');
    }
    if (el.chatSubmitBtn) {
      el.chatSubmitBtn.classList.add('is-loading');
      el.chatSubmitBtn.setAttribute('aria-busy', 'true');
    }
    if (el.cancelBtn) el.cancelBtn.hidden = kind !== 'image';
    if (el.chatCancelBtn) el.chatCancelBtn.hidden = kind !== 'chat';
    setText(el.loadingText, '正在请求板端…');
    paintMetrics(null);
    paintSlices(null, state.lastMode);

    if (state.timer) window.clearInterval(state.timer);
    state.timer = window.setInterval(function () {
      var secs = ((Date.now() - state.startedAt) / 1000).toFixed(1);
      setText(el.loadingText, (kind === 'chat' ? '对话中' : '识别中') + '… ' + secs + ' s');
    }, 100);
  }

  function endLoading() {
    state.loading = false;
    state.controller = null;
    if (state.timer) {
      window.clearInterval(state.timer);
      state.timer = null;
    }
    if (el.resultLoading) el.resultLoading.hidden = true;
    if (el.submitBtn) {
      el.submitBtn.classList.remove('is-loading');
      el.submitBtn.removeAttribute('aria-busy');
    }
    if (el.chatSubmitBtn) {
      el.chatSubmitBtn.classList.remove('is-loading');
      el.chatSubmitBtn.removeAttribute('aria-busy');
    }
    if (el.cancelBtn) el.cancelBtn.hidden = true;
    if (el.chatCancelBtn) el.chatCancelBtn.hidden = true;
  }

  function showError(message) {
    endLoading();
    setResultState('error', '失败');
    if (el.resultEmpty) el.resultEmpty.hidden = true;
    if (el.resultLoading) el.resultLoading.hidden = true;
    if (el.resultContent) el.resultContent.hidden = true;
    if (el.resultError) el.resultError.hidden = false;
    setText(el.resultErrorText, message);
    setText(el.metricsNote, '本次请求失败，没有指标');
    paintMetrics(null);
    paintSlices(null, null);
  }

  function handleFailure(error, kind) {
    if (error && error.name === 'AbortError') {
      endLoading();
      setResultState('idle', '已取消');
      if (el.resultError) el.resultError.hidden = false;
      setText(el.resultErrorText, '请求已取消（连接已断开）。');
      return;
    }
    showError((kind === 'chat' ? '文本对话失败：' : '图片识别失败：') + (error && error.message ? error.message : String(error)));
  }

  // ================================================================ 指标
  function num(value) {
    return (typeof value === 'number' && isFinite(value)) ? value : null;
  }

  function normalizeMetrics(raw) {
    var m = raw || {};
    var usage = m.usage || {};
    return {
      ttft_ms: num(m.ttft_ms),
      decode_tok_s: num(m.decode_tok_s),
      elapsed_s: num(m.elapsed_s),
      prompt_tokens: num(m.prompt_tokens) !== null ? num(m.prompt_tokens) : num(usage.prompt_tokens),
      completion_tokens: num(m.completion_tokens) !== null ? num(m.completion_tokens) : num(usage.completion_tokens),
      slices: num(m.slices),
      slice_times: Array.isArray(m.slice_times) ? m.slice_times.filter(function (v) { return typeof v === 'number' && isFinite(v); }) : []
    };
  }

  function setKpi(node, value, suffix) {
    setText(node, value === null ? '—' : String(value) + (suffix || ''));
  }

  function paintMetrics(rawMetrics) {
    if (!rawMetrics) {
      setKpi(el.kpiTtft, null);
      setKpi(el.kpiDecode, null);
      setKpi(el.kpiElapsed, null);
      setKpi(el.kpiPromptTokens, null);
      setKpi(el.kpiCompletionTokens, null);
      setKpi(el.kpiSlices, null);
      if (el.metricsNote) el.metricsNote.textContent = '等待请求';
      return null;
    }
    var m = normalizeMetrics(rawMetrics);
    setKpi(el.kpiTtft, m.ttft_ms === null ? null : Math.round(m.ttft_ms).toLocaleString('en-US'));
    setKpi(el.kpiDecode, m.decode_tok_s === null ? null : m.decode_tok_s.toFixed(1));
    setKpi(el.kpiElapsed, m.elapsed_s === null ? null : m.elapsed_s.toFixed(2), ' s');
    setKpi(el.kpiPromptTokens, m.prompt_tokens === null ? null : Math.round(m.prompt_tokens).toLocaleString('en-US'));
    setKpi(el.kpiCompletionTokens, m.completion_tokens === null ? null : Math.round(m.completion_tokens).toLocaleString('en-US'));
    setKpi(el.kpiSlices, m.slices === null ? null : String(m.slices));
    return m;
  }

  function sliceLabel(value) {
    // 按 README 约定：slice_times 单位为毫秒
    return Math.round(value).toLocaleString('en-US') + ' ms';
  }

  function paintSlices(rawMetrics, mode) {
    if (!el.slicesPanel) return;
    if (!rawMetrics) {
      el.slicesPanel.hidden = true;
      return;
    }
    var m = normalizeMetrics(rawMetrics);
    var times = m.slice_times;
    var count = m.slices;
    var visible = (count !== null && count > 0) || times.length > 0;
    if (!visible || mode === 'page' || mode === 'chat') {
      el.slicesPanel.hidden = true;
      return;
    }
    el.slicesPanel.hidden = false;
    var total = count !== null ? count : times.length;
    setText(el.slicesCount, total + ' 片');
    if (el.sliceList) el.sliceList.textContent = '';

    if (times.length) {
      times.forEach(function (value, index) {
        var chip = document.createElement('span');
        chip.className = 'slice-chip';
        var idx = document.createElement('span');
        idx.className = 'slice-chip__idx';
        idx.textContent = '片 ' + (index + 1) + ' ·';
        var val = document.createElement('span');
        val.className = 'slice-chip__val';
        val.textContent = sliceLabel(value);
        chip.appendChild(idx);
        chip.appendChild(val);
        el.sliceList.appendChild(chip);
      });
      var sum = times.reduce(function (a, b) { return a + b; }, 0);
      var max = Math.max.apply(null, times);
      setText(el.slicesHelp, '共 ' + times.length + ' 片 · 切片合计 ' + sliceLabel(sum) + ' · 最慢一片 ' +
        sliceLabel(max) + '（单位按接口约定为 ms，顺序与 slice_times 一致）');
    } else if (el.sliceList) {
      var note = document.createElement('span');
      note.className = 'm5-caption';
      note.textContent = '服务端只返回了片数，slice_times 为空，没有每片耗时。';
      el.sliceList.appendChild(note);
      setText(el.slicesHelp, '片数来自 metrics.slices。');
    }
  }

  // ================================================================ Markdown 渲染（最小实现）
  function renderInline(text) {
    var s = escapeHtml(text);
    s = s.replace(/`([^`]+)`/g, '<code>$1</code>');
    s = s.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    s = s.replace(/(^|[^*])\*([^*\n]+)\*/g, '$1<em>$2</em>');
    s = s.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, function (_, label, href) {
      return '<a href="' + href + '" rel="noopener noreferrer" target="_blank">' + label + '</a>';
    });
    return s;
  }

  function splitTableRow(line) {
    var s = String(line).trim();
    if (s.charAt(0) === '|') s = s.slice(1);
    if (s.charAt(s.length - 1) === '|') s = s.slice(0, -1);
    return s.split('|').map(function (cell) { return cell.trim(); });
  }

  function isTableSeparator(line) {
    var t = String(line || '').trim();
    if (t.indexOf('|') === -1 || t.indexOf('-') === -1) return false;
    if (!/^[|\s:-]+$/.test(t)) return false;
    return t.split('|').filter(function (c) { return c.trim() !== ''; }).length >= 1;
  }

  function tableAligns(separatorLine) {
    return splitTableRow(separatorLine).map(function (cell) {
      var left = cell.charAt(0) === ':';
      var right = cell.charAt(cell.length - 1) === ':';
      if (left && right) return 'center';
      if (right) return 'right';
      if (left) return 'left';
      return '';
    });
  }

  function cellClass(align) {
    if (align === 'right') return ' class="is-num"';
    if (align === 'center') return ' class="is-center"';
    return '';
  }

  function renderTable(headerLine, separatorLine, bodyLines) {
    var aligns = tableAligns(separatorLine);
    var headCells = splitTableRow(headerLine);
    var out = ['<div class="m5-table-wrap"><table class="m5-table"><thead><tr>'];
    headCells.forEach(function (cell, i) {
      out.push('<th' + cellClass(aligns[i]) + '>' + renderInline(cell) + '</th>');
    });
    out.push('</tr></thead><tbody>');
    bodyLines.forEach(function (row) {
      out.push('<tr>');
      splitTableRow(row).forEach(function (cell, i) {
        out.push('<td' + cellClass(aligns[i]) + '>' + renderInline(cell) + '</td>');
      });
      out.push('</tr>');
    });
    out.push('</tbody></table></div>');
    return out.join('');
  }

  function renderMarkdown(source) {
    var text = String(source == null ? '' : source).replace(/\r\n?/g, '\n').replace(/\t/g, '    ');
    var lines = text.split('\n');
    var out = [];
    var i = 0;

    while (i < lines.length) {
      var line = lines[i];
      if (!line.trim()) { i++; continue; }

      // 围栏代码块
      if (/^\s*```/.test(line)) {
        i++;
        var code = [];
        while (i < lines.length && !/^\s*```\s*$/.test(lines[i])) {
          code.push(lines[i]);
          i++;
        }
        if (i < lines.length) i++;
        out.push('<pre class="code-block"><code>' + escapeHtml(code.join('\n')) + '</code></pre>');
        continue;
      }

      // 标题（# → h3，避免与页面 h1/h2 抢层级）
      var heading = line.match(/^\s{0,3}(#{1,6})\s+(.*)$/);
      if (heading) {
        var level = Math.min(heading[1].length + 2, 6);
        out.push('<h' + level + '>' + renderInline(heading[2].replace(/\s+#+\s*$/, '')) + '</h' + level + '>');
        i++;
        continue;
      }

      // 表格：当前行含 | 且下一行是分隔行
      if (line.indexOf('|') !== -1 && i + 1 < lines.length && isTableSeparator(lines[i + 1])) {
        var headerLine = line;
        var separatorLine = lines[i + 1];
        var bodyLines = [];
        i += 2;
        while (i < lines.length && lines[i].trim() && lines[i].indexOf('|') !== -1) {
          bodyLines.push(lines[i]);
          i++;
        }
        out.push(renderTable(headerLine, separatorLine, bodyLines));
        continue;
      }

      // 分隔线
      if (/^\s*([-*_])(\s*\1){2,}\s*$/.test(line)) {
        out.push('<hr>');
        i++;
        continue;
      }

      // 引用
      if (/^\s*>\s?/.test(line)) {
        var quote = [];
        while (i < lines.length && /^\s*>\s?/.test(lines[i])) {
          quote.push(lines[i].replace(/^\s*>\s?/, ''));
          i++;
        }
        out.push('<blockquote>' + quote.map(renderInline).join('<br>') + '</blockquote>');
        continue;
      }

      // 无序列表
      if (/^\s*[-*+]\s+/.test(line)) {
        var items = [];
        while (i < lines.length && /^\s*[-*+]\s+/.test(lines[i])) {
          items.push(lines[i].replace(/^\s*[-*+]\s+/, ''));
          i++;
        }
        out.push('<ul>' + items.map(function (item) { return '<li>' + renderInline(item) + '</li>'; }).join('') + '</ul>');
        continue;
      }

      // 有序列表
      if (/^\s*\d+[.)]\s+/.test(line)) {
        var ordered = [];
        while (i < lines.length && /^\s*\d+[.)]\s+/.test(lines[i])) {
          ordered.push(lines[i].replace(/^\s*\d+[.)]\s+/, ''));
          i++;
        }
        out.push('<ol>' + ordered.map(function (item) { return '<li>' + renderInline(item) + '</li>'; }).join('') + '</ol>');
        continue;
      }

      // 段落：保留行内换行（OCR 原文逐行更接近原貌）
      var para = [line];
      i++;
      while (
        i < lines.length && lines[i].trim() &&
        !/^\s*```/.test(lines[i]) &&
        !/^\s{0,3}#{1,6}\s+/.test(lines[i]) &&
        !(lines[i].indexOf('|') !== -1 && i + 1 < lines.length && isTableSeparator(lines[i + 1]))
      ) {
        para.push(lines[i]);
        i++;
      }
      out.push('<p>' + para.map(renderInline).join('<br>') + '</p>');
    }

    return out.join('\n');
  }

  function isJsonText(value) {
    var t = String(value || '').trim();
    if (!t) return false;
    if (t.charAt(0) !== '{' && t.charAt(0) !== '[') return false;
    try {
      JSON.parse(t);
      return true;
    } catch (e) {
      return false;
    }
  }

  function prettyJson(value) {
    try {
      return JSON.stringify(JSON.parse(String(value).trim()), null, 2);
    } catch (e) {
      return String(value);
    }
  }

  // ================================================================ 渲染结果
  function renderResult(data, kind, mode) {
    var raw = data && data.content != null ? String(data.content) : '';
    var markdown = (data && typeof data.markdown === 'string') ? data.markdown.trim() : '';
    state.raw = raw;

    setText(el.rawOut, raw);
    if (el.rawDetails) el.rawDetails.open = false;

    var html;
    if (!raw && !markdown) {
      html = '<div class="m5-alert m5-alert--danger">板端返回了空结果：content 与 markdown 都为空。</div>';
    } else if (markdown) {
      html = renderMarkdown(markdown);
    } else if (/<otsl>/i.test(raw)) {
      // OTSL 的 Markdown 化由服务端负责，前端只提示并保留原文，避免两套解析口径
      html = '<div class="m5-alert">服务端没有返回 <code>markdown</code> 字段，原始 OTSL 输出已保留在下方「原始输出」中。表格渲染依赖板端的 OTSL → Markdown 后处理。</div>';
    } else if (isJsonText(raw)) {
      html = '<pre class="code-block"><code>' + escapeHtml(prettyJson(raw)) + '</code></pre>';
    } else {
      html = renderMarkdown(raw);
    }

    if (el.markdownOut) el.markdownOut.innerHTML = html;
    if (el.resultEmpty) el.resultEmpty.hidden = true;
    if (el.resultLoading) el.resultLoading.hidden = true;
    if (el.resultError) el.resultError.hidden = true;
    if (el.resultContent) el.resultContent.hidden = false;
    if (el.copyResult) {
      el.copyResult.hidden = false;
      el.copyResult.textContent = '复制原始输出';
    }

    var resolved = (data && data.metrics && data.metrics.mode) || mode;
    var label = kind === 'chat' ? '完成 · 文本对话' : (resolved === 'table' ? '完成 · 表格' : (resolved === 'page' ? '完成 · 整页' : '完成 · 单区域'));
    setResultState('done', label);

    var metrics = paintMetrics(data && data.metrics ? data.metrics : null);
    paintSlices(data && data.metrics ? data.metrics : null, mode);

    if (el.metricsNote) {
      var missing = [];
      if (!metrics) {
        el.metricsNote.textContent = '接口未返回 metrics';
      } else {
        if (metrics.ttft_ms === null) missing.push('ttft_ms');
        if (metrics.decode_tok_s === null) missing.push('decode_tok_s');
        if (metrics.elapsed_s === null) missing.push('elapsed_s');
        if (metrics.prompt_tokens === null) missing.push('prompt_tokens');
        if (metrics.completion_tokens === null) missing.push('completion_tokens');
        el.metricsNote.textContent = (kind === 'chat' ? '文本对话' : '图片识别') +
          ' · ' + formatClock(new Date()) +
          (missing.length ? ' · 缺 ' + missing.join(' / ') : '');
      }
    }
  }

  // ================================================================ 提交
  function submitImage() {
    if (state.loading) return;
    if (!state.file) {
      showError('请先选择一张图片：拖拽、点击上传区，或直接粘贴截图。');
      if (el.dropzone) el.dropzone.focus();
      return;
    }
    var prompt = currentPrompt();
    if (!prompt) {
      showError('自定义 prompt 为空，请填写内容或换回预设任务类型。');
      if (el.customPrompt) el.customPrompt.focus();
      return;
    }
    var mode = currentMode();
    state.lastMode = mode;

    var form = new FormData();
    form.append('image', state.file, state.file.name || 'image.png');
    form.append('prompt', prompt);
    form.append('mode', mode);
    form.append('max_tokens', String(clampTokens(el.maxTokens && el.maxTokens.value, 1024)));

    beginLoading('image');
    request('/api/ocr', { method: 'POST', body: form })
      .then(function (data) {
        endLoading();
        renderResult(data, 'image', mode);
      })
      .catch(function (error) {
        endLoading();
        handleFailure(error, 'image');
      });
  }

  function submitChat() {
    if (state.loading) return;
    var prompt = el.chatPrompt ? (el.chatPrompt.value || '').trim() : '';
    if (!prompt) {
      showError('请先输入对话内容。');
      if (el.chatPrompt) el.chatPrompt.focus();
      return;
    }
    state.lastMode = 'chat';
    beginLoading('chat');
    request('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        prompt: prompt,
        max_tokens: clampTokens(el.chatMaxTokens && el.chatMaxTokens.value, 512)
      })
    })
      .then(function (data) {
        endLoading();
        renderResult(data, 'chat', 'chat');
      })
      .catch(function (error) {
        endLoading();
        handleFailure(error, 'chat');
      });
  }

  on(el.form, 'submit', function (event) {
    event.preventDefault();
    if (state.tab === 'image') submitImage();
    else submitChat();
  });

  function cancelRequest() {
    if (state.controller) {
      try { state.controller.abort(); } catch (e) { /* 忽略 */ }
    }
  }

  on(el.cancelBtn, 'click', cancelRequest);
  on(el.chatCancelBtn, 'click', cancelRequest);

  on(el.chatPrompt, 'keydown', function (event) {
    if ((event.metaKey || event.ctrlKey) && event.key === 'Enter') {
      event.preventDefault();
      submitChat();
    }
  });

  // ================================================================ 复制
  function legacyCopy(text) {
    try {
      var area = document.createElement('textarea');
      area.value = text;
      area.setAttribute('readonly', '');
      area.className = 'visually-hidden';
      document.body.appendChild(area);
      area.select();
      var ok = document.execCommand('copy');
      document.body.removeChild(area);
      return ok;
    } catch (e) {
      return false;
    }
  }

  function flashCopy(message) {
    if (!el.copyResult) return;
    el.copyResult.textContent = message;
    window.setTimeout(function () {
      if (el.copyResult) el.copyResult.textContent = '复制原始输出';
    }, 2000);
  }

  on(el.copyResult, 'click', function () {
    var text = state.raw || '';
    if (!text) {
      flashCopy('没有可复制的内容');
      return;
    }
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(
        function () { flashCopy('已复制'); },
        function () { flashCopy(legacyCopy(text) ? '已复制' : '复制失败，请手动选择'); }
      );
    } else {
      flashCopy(legacyCopy(text) ? '已复制' : '复制失败，请手动选择');
    }
  });

  // ================================================================ 健康检查
  function setHealthTag(kind, text) {
    if (!el.healthTag) return;
    var cls = 'm5-tag';
    if (kind === 'ok') cls += '';
    else if (kind === 'error') cls += ' m5-tag--danger';
    else cls += ' m5-tag--neutral';
    el.healthTag.className = cls;
    setText(el.healthText, text);
  }

  function boardSummary(board) {
    if (!board) return '—';
    if (typeof board === 'string') return board;
    if (Array.isArray(board.data)) {
      var ids = board.data.map(function (item) { return item && (item.id || item.model); }).filter(Boolean);
      return ids.length ? ids.join(', ') : '（/v1/models 未返回模型 id）';
    }
    if (board.model) return String(board.model);
    try {
      return JSON.stringify(board).slice(0, 120);
    } catch (e) {
      return '—';
    }
  }

  function boardUrl(board) {
    if (!board) return '—';
    if (typeof board === 'string') return board;
    var url = board.url || board.base_url || board.endpoint;
    return url ? String(url) : '—';
  }

  function paintBoardMeta(ok, board) {
    if (!el.boardMeta) return;
    el.boardMeta.textContent = '';
    var tag = document.createElement('span');
    tag.className = 'm5-tag' + (ok ? '' : ' m5-tag--danger');
    var dot = document.createElement('i');
    dot.className = 'm5-dot';
    dot.setAttribute('aria-hidden', 'true');
    var text = document.createElement('span');
    text.textContent = ok ? ('板端在线' + (boardSummary(board) !== '—' ? ' · ' + boardSummary(board) : '')) : '板端离线';
    tag.appendChild(dot);
    tag.appendChild(text);
    el.boardMeta.appendChild(tag);
  }

  function checkHealth() {
    setHealthTag('busy', '检测中');
    setText(el.healthService, '检测中…');
    if (el.healthHint) {
      el.healthHint.innerHTML = 'GET <code>/api/health</code> 期望返回 <code>{"ok": true, "board": {...}}</code>。';
    }

    fetch('/api/health', { headers: { Accept: 'application/json' }, cache: 'no-store' })
      .then(function (response) {
        return response.text().then(function (text) {
          var data = null;
          try { data = text ? JSON.parse(text) : null; } catch (e) { data = null; }
          if (data === null) throw new Error('返回不是 JSON（HTTP ' + response.status + '）');
          if (!response.ok) throw new Error(data.error || ('HTTP ' + response.status));
          return data;
        });
      })
      .then(function (data) {
        var ok = data.ok !== false;
        var board = data.board || null;
        setText(el.healthTime, formatClock(new Date()));
        if (ok) {
          setHealthTag('ok', '在线');
          setText(el.healthService, '正常');
          setText(el.healthModel, boardSummary(board));
          setText(el.healthBoard, boardUrl(board));
        } else {
          setHealthTag('error', '离线');
          setText(el.healthService, '异常：' + (data.error ? data.error : 'ok=false'));
          setText(el.healthModel, '—');
          setText(el.healthBoard, boardUrl(board));
        }
        paintBoardMeta(ok, board);
      })
      .catch(function (error) {
        setHealthTag('error', '离线');
        setText(el.healthService, '连接失败');
        setText(el.healthModel, '—');
        setText(el.healthBoard, '—');
        setText(el.healthTime, formatClock(new Date()));
        if (el.healthHint) {
          el.healthHint.textContent = 'GET /api/health 失败：' + (error && error.message ? error.message : String(error)) +
            '。请确认页面由板端 Flask 提供（同源），或网络可达。';
        }
        paintBoardMeta(false, null);
      });
  }

  on(el.healthRefresh, 'click', checkHealth);

  // ================================================================ 初始化
  function init() {
    applyTheme(currentThemePref());
    updatePromptEcho();
    updateModeHint();
    paintMetrics(null);
    paintSlices(null, null);
    setTab('image');
    checkHealth();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

  window.addEventListener('beforeunload', function () {
    if (state.objectUrl) URL.revokeObjectURL(state.objectUrl);
  });

  /* 调试钩子：只暴露纯函数，不含运行时状态。
     用途：在板端用浏览器控制台快速核对渲染与指标口径，例如
       __MOC__.renderMarkdown('| a | b |\n| --- | --- |\n| 1 | 2 |') */
  window.__MOC__ = {
    version: '1.0.0',
    renderMarkdown: renderMarkdown,
    normalizeMetrics: normalizeMetrics,
    escapeHtml: escapeHtml
  };
})();
