(() => {
  const statuses = { queued: "排队中", running: "生成中", done: "已完成", error: "失败" };
  const stages = {
    queued: "等待执行",
    encoding: "编码输入",
    loading: "加载模型",
    warmup: "模型预热",
    denoise: "去噪中",
    decode: "解码视频",
    done: "生成完成",
    error: "生成失败",
  };
  const samplingLabels = { accelerated: "加速采样", standard: "标准采样", custom: "自定义模型" };

  function taskMetadata(task) {
    const details = [];
    const positive = (value) => Number.isFinite(value) && value > 0;
    if (positive(task.width) && positive(task.height)) details.push(`${task.width} × ${task.height}`);
    else {
      if (positive(task.width)) details.push(`宽 ${task.width}`);
      if (positive(task.height)) details.push(`高 ${task.height}`);
    }
    if (positive(task.actual_duration)) details.push(`实际 ${task.actual_duration.toFixed(3)} 秒`);
    if (positive(task.fps)) details.push(`导出 ${task.fps} FPS`);
    if (positive(task.frames)) details.push(`原生 ${task.frames} 帧`);
    if (task.sampling) details.push(samplingLabels[task.sampling] || `采样 ${task.sampling}`);
    if (positive(task.steps)) details.push(`${task.steps} 步`);
    if (task.tensor_parallel) {
      details.push(`TP ${task.tp_parity || "row_stream"}`
        + (task.offload ? ` · ${task.tp_offload_strategy || "retained"}` : "")
        + (task.tp_merge_device === "npu"
          ? ` · NPU merge ${task.tp_merge_delta_device || "npu"}` : " · CPU merge"));
    }
    return details;
  }

  function statusLabel(task) {
    return statuses[task.status] || "状态未知";
  }

  function hasDenoiseSteps(task) {
    return task.status === "running" && task.stage === "denoise"
      && Number.isInteger(task.denoise_step) && task.denoise_step >= 0
      && Number.isInteger(task.denoise_total) && task.denoise_total > 0;
  }

  function stageLabel(task) {
    if (task.status === "queued") return stages.queued;
    if (task.status === "done") return stages.done;
    if (task.status === "error") return stages.error;
    if (hasDenoiseSteps(task)) return `去噪 ${task.denoise_step}/${task.denoise_total}`;
    return stages[task.stage] || "准备中";
  }

  function progressValue(task) {
    return Number.isFinite(task.progress) ? Math.max(0, Math.min(100, task.progress)) : 0;
  }

  function renderProgress(task, prefix) {
    const stage = stageLabel(task);
    const progress = progressValue(task);
    document.getElementById(`${prefix}-status`).textContent = `${statusLabel(task)} · ${task.mode || ""}`;
    document.getElementById(`${prefix}-stage`).textContent = stage;
    const bar = document.getElementById(`${prefix}-progress-bar`);
    bar.style.width = `${progress}%`;
    bar.setAttribute("aria-valuenow", String(progress));
    bar.setAttribute("aria-valuetext", `${stage}，总体进度 ${progress}%`);
    const details = [`总体进度 ${progress}%`];
    if (task.device) details.push(task.device);
    if (hasDenoiseSteps(task) && Number.isFinite(task.step_seconds) && task.step_seconds >= 0) {
      details.push(`本步 ${task.step_seconds.toFixed(2)} 秒`);
    }
    details.push(...taskMetadata(task));
    if (task.dry_run) details.push("试运行：仅生成占位文件，无法播放视频");
    document.getElementById(`${prefix}-progress-text`).textContent = details.join(" · ");
    document.getElementById(`${prefix}-error`).textContent = task.error || "";
  }

  window.WebUITask = { statusLabel, stageLabel, progressValue, renderProgress, taskMetadata };

  const form = document.getElementById("task-form");
  if (!form) return;
  const mode = document.getElementById("mode");
  const device = document.getElementById("device");
  const offload = document.getElementById("offload");
  const residentBlocks = document.getElementById("npu-resident-blocks");
  const tensorParallel = document.getElementById("tensor-parallel");
  const tpOptions = document.getElementById("tp-options");
  const tpParity = document.getElementById("tp-parity");
  const tpOffloadStrategy = document.getElementById("tp-offload-strategy");
  const tpMergeDevice = document.getElementById("tp-merge-device");
  const tpCache = document.getElementById("tp-cache");
  const tpMergeDelta = document.getElementById("tp-merge-delta");
  const tpDeltaRow = document.getElementById("tp-delta-row");
  const optimizedRow = document.getElementById("optimized-row");
  const optimized = document.getElementById("optimized-cuda");
  const error = document.getElementById("form-error");
  const submit = document.getElementById("submit-button");
  const FORM_STATE_KEY = "vdn-h3-form-state";
  const UPLOAD_DB = "vdn-h3-webui";
  const UPLOAD_STORE = "uploads";
  let persistTimer = null;
  let uploadsRestored = false;
  const taskUI = window.WebUITask;
  let offloadTouched = false;
  let polling = false;
  let hasTaskData = false;

  const sampling = document.getElementById("sampling");
  const checkpoint = document.getElementById("checkpoint");
  const steps = document.getElementById("steps");
  const resolution = document.getElementById("resolution");
  const width = document.getElementById("width");
  const height = document.getElementById("height");
  const duration = document.getElementById("duration");
  const fps = document.getElementById("fps");
  const frames = document.getElementById("frames");
  const presets = {
    accelerated: { checkpoint: "ckpts/stage-dmd-step-250", steps: 8 },
    standard: { checkpoint: "ckpts/stage-b-step-2000", steps: 50 },
  };
  const resolutions = {
    "1920x1088": [1920, 1088],
    "1344x768": [1344, 768], "896x512": [896, 512], "1280x736": [1280, 736],
    "768x1344": [768, 1344], "768x768": [768, 768], "512x288": [512, 288],
  };
  const exportRates = [12, 24, 25, 30, 48, 60];
  let previousSampling = sampling.value;
  let customCheckpoint = checkpoint.value;

  function nativeFramesFor(seconds) {
    if (!Number.isFinite(seconds) || seconds < 1 || seconds > 15) return null;
    return Math.max(22, 5 + 17 * Math.floor((seconds * 24 - 5) / 17 + 0.5));
  }

  // Do not coerce empty fields, non-decimal text, or Infinity into usable values.
  function decimal(input) {
    const text = input.value.trim();
    if (!/^(?:\d+(?:\.\d*)?|\.\d+)$/.test(text)) return NaN;
    const value = Number(text);
    return Number.isFinite(value) ? value : NaN;
  }

  function updateConfig() {
    const w = decimal(width);
    const h = decimal(height);
    const seconds = decimal(duration);
    const rate = decimal(fps);
    const count = decimal(steps);
    const dimensionValid = (value) => Number.isInteger(value) && value >= 256 && value <= 2048 && value % 32 === 0;
    let sizeError = "";
    if (!dimensionValid(w) || !dimensionValid(h)) sizeError = "宽高须为 256–2048 范围内的整数，且为 32 的倍数。";
    else if (w / h < 0.25 || w / h > 4) sizeError = "宽高比须在 1:4 至 4:1 之间。";
    else if (resolution.value !== "custom" && !Object.hasOwn(resolutions, resolution.value)) sizeError = "请选择有效的分辨率方案。";
    const durationValid = Number.isFinite(seconds) && seconds >= 1 && seconds <= 15
      && Math.abs(seconds * 1000 - Math.round(seconds * 1000)) < 1e-7;
    const durationError = durationValid ? "" : "时长须为 1–15 秒的有效数字，最多精确到毫秒。";
    const fpsError = exportRates.includes(rate) ? "" : "请选择有效的导出 FPS。";
    const stepsError = Number.isInteger(count) && count >= 1 && count <= 100 ? "" : "去噪步数须为 1–100 的整数。";
    const samplingError = Object.hasOwn(samplingLabels, sampling.value) ? "" : "请选择有效的采样方案。";
    width.setCustomValidity(sizeError);
    height.setCustomValidity(sizeError);
    duration.setCustomValidity(durationError);
    fps.setCustomValidity(fpsError);
    steps.setCustomValidity(stepsError);
    sampling.setCustomValidity(samplingError);

    if (!sizeError) form.style.setProperty("--preview-aspect", `${w} / ${h}`);
    else form.style.removeProperty("--preview-aspect");

    // Temporal conditioning always uses the native 24 FPS, never export FPS.
    const nativeFrames = durationValid ? nativeFramesFor(seconds) : null;
    const actualDuration = nativeFrames === null ? null : nativeFrames / 24;
    frames.value = nativeFrames === null ? "" : String(nativeFrames);

    // Residency is only meaningful on NPU with block offload enabled.
    const resident = decimal(residentBlocks);
    let residentError = "";
    if (!Number.isInteger(resident) || resident < 0 || resident > 49) {
      residentError = "常驻 block 数须为 0–49 的整数。";
    } else if (resident > 0 && (!isNpu() || !offload.checked)) {
      residentError = "常驻 block 仅在 NPU 且开启分块卸载时可用。";
    }
    residentBlocks.setCustomValidity(residentError);

    // TP is experimental. The UI deliberately does not gate frame count or
    // canvas size; oversized runs fail as OOM rather than being blocked here.
    let tpError = "";
    if (tensorParallel.checked) {
      if (!isNpu()) tpError = "实验性 TP 仅支持 NPU。";
      else if (!Number.isInteger(resident) || resident !== 0) tpError = "实验性 TP 要求常驻 block 为 0。";
      else if (nativeFrames === null) tpError = "请先设置有效时长。";

    }
    tensorParallel.setCustomValidity(tpError);

    document.getElementById("duration-hint").textContent = durationError || (
      `原生 ${nativeFrames} 帧 · 实际 ${actualDuration.toFixed(3)} 秒`
      + (fpsError ? ` · ${fpsError}` : ` · 预计输出 ${Math.round(nativeFrames * rate / 24)} 帧。`)
    );
    const message = sizeError || durationError || fpsError || stepsError || samplingError || residentError || tpError;
    document.getElementById("config-summary").textContent = message ? `请检查参数：${message}`
      : `${w} × ${h} · 实际 ${actualDuration.toFixed(3)} 秒 · 导出 ${rate} FPS · ${samplingLabels[sampling.value]} · ${count} 步`
        + (resident > 0 ? ` · 常驻 ${resident} block` : "")
        + (tensorParallel.checked
          ? ` · TP ${tpParity.value}`
            + (offload.checked ? " · CPU 分块卸载" : " · 权重常驻")
            + (tpMergeDevice.value === "npu"
              ? ` · NPU merge ${tpMergeDelta.value}` : " · CPU merge")
          : "");
    return message;
  }

  function updateSampling() {
    if (previousSampling === "custom") customCheckpoint = checkpoint.value;
    const selected = sampling.value;
    const preset = Object.hasOwn(presets, selected) ? presets[selected] : null;
    checkpoint.readOnly = selected !== "custom";
    if (preset) {
      checkpoint.value = preset.checkpoint;
      steps.value = String(preset.steps);
      document.getElementById("sampling-hint").textContent = `${samplingLabels[selected]}使用 ${preset.checkpoint}，推荐 ${preset.steps} 步；步数可手动调整。`;
    } else if (selected === "custom") {
      checkpoint.value = customCheckpoint;
      document.getElementById("advanced-settings").open = true;
      document.getElementById("sampling-hint").textContent = "在高级设置中填写兼容权重路径，并按模型调整步数。加速权重推荐 8 步，标准权重推荐 50 步。";
    }
    previousSampling = selected;
    updateConfig();
  }

  function updateResolution() {
    const custom = resolution.value === "custom";
    width.readOnly = !custom;
    height.readOnly = !custom;
    if (Object.hasOwn(resolutions, resolution.value)) {
      const [w, h] = resolutions[resolution.value];
      width.value = String(w);
      height.value = String(h);
    }
    updateConfig();
  }

  sampling.addEventListener("change", updateSampling);
  resolution.addEventListener("change", updateResolution);
  fps.addEventListener("change", updateConfig);
  for (const input of [width, height, duration, steps, residentBlocks]) {
    input.addEventListener("input", updateConfig);
    input.addEventListener("change", updateConfig);
  }
  updateSampling();
  updateResolution();

  const uploads = ["first", "last", "refs"].map((name) => ({
    name, input: document.getElementById(name),
    preview: document.getElementById(`${name}-preview`),
    clear: document.getElementById(`clear-${name}`), entries: [], files: [],
  }));

  function syncAccumulatedFiles(upload) {
    if (upload.name !== "refs" || typeof DataTransfer === "undefined") return;
    const transfer = new DataTransfer();
    for (const file of upload.files) transfer.items.add(file);
    upload.input.files = transfer.files;
  }

  function releasePreviews(upload) {
    for (const entry of upload.entries) {
      entry.active = false;
      entry.image.removeAttribute("src");
      URL.revokeObjectURL(entry.url);
    }
    upload.entries = [];
    upload.preview.replaceChildren();
  }

  function renderPreviews(upload) {
    releasePreviews(upload);
    const files = upload.files;
    upload.clear.disabled = files.length === 0;
    files.forEach((file, index) => {
      const label = upload.name === "refs" ? `Picture ${index + 1}` : (upload.name === "first" ? "首帧" : "尾帧");
      const figure = document.createElement("figure");
      figure.className = "upload-preview";
      const frame = document.createElement("div");
      frame.className = "preview-image-frame";
      const image = document.createElement("img");
      image.alt = `${label}图片预览：${file.name}`;
      const caption = document.createElement("figcaption");
      const filename = document.createElement("strong");
      filename.textContent = `${label} · ${file.name}`;
      const dimensions = document.createElement("small");
      dimensions.textContent = "正在读取图片尺寸…";
      frame.append(image);
      caption.append(filename, dimensions);
      figure.append(frame, caption);
      upload.preview.append(figure);
      try {
        const entry = { image, url: URL.createObjectURL(file), active: true };
        upload.entries.push(entry);
        image.addEventListener("load", () => {
          if (entry.active) dimensions.textContent = `${image.naturalWidth} × ${image.naturalHeight} 像素`;
        });
        image.addEventListener("error", () => {
          if (entry.active) dimensions.textContent = "无法读取图片，请检查文件格式。";
        });
        image.src = entry.url;
      } catch (_) {
        dimensions.textContent = "当前浏览器无法预览此图片。";
      }
    });
  }

  for (const upload of uploads) {
    upload.input.addEventListener("change", () => {
      const selected = Array.from(upload.input.files);
      upload.files = upload.name === "refs" ? upload.files.concat(selected) : selected;
      if (upload.name === "refs") syncAccumulatedFiles(upload);
      renderPreviews(upload);
      void persistUploads();
    });
    upload.clear.addEventListener("click", () => {
      upload.input.value = "";
      upload.files = [];
      renderPreviews(upload);
      void persistUploads();
    });
    renderPreviews(upload);
  }
  window.addEventListener("pagehide", () => uploads.forEach(releasePreviews));

  function isNpu() { return device.value.trim().toLowerCase().startsWith("npu:"); }
  function isCuda() { return device.value.trim().toLowerCase().startsWith("cuda:"); }

  function updateModeFields() {
    const selected = mode.value;
    const visible = {first: ["i2va", "fl2va"].includes(selected), last: selected === "fl2va", refs: selected === "ref2va"};
    for (const [name, show] of Object.entries(visible)) {
      document.getElementById(`${name}-field`).classList.toggle("hidden", !show);
      const input = document.getElementById(name);
      input.disabled = !show;
      input.required = show;
    }
    document.getElementById("image-fields").classList.toggle("hidden", selected === "t2va");
    const hint = document.getElementById("mode-hint");
    hint.textContent = selected === "ref2va"
      ? "使用现有 FL2VA 权重，请在提示词中用 <Picture 1>、<Picture 2> 等引用参考图。"
      : "首尾帧会等比裁剪至所选分辨率，预览保留原图。提示词首行需包含对应模式指令。";
  }

  function updateTpState() {
    const enabled = isNpu() && tensorParallel.checked;
    const npuMerge = enabled && tpMergeDevice.value === "npu";
    tpOptions.classList.toggle("hidden", !enabled);
    tpMergeDevice.disabled = !enabled;
    tpMergeDelta.disabled = !npuMerge;
    tpDeltaRow.classList.toggle("hidden", !npuMerge);
    tpParity.disabled = !enabled;
    tpOffloadStrategy.disabled = !enabled || !offload.checked;
    if (!offload.checked) tpOffloadStrategy.value = "retained";
    tpCache.disabled = !enabled || npuMerge;
    updateConfig();
  }

  function updateDeviceFields() {
    optimizedRow.classList.toggle("hidden", !isCuda());
    if (!isCuda()) optimized.checked = false;
    tensorParallel.disabled = !isNpu();
    if (!isNpu()) tensorParallel.checked = false;
    if (!offloadTouched) offload.checked = isNpu();
    residentBlocks.disabled = !(isNpu() && offload.checked) || tensorParallel.checked;
    if (residentBlocks.disabled) residentBlocks.value = "0";
    updateTpState();
  }

  function updateTpFields() {
    const enabled = isNpu() && tensorParallel.checked;
    if (enabled) residentBlocks.value = "0";
    updateDeviceFields();
  }

  mode.addEventListener("change", updateModeFields);
  device.addEventListener("input", updateDeviceFields);
  offload.addEventListener("change", () => { offloadTouched = true; updateDeviceFields(); });
  tensorParallel.addEventListener("change", updateTpFields);
  tpParity.addEventListener("change", updateConfig);
  tpMergeDevice.addEventListener("change", updateTpState);
  tpMergeDelta.addEventListener("change", updateConfig);
  optimized.addEventListener("change", () => {
    if (optimized.checked) { offload.checked = false; offloadTouched = true; }
    updateDeviceFields();
  });
  updateModeFields();
  updateDeviceFields();
  updateTpFields();

  function formSnapshot() {
    if (!form.elements) return null;
    const fields = {};
    for (const element of form.elements) {
      if (!element.name || element.type === "file" || element.type === "submit") continue;
      fields[element.name] = element.type === "checkbox" ? element.checked : element.value;
    }
    return fields;
  }

  function saveFormState() {
    if (typeof sessionStorage === "undefined") return;
    const fields = formSnapshot();
    if (!fields) return;
    try {
      sessionStorage.setItem(FORM_STATE_KEY, JSON.stringify(fields));
    } catch (_) { /* storage full or blocked: the form still works */ }
  }

  function schedulePersist() {
    if (persistTimer !== null) return;
    if (typeof setTimeout === "undefined") { saveFormState(); return; }
    persistTimer = setTimeout(() => {
      persistTimer = null;
      saveFormState();
    }, 250);
  }

  function openUploadDb() {
    return new Promise((resolve) => {
      if (typeof indexedDB === "undefined") { resolve(null); return; }
      const request = indexedDB.open(UPLOAD_DB, 1);
      request.onupgradeneeded = () => request.result.createObjectStore(UPLOAD_STORE);
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => resolve(null);
    });
  }

  async function persistUploads() {
    if (!uploadsRestored || typeof File === "undefined") return;
    const db = await openUploadDb();
    if (!db) return;
    const payload = {};
    for (const upload of uploads) {
      payload[upload.name] = upload.files.map((file) => ({
        name: file.name, type: file.type, lastModified: file.lastModified, blob: file,
      }));
    }
    await new Promise((resolve) => {
      const transaction = db.transaction(UPLOAD_STORE, "readwrite");
      transaction.objectStore(UPLOAD_STORE).put(payload, "files");
      transaction.oncomplete = resolve;
      transaction.onerror = resolve;
    });
    db.close();
  }

  async function restoreUploads() {
    if (typeof File === "undefined") { uploadsRestored = true; return; }
    const db = await openUploadDb();
    if (!db) { uploadsRestored = true; return; }
    const payload = await new Promise((resolve) => {
      const transaction = db.transaction(UPLOAD_STORE, "readonly");
      const request = transaction.objectStore(UPLOAD_STORE).get("files");
      request.onsuccess = () => resolve(request.result || {});
      request.onerror = () => resolve({});
    });
    db.close();
    for (const upload of uploads) {
      const records = Array.isArray(payload[upload.name]) ? payload[upload.name] : [];
      upload.files = records.map((record) => new File([record.blob], record.name,
        { type: record.type, lastModified: record.lastModified }));
      syncAccumulatedFiles(upload);
      renderPreviews(upload);
    }
    uploadsRestored = true;
  }

  function restoreFormState() {
    if (typeof sessionStorage === "undefined") return;
    let fields;
    try {
      fields = JSON.parse(sessionStorage.getItem(FORM_STATE_KEY) || "null");
    } catch (_) { fields = null; }
    if (!fields || typeof fields !== "object") return;
    const byName = new Map();
    if (form.elements) {
      for (const element of form.elements) {
        if (element.name && !byName.has(element.name)) byName.set(element.name, element);
      }
    }
    for (const [name, value] of Object.entries(fields)) {
      const element = byName.get(name);
      if (!element) continue;
      if (element.type === "checkbox") element.checked = Boolean(value);
      else element.value = String(value);
    }
    if (Object.hasOwn(fields, "offload")) offloadTouched = true;
    if (byName.has("sampling")) updateSampling();
    if (byName.has("checkpoint")) checkpoint.value = String(fields.checkpoint ?? checkpoint.value);
    if (byName.has("steps")) steps.value = String(fields.steps ?? steps.value);
    if (byName.has("resolution")) updateResolution();
    if (byName.has("width")) width.value = String(fields.width ?? width.value);
    if (byName.has("height")) height.value = String(fields.height ?? height.value);
    updateModeFields();
    updateDeviceFields();
    updateTpFields();
    updateConfig();
  }

  function renderLatest(task) {
    const empty = document.getElementById("latest-empty");
    const panel = document.getElementById("latest-task");
    const link = document.getElementById("latest-log-link");
    if (!task) {
      empty.textContent = "暂无任务，提交后可在此查看进度。";
      empty.classList.remove("hidden"); panel.classList.add("hidden"); link.classList.add("hidden"); return;
    }
    empty.classList.add("hidden"); panel.classList.remove("hidden"); link.classList.remove("hidden");
    link.href = `/logs/${encodeURIComponent(task.id)}`;
    taskUI.renderProgress(task, "latest");
    const videoPanel = document.getElementById("latest-video");
    const player = document.getElementById("latest-video-player");
    const playable = task.status === "done" && !task.dry_run;
    videoPanel.classList.toggle("hidden", !playable);
    if (playable) {
      const source = `/api/tasks/${encodeURIComponent(task.id)}/video`;
      if (player.getAttribute("src") !== source) player.src = source;
      document.getElementById("latest-video-link").href = source;
    } else {
      player.removeAttribute("src");
    }
  }

  function renderHistory(tasks) {
    const root = document.getElementById("history");
    root.replaceChildren();
    if (!tasks.length) {
      const empty = document.createElement("div");
      empty.className = "muted";
      empty.textContent = "暂无历史任务。";
      root.append(empty);
      return;
    }
    for (const task of tasks) {
      const item = document.createElement("div");
      item.className = "history-item";
      const info = document.createElement("div");
      const title = document.createElement("strong");
      title.textContent = `${taskUI.statusLabel(task)} · ${task.mode}${task.dry_run ? " · 试运行" : ""}`;
      const details = document.createElement("small");
      details.textContent = `${task.id} · ${taskUI.stageLabel(task)} · ${taskUI.progressValue(task)}%${task.error ? ` · ${task.error}` : ""}`;
      info.append(title, details);
      const metadata = taskUI.taskMetadata(task);
      if (metadata.length) {
        const parameters = document.createElement("small");
        parameters.textContent = metadata.join(" · ");
        info.append(parameters);
      }
      const link = document.createElement("a");
      link.href = `/logs/${encodeURIComponent(task.id)}`;
      link.textContent = "查看日志";
      item.append(info, link);
      root.append(item);
    }
  }

  async function refresh() {
    if (polling) return;
    polling = true;
    try {
      const response = await fetch("/api/tasks", { cache: "no-store" });
      if (!response.ok) throw new Error("任务状态请求失败");
      const data = await response.json();
      if (!Array.isArray(data.tasks)) throw new Error("任务列表格式无效");
      renderHistory(data.tasks);
      renderLatest(data.tasks[0]);
      hasTaskData = true;
      document.getElementById("latest-refresh-error").textContent = "";
    } catch (_) {
      document.getElementById("latest-refresh-error").textContent = "无法获取任务状态，请检查服务连接，稍后自动重试。";
      if (!hasTaskData) {
        document.getElementById("latest-empty").textContent = "任务状态暂不可用。";
        document.getElementById("history").textContent = "历史记录暂不可用。";
      }
    } finally { polling = false; }
  }

  form.addEventListener("input", schedulePersist);
  form.addEventListener("change", schedulePersist);
  form.addEventListener("invalid", (event) => {
    const details = event.target.closest("details");
    if (details) details.open = true;
    error.textContent = event.target.validationMessage || "请检查表单参数。";
  }, true);

  window.addEventListener("pageshow", () => {
    submit.disabled = false;
    submit.textContent = "开始生成";
    // A page restored from the back-forward cache still retains its FileList.
    for (const upload of uploads) {
      if (!upload.entries.length && upload.files.length) renderPreviews(upload);
    }
  });

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    error.textContent = "";
    submit.disabled = true;
    submit.textContent = "正在提交…";
    try {
      const configError = updateConfig();
      if (configError) throw new Error(configError);
      updateModeFields();
      const body = new FormData(form);
      body.set("offload", String(offload.checked));
      body.set("npu_resident_blocks", residentBlocks.value || "0");
      body.set("tensor_parallel", String(tensorParallel.checked));
      body.set("tp_parity", tpParity.value || "row_stream");
      body.set("tp_offload_strategy", tpOffloadStrategy.value || "retained");
      body.set("tp_cache", tpMergeDevice.value === "cpu" ? tpCache.value.trim() : "");
      body.set("tp_merge_device", tpMergeDevice.value || "cpu");
      body.set("tp_merge_delta_device", tpMergeDelta.value || "npu");
      body.set("optimized_cuda", String(optimized.checked));
      for (const upload of uploads) {
        if (upload.input.disabled) continue;
        for (const file of upload.files) {
          if (file.size > 20 * 1024 * 1024) throw new Error(`${file.name} 超过单张 20 MiB 的限制。`);
        }
      }
      body.delete("refs");
      const refs = uploads.find((upload) => upload.name === "refs");
      if (!refs.input.disabled) {
        for (const file of refs.files) body.append("refs", file, file.name);
      }
      let response;
      try {
        response = await fetch("/api/tasks", { method: "POST", body });
      } catch (_) {
        throw new Error("无法连接服务，请检查连接后重试。");
      }
      let data;
      try {
        data = await response.json();
      } catch (_) {
        throw new Error("提交失败：服务返回无效响应。");
      }
      if (!response.ok) throw new Error(`提交失败：${data.error || `HTTP ${response.status}`}`);
      if (!data.id) throw new Error("提交失败：服务未返回任务编号。");
      saveFormState();
      await persistUploads();
      window.location.href = `/logs/${encodeURIComponent(data.id)}`;
    } catch (exc) {
      error.textContent = exc.message;
      submit.disabled = false;
      submit.textContent = "开始生成";
    }
  });

  restoreFormState();
  void restoreUploads();
  refresh();
  window.setInterval(refresh, 1000);
})();
