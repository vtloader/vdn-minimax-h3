(() => {
  const id = window.WEBUI_TASK_ID;
  const log = document.getElementById("task-log");
  const videoPanel = document.getElementById("task-video");
  const video = document.getElementById("video-player");
  const videoLink = document.getElementById("video-link");
  const taskUI = window.WebUITask;
  let polling = false;
  let hasStatus = false;
  let hasLog = false;

  async function refreshStatus() {
    const response = await fetch(`/api/tasks/${encodeURIComponent(id)}`, { cache: "no-store" });
    if (!response.ok) throw new Error("任务状态请求失败");
    const task = await response.json();
    taskUI.renderProgress(task, "task");
    hasStatus = true;
    const playable = task.status === "done" && !task.dry_run;
    videoPanel.classList.toggle("hidden", !playable);
    if (playable) {
      const source = `/api/tasks/${encodeURIComponent(id)}/video`;
      if (video.getAttribute("src") !== source) video.src = source;
      videoLink.href = source;
    } else {
      video.removeAttribute("src");
    }
  }

  async function refreshLog() {
    const response = await fetch(`/api/tasks/${encodeURIComponent(id)}/log`, { cache: "no-store" });
    if (!response.ok) throw new Error("日志请求失败");
    const text = await response.text();
    const atBottom = log.scrollTop + log.clientHeight >= log.scrollHeight - 24;
    const content = text || "暂无日志，等待任务开始…";
    if (log.textContent !== content) log.textContent = content;
    if (atBottom) log.scrollTop = log.scrollHeight;
    hasLog = true;
  }

  async function refresh() {
    if (polling) return;
    polling = true;
    try {
      const [statusResult, logResult] = await Promise.allSettled([refreshStatus(), refreshLog()]);
      const errors = [];
      if (statusResult.status === "rejected") {
        errors.push("任务状态暂时无法获取，将自动重试。");
        if (!hasStatus) document.getElementById("task-status").textContent = "状态暂不可用";
      }
      if (logResult.status === "rejected") {
        errors.push("日志暂时无法获取，将自动重试。");
        if (!hasLog) log.textContent = "日志暂不可用，正在重试…";
      }
      document.getElementById("task-refresh-error").textContent = errors.join(" ");
    } finally { polling = false; }
  }

  refresh();
  window.setInterval(refresh, 1000);
})();
