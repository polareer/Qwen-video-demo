const API_BASE = '/api';

const translations = {
  en: {
    page_title: "AR Assembly Video Analysis | AI Vision Lab",
    model_offline: "MODEL OFFLINE",
    model_online: "MODEL ONLINE",
    input_source: "INPUT SOURCE",
    load_model: "LOAD MODEL",
    drop_video: "DROP VIDEO FILE HERE",
    click_browse: "or click to browse",
    configuration: "CONFIGURATION",
    sampling_fps: "SAMPLING FPS",
    max_tokens: "MAX TOKENS",
    max_pixels: "MAX PIXELS",
    analysis_prompt: "ANALYSIS PROMPT",
    start_analysis: "START ANALYSIS",
    clear: "CLEAR",
    analysis_output: "ANALYSIS OUTPUT",
    processing: "PROCESSING...",
    tab_summary: "SUMMARY",
    tab_steps: "STEPS",
    tab_raw: "RAW OUTPUT",
    awaiting_analysis: "Awaiting video analysis...",
    upload_hint: "Upload a video and start analysis to see results here",
    video_summary: "VIDEO SUMMARY",
    assembly_goal: "ASSEMBLY GOAL",
    overall_risks: "OVERALL RISKS",
    suggestions: "SUGGESTIONS",
    no_steps: "No steps to display",
    raw_hint: "Raw output will appear here",
    notification_loading_model: "Loading model...",
    notification_model_loading: "Model loading in background...",
    notification_model_loaded: "Model loaded successfully!",
    notification_load_model_error: "Failed to load model: ",
    notification_video_loaded: "Video loaded",
    notification_please_drop_video: "Please drop a video file",
    notification_analyzing: "Analyzing video...",
    notification_analysis_complete: "Analysis complete!",
    notification_error: "Error: ",
    no_summary_available: "No summary available",
    no_goal_specified: "No goal specified",
    no_risks_identified: "No risks identified",
    no_suggestions_available: "No suggestions available",
    no_action_described: "No action described",
    step: "STEP",
    approximate: "approximate",
    ar_guide: "AR GUIDE",
    risk: "RISK",
    confidence: "confidence",
    high: "high",
    medium: "medium",
    low: "low",
    prompt: `Please understand this first-person augmented reality assembly video and output a structured analysis result.

Please return in JSON format with the following fields:
{
  "video_summary": "One sentence summary of the video content",
  "assembly_goal": "The goal of this assembly task",
  "steps": [
    {
      "step_id": 1,
      "time_range": "start and end time, write 'approximate' if unknown",
      "action": "What action was performed",
      "objects": ["part/tool/equipment"],
      "ar_guidance": "AR guidance information in the screen",
      "possible_issue": "Possible issues, ambiguities or risks",
      "confidence": "high/medium/low"
    }
  ],
  "overall_risks": ["Overall risk points"],
  "improvement_suggestions": [
    "How to improve the video understanding system",
    "How to improve the assembly process or AR guidance"
  ]
}

If you cannot pinpoint the exact time, please state it is an approximate judgment, but still try to break down the steps.`
  },
  zh: {
    page_title: "AR 装配视频分析 | AI Vision Lab",
    model_offline: "模型未加载",
    model_online: "模型已就绪",
    input_source: "视频输入",
    load_model: "加载模型",
    drop_video: "拖放视频文件到这里",
    click_browse: "或点击浏览",
    configuration: "参数配置",
    sampling_fps: "采样帧率",
    max_tokens: "最大 Token 数",
    max_pixels: "最大像素数",
    analysis_prompt: "分析提示词",
    start_analysis: "开始分析",
    clear: "清空",
    analysis_output: "分析结果",
    processing: "处理中...",
    tab_summary: "摘要",
    tab_steps: "步骤",
    tab_raw: "原始输出",
    awaiting_analysis: "等待视频分析...",
    upload_hint: "上传视频并开始分析以查看结果",
    video_summary: "视频摘要",
    assembly_goal: "装配目标",
    overall_risks: "整体风险",
    suggestions: "改进建议",
    no_steps: "暂无步骤显示",
    raw_hint: "原始输出将显示在这里",
    notification_loading_model: "正在加载模型...",
    notification_model_loading: "模型正在后台加载...",
    notification_model_loaded: "模型加载成功！",
    notification_load_model_error: "加载模型失败：",
    notification_video_loaded: "视频已加载",
    notification_please_drop_video: "请拖放一个视频文件",
    notification_analyzing: "正在分析视频...",
    notification_analysis_complete: "分析完成！",
    notification_error: "错误：",
    no_summary_available: "暂无摘要",
    no_goal_specified: "未指定目标",
    no_risks_identified: "未识别到风险",
    no_suggestions_available: "暂无建议",
    no_action_described: "未描述动作",
    step: "步骤",
    approximate: "近似",
    ar_guide: "AR 指引",
    risk: "风险",
    confidence: "置信度",
    high: "高",
    medium: "中",
    low: "低",
    prompt: `请理解这个第一人称增强现实装配视频，并输出一个结构化分析结果。

请尽量按 JSON 返回，字段建议如下：
{
  "video_summary": "一句话概述视频内容",
  "assembly_goal": "本次装配任务目标",
  "steps": [
    {
      "step_id": 1,
      "time_range": "起止时间，未知可写 approximate",
      "action": "执行了什么动作",
      "objects": ["零件/工具/设备"],
      "ar_guidance": "画面中的 AR 指引信息",
      "possible_issue": "可能的问题、模糊点或风险",
      "confidence": "high/medium/low"
    }
  ],
  "overall_risks": ["整体风险点"],
  "improvement_suggestions": [
    "如何改进视频理解系统",
    "如何改进装配流程或 AR 引导"
  ]
}

如果无法精确定位时间，请明确说明是近似判断，但仍然要尽量拆出步骤。`
  }
};

let currentLang = 'zh';

const state = {
  videoFile: null,
  videoURL: null,
  modelLoaded: false,
  currentResult: null,
};

const els = {
  langBtns: document.querySelectorAll('.lang-btn'),
  modelStatus: document.getElementById('model-status'),
  currentTime: document.getElementById('current-time'),
  uploadArea: document.getElementById('upload-area'),
  videoInput: document.getElementById('video-input'),
  videoPreviewContainer: document.getElementById('video-preview-container'),
  videoPreview: document.getElementById('video-preview'),
  videoName: document.getElementById('video-name'),
  videoDuration: document.getElementById('video-duration'),
  loadModelBtn: document.getElementById('load-model-btn'),
  fpsSlider: document.getElementById('fps-slider'),
  fpsValue: document.getElementById('fps-value'),
  tokensSlider: document.getElementById('tokens-slider'),
  tokensValue: document.getElementById('tokens-value'),
  pixelsSlider: document.getElementById('pixels-slider'),
  pixelsValue: document.getElementById('pixels-value'),
  promptInput: document.getElementById('prompt-input'),
  analyzeBtn: document.getElementById('analyze-btn'),
  clearBtn: document.getElementById('clear-btn'),
  progressIndicator: document.getElementById('progress-indicator'),
  tabBtns: document.querySelectorAll('.tab-btn'),
  summaryEmpty: document.getElementById('summary-empty'),
  summaryContent: document.getElementById('summary-content'),
  summaryText: document.getElementById('summary-text'),
  goalText: document.getElementById('goal-text'),
  riskList: document.getElementById('risk-list'),
  suggestionList: document.getElementById('suggestion-list'),
  stepsEmpty: document.getElementById('steps-empty'),
  stepsContent: document.getElementById('steps-content'),
  rawEmpty: document.getElementById('raw-empty'),
  rawContent: document.getElementById('raw-content'),
};

function t(key) {
  return translations[currentLang][key] || translations['en'][key] || key;
}

function updateLanguage(lang) {
  currentLang = lang;
  document.documentElement.lang = lang === 'zh' ? 'zh-CN' : 'en';

  els.langBtns.forEach(btn => {
    const isActive = btn.dataset.lang === lang;
    btn.classList.toggle('active', isActive);
    btn.setAttribute('aria-pressed', isActive ? 'true' : 'false');
  });

  document.querySelectorAll('[data-i18n]').forEach(el => {
    const key = el.getAttribute('data-i18n');
    if (translations[currentLang][key]) {
      el.textContent = translations[currentLang][key];
    }
  });

  document.title = t('page_title');
  els.promptInput.value = t('prompt');

  if (state.currentResult) {
    renderResult(state.currentResult);
  }
}

function updateTime() {
  const now = new Date();
  const timeStr = now.toLocaleTimeString(currentLang === 'zh' ? 'zh-CN' : 'en-US', {
    hour12: false,
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  });
  els.currentTime.textContent = timeStr;
}

function formatDuration(seconds) {
  if (!Number.isFinite(seconds) || seconds <= 0) return '-';
  const mins = Math.floor(seconds / 60);
  const secs = Math.round(seconds % 60);
  return `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
}

function showNotification(message, type = 'info') {
  const container = document.getElementById('notifications');
  const notification = document.createElement('div');
  notification.className = `notification ${type}`;
  notification.textContent = message;
  container.appendChild(notification);
  setTimeout(() => notification.remove(), 4000);
}

async function checkModelStatus() {
  try {
    const response = await fetch(`${API_BASE}/status`);
    const data = await response.json();
    state.modelLoaded = data.model_loaded;
    updateModelStatus();
  } catch (e) {
    console.error('Failed to check model status:', e);
  }
}

function updateModelStatus() {
  const statusText = els.modelStatus.querySelector('.status-text');
  if (state.modelLoaded) {
    els.modelStatus.classList.add('online');
    statusText.textContent = t('model_online');
    els.loadModelBtn.disabled = true;
    els.loadModelBtn.style.opacity = '0.3';
  } else {
    els.modelStatus.classList.remove('online');
    statusText.textContent = t('model_offline');
    els.loadModelBtn.disabled = false;
    els.loadModelBtn.style.opacity = '1';
  }
  updateAnalyzeButton();
}

async function loadModel() {
  try {
    showNotification(t('notification_loading_model'), 'info');
    els.loadModelBtn.disabled = true;
    els.loadModelBtn.style.opacity = '0.5';

    const response = await fetch(`${API_BASE}/load-model`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
    });

    const data = await response.json();

    if (data.status === 'loading') {
      showNotification(t('notification_model_loading'), 'info');
      const checkInterval = setInterval(async () => {
        await checkModelStatus();
        if (state.modelLoaded) {
          clearInterval(checkInterval);
          showNotification(t('notification_model_loaded'), 'success');
        }
      }, 2000);
    }
  } catch (e) {
    showNotification(t('notification_load_model_error') + e.message, 'error');
    els.loadModelBtn.disabled = false;
    els.loadModelBtn.style.opacity = '1';
  }
}

function updateAnalyzeButton() {
  const canAnalyze = state.videoFile && state.modelLoaded;
  els.analyzeBtn.disabled = !canAnalyze;
}

function resetVideo() {
  if (state.videoURL) {
    URL.revokeObjectURL(state.videoURL);
  }
  state.videoFile = null;
  state.videoURL = null;
  els.videoInput.value = '';
  els.videoPreview.pause();
  els.videoPreview.removeAttribute('src');
  els.uploadArea.style.display = 'flex';
  els.videoPreviewContainer.style.display = 'none';
  updateAnalyzeButton();
}

function handleVideoFile(file) {
  if (!file) {
    resetVideo();
    return;
  }

  if (state.videoURL) {
    URL.revokeObjectURL(state.videoURL);
  }

  state.videoFile = file;
  state.videoURL = URL.createObjectURL(file);

  els.videoPreview.src = state.videoURL;
  els.uploadArea.style.display = 'none';
  els.videoPreviewContainer.style.display = 'flex';
  els.videoName.textContent = file.name;

  els.videoPreview.onloadedmetadata = () => {
    els.videoDuration.textContent = formatDuration(els.videoPreview.duration);
  };

  updateAnalyzeButton();
  showNotification(t('notification_video_loaded'), 'success');
}

async function analyzeVideo() {
  if (!state.videoFile || !state.modelLoaded) return;

  const formData = new FormData();
  formData.append('video', state.videoFile);
  formData.append('prompt', els.promptInput.value);
  formData.append('fps', els.fpsSlider.value);
  formData.append('max_pixels', els.pixelsSlider.value);
  formData.append('max_new_tokens', els.tokensSlider.value);

  els.progressIndicator.style.display = 'flex';
  els.analyzeBtn.disabled = true;

  try {
    showNotification(t('notification_analyzing'), 'info');

    const response = await fetch(`${API_BASE}/analyze`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || 'Analysis failed');
    }

    const result = await response.json();
    state.currentResult = result;
    renderResult(result);
    showNotification(t('notification_analysis_complete'), 'success');
  } catch (e) {
    showNotification(t('notification_error') + e.message, 'error');
  } finally {
    els.progressIndicator.style.display = 'none';
    els.analyzeBtn.disabled = false;
  }
}

function renderResult(result) {
  const data = result.parsed || {};

  if (data.video_summary || data.assembly_goal || data.overall_risks || data.improvement_suggestions) {
    els.summaryEmpty.style.display = 'none';
    els.summaryContent.style.display = 'grid';

    els.summaryText.textContent = data.video_summary || t('no_summary_available');
    els.goalText.textContent = data.assembly_goal || t('no_goal_specified');

    els.riskList.innerHTML = '';
    if (Array.isArray(data.overall_risks)) {
      data.overall_risks.forEach(risk => {
        const li = document.createElement('li');
        li.textContent = risk;
        els.riskList.appendChild(li);
      });
    }
    if (!els.riskList.children.length) {
      const li = document.createElement('li');
      li.textContent = t('no_risks_identified');
      els.riskList.appendChild(li);
    }

    els.suggestionList.innerHTML = '';
    if (Array.isArray(data.improvement_suggestions)) {
      data.improvement_suggestions.forEach(suggestion => {
        const li = document.createElement('li');
        li.textContent = suggestion;
        els.suggestionList.appendChild(li);
      });
    }
    if (!els.suggestionList.children.length) {
      const li = document.createElement('li');
      li.textContent = t('no_suggestions_available');
      els.suggestionList.appendChild(li);
    }
  }

  if (Array.isArray(data.steps) && data.steps.length > 0) {
    els.stepsEmpty.style.display = 'none';
    els.stepsContent.style.display = 'grid';
    els.stepsContent.innerHTML = '';

    data.steps.forEach(step => {
      const card = document.createElement('div');
      card.className = 'step-card';

      const header = document.createElement('div');
      header.className = 'step-header';

      const number = document.createElement('span');
      number.className = 'step-number';
      number.textContent = `${t('step')} ${step.step_id || '-'}`;

      const time = document.createElement('span');
      time.className = 'step-time';
      time.textContent = step.time_range || t('approximate');

      header.appendChild(number);
      header.appendChild(time);
      card.appendChild(header);

      const action = document.createElement('div');
      action.className = 'step-action';
      action.textContent = step.action || t('no_action_described');
      card.appendChild(action);

      const details = document.createElement('div');
      details.className = 'step-details';

      if (step.ar_guidance) {
        const detail = document.createElement('div');
        detail.className = 'step-detail';
        detail.innerHTML = `<span class="step-detail-label">${t('ar_guide')}</span><span class="step-detail-value">${step.ar_guidance}</span>`;
        details.appendChild(detail);
      }

      if (step.possible_issue) {
        const detail = document.createElement('div');
        detail.className = 'step-detail';
        detail.innerHTML = `<span class="step-detail-label">${t('risk')}</span><span class="step-detail-value">${step.possible_issue}</span>`;
        details.appendChild(detail);
      }

      card.appendChild(details);

      const tags = document.createElement('div');
      tags.className = 'step-tags';

      if (Array.isArray(step.objects)) {
        step.objects.forEach(obj => {
          const tag = document.createElement('span');
          tag.className = 'step-tag';
          tag.textContent = obj;
          tags.appendChild(tag);
        });
      }

      if (step.confidence) {
        const tag = document.createElement('span');
        const confText = step.confidence === 'high' ? t('high') :
                         step.confidence === 'medium' ? t('medium') : t('low');
        tag.className = `step-tag confidence-${step.confidence}`;
        tag.textContent = `${t('confidence')}: ${confText}`;
        tags.appendChild(tag);
      }

      card.appendChild(tags);
      els.stepsContent.appendChild(card);
    });
  }

  if (result.raw_text) {
    els.rawEmpty.style.display = 'none';
    els.rawContent.style.display = 'block';
    els.rawContent.textContent = result.raw_text;
  }
}

function switchTab(tabName) {
  els.tabBtns.forEach(btn => {
    const isActive = btn.dataset.tab === tabName;
    btn.classList.toggle('active', isActive);
    btn.setAttribute('aria-selected', isActive ? 'true' : 'false');
  });

  document.querySelectorAll('.tab-content').forEach(content => {
    content.classList.remove('active');
    content.hidden = true;
  });

  const activeTab = document.getElementById(`tab-${tabName}`);
  activeTab.classList.add('active');
  activeTab.hidden = false;
}

function bindEvents() {
  setInterval(updateTime, 1000);
  updateTime();

  updateLanguage(currentLang);

  els.langBtns.forEach(btn => {
    btn.addEventListener('click', () => updateLanguage(btn.dataset.lang));
  });

  checkModelStatus();
  setInterval(checkModelStatus, 10000);

  els.loadModelBtn.addEventListener('click', loadModel);

  els.uploadArea.addEventListener('click', () => els.videoInput.click());
  els.uploadArea.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      els.videoInput.click();
    }
  });
  els.videoInput.addEventListener('change', (e) => handleVideoFile(e.target.files[0]));

  els.uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    els.uploadArea.classList.add('dragover');
  });

  els.uploadArea.addEventListener('dragleave', () => {
    els.uploadArea.classList.remove('dragover');
  });

  els.uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    els.uploadArea.classList.remove('dragover');
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('video/')) {
      handleVideoFile(file);
    } else {
      showNotification(t('notification_please_drop_video'), 'error');
    }
  });

  els.fpsSlider.addEventListener('input', () => {
    els.fpsValue.textContent = els.fpsSlider.value;
    els.fpsSlider.setAttribute('aria-valuenow', els.fpsSlider.value);
  });

  els.tokensSlider.addEventListener('input', () => {
    els.tokensValue.textContent = els.tokensSlider.value;
    els.tokensSlider.setAttribute('aria-valuenow', els.tokensSlider.value);
  });

  els.pixelsSlider.addEventListener('input', () => {
    els.pixelsValue.textContent = els.pixelsSlider.value;
    els.pixelsSlider.setAttribute('aria-valuenow', els.pixelsSlider.value);
  });

  els.analyzeBtn.addEventListener('click', analyzeVideo);
  els.clearBtn.addEventListener('click', resetVideo);

  els.tabBtns.forEach(btn => {
    btn.addEventListener('click', () => switchTab(btn.dataset.tab));
  });

  els.promptInput.value = t('prompt');
}

bindEvents();
