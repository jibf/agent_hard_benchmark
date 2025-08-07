// ===============================================
// 1. 数据定义
// 重新组织数据结构，以匹配新的表格列。
// ===============================================

// 示例：从你最新提供的图片中提取的 combinedModels 数据 (请根据你的完整数据补充)
const allCombinedModels = [
  // Closed-Source Large Language Models
  { category: "closed", model: "GPT-4o", atom: 93.4, single: 84.5, multi: 77.0, api: 85.0, pref: 83.0, summary: 87.6, special: 93.0, agent: 63.8, overall: 85.4 },
  { category: "closed", model: "Gemini-2.5-Pro-05-06", atom: 87.5, single: 78.2, multi: 70.0, api: 78.0, pref: 69.0, summary: 80.7, special: 90.0, agent: 63.8, overall: 80.6 },
  { category: "closed", model: "Qwen-Max", atom: 91.2, single: 80.5, multi: 68.0, api: 83.0, pref: 83.0, summary: 84.2, special: 74.0, agent: 64.3, overall: 78.4 },
  { category: "closed", model: "GPT-4o-Mini", atom: 86.5, single: 76.0, multi: 66.5, api: 77.0, pref: 78.0, summary: 79.9, special: 79.0, agent: 33.3, overall: 72.5 },
  { category: "closed", model: "Gemini-1.5-Pro", atom: 84.5, single: 76.8, multi: 64.5, api: 80.0, pref: 78.0, summary: 79.0, special: 78.7, agent: 25.5, overall: 70.7 },
  { category: "closed", model: "Claude-3-5-Sonnet", atom: 76.9, single: 72.5, multi: 62.5, api: 71.0, pref: 72.0, summary: 72.9, special: 77.4, agent: 39.5, overall: 68.9 },
  { category: "closed", model: "Doubao-Pro-32K", atom: 79.8, single: 55.5, multi: 58.0, api: 76.0, pref: 66.0, summary: 70.7, special: 55.0, agent: 25.0, overall: 59.4 },

  // Open-Source Large Language Models
  { category: "open", model: "Kimi-k2-0711", atom: 89.2, single: 83.0, multi: 72.5, api: 75.0, pref: 80.0, summary: 83.4, special: 84.0, agent: 74.2, overall: 82.2 },
  { category: "open", model: "Qwen2.5-Coder-32B-Instruct", atom: 90.2, single: 81.0, multi: 71.0, api: 83.0, pref: 81.0, summary: 84.1, special: 80.7, agent: 60.8, overall: 79.6 },
  { category: "open", model: "DeepSeekV3", atom: 91.5, single: 84.0, multi: 77.0, api: 83.0, pref: 83.0, summary: 86.5, special: 73.0, agent: 34.5, overall: 74.8 },
  { category: "open", model: "Qwen2.5-72B-Instruct", atom: 86.8, single: 80.3, multi: 69.5, api: 83.0, pref: 81.0, summary: 82.1, special: 75.7, agent: 45.0, overall: 74.7 },
  { category: "open", model: "Llama-3.1-70B-Instruct", atom: 82.5, single: 68.3, multi: 63.5, api: 79.0, pref: 68.0, summary: 75.5, special: 38.3, agent: 42.3, overall: 60.4 },
  { category: "open", model: "Qwen2.5-7B-Instruct", atom: 76.0, single: 60.3, multi: 58.5, api: 72.0, pref: 67.0, summary: 69.4, special: 47.0, agent: 13.8, overall: 54.8 },
  { category: "open", model: "DeepSeek-Coder-V2-Lite-Instruct", atom: 75.2, single: 57.8, multi: 46.5, api: 72.0, pref: 65.0, summary: 66.4, special: 40.3, agent: 2.0, overall: 49.5 },
  { category: "open", model: "Qwen2.5-Coder-7B-Instruct", atom: 76.0, single: 63.8, multi: 57.5, api: 74.0, pref: 68.0, summary: 70.1, special: 22.3, agent: 15.5, overall: 48.9 },
  { category: "open", model: "Watt-Tool-8B", atom: 85.7, single: 69.3, multi: 55.5, api: 79.0, pref: 64.0, summary: 75.6, special: 6.0, agent: 2.8, overall: 45.7 },
  { category: "open", model: "Hammer2.1-7B", atom: 73.7, single: 57.5, multi: 40.0, api: 62.0, pref: 55.0, summary: 62.8, special: 14.7, agent: 16.8, overall: 42.9 },
  { category: "open", model: "Llama-3.1-8B-Instruct", atom: 51.9, single: 39.8, multi: 28.0, api: 66.0, pref: 46.0, summary: 46.6, special: 21.0, agent: 5.3, overall: 33.4 },
  { category: "open", model: "Phi-3-Mini-128K-Instruct", atom: 57.2, single: 39.3, multi: 23.0, api: 58.0, pref: 32.0, summary: 46.5, special: 18.7, agent: 0.8, overall: 32.0 },
  { category: "open", model: "xLAM-7b-r", atom: 43.5, single: 22.0, multi: 19.0, api: 61.0, pref: 0.0, summary: 33.7, special: 2.7, agent: 8.8, overall: 21.6 },
  { category: "open", model: "Llama-3.2-3B-Instruct", atom: 38.7, single: 15.3, multi: 9.0, api: 42.0, pref: 32.0, summary: 29.6, special: 9.4, agent: 0.0, overall: 19.6 },
  { category: "open", model: "Hammer2.1-3B", atom: 22.4, single: 11.5, multi: 3.5, api: 40.0, pref: 20.0, summary: 18.7, special: 1.0, agent: 1.5, overall: 11.3 },
];

// !!! 占位符：请将你的中文和英文榜单数据填充到下方数组中 !!!
// 确保每个模型对象都有 'category', 'model', 'atom', 'single', 'multi', 'api', 'pref', 'summary', 'special', 'agent', 'overall' 这些属性
// 1. 数据定义
// 填充了中文和英文榜单数据。
// ===============================================

// 示例：从你提供的图片中提取的 combinedModels 数据
// （这部分与你之前的 `allCombinedModels` 相同，为了完整性再次包含）

const allChineseModels = [
  // Closed-Source Large Language Models
  { category: "closed", model: "GPT-4o", atom: 96.7, single: 91.0, multi: 86.0, api: 90.0, pref: 88.0, summary: 92.7, special: 93.3, agent: 71.5, overall: 89.6 },
  { category: "closed", model: "Gemini-2.5-Pro-05-06", atom: 91.3, single: 83.0, multi: 79.0, api: 84.0, pref: 80.0, summary: 86.3, special: 89.3, agent: 75.0, overall: 85.4 },
  { category: "closed", model: "Qwen-Max", atom: 94.3, single: 86.0, multi: 75.0, api: 92.0, pref: 84.0, summary: 88.7, special: 74.0, agent: 68.5, overall: 81.7 },
  { category: "closed", model: "GPT-4o-Mini", atom: 88.7, single: 78.5, multi: 74.0, api: 80.0, pref: 84.0, summary: 83.4, special: 81.3, agent: 39.0, overall: 76.0 },
  { category: "closed", model: "Claude-3-5-Sonnet", atom: 87.0, single: 81.0, multi: 79.0, api: 84.0, pref: 76.0, summary: 83.5, special: 82.0, agent: 35.0, overall: 75.6 },
  { category: "closed", model: "Gemini-1.5-Pro", atom: 86.7, single: 80.5, multi: 68.0, api: 86.0, pref: 84.0, summary: 82.2, special: 80.0, agent: 25.0, overall: 72.8 },
  { category: "closed", model: "Doubao-Pro-32K", atom: 84.3, single: 53.0, multi: 64.0, api: 82.0, pref: 78.0, summary: 75.0, special: 59.3, agent: 23.5, overall: 62.8 },

  // Open-Source Large Language Models
  { category: "open", model: "Kimi-k2-0711", atom: 91.3, single: 87.5, multi: 83.0, api: 80.0, pref: 86.0, summary: 87.9, special: 86.7, agent: 83.4, overall: 86.9 },
  { category: "open", model: "Qwen2.5-Coder-32B-Instruct", atom: 94.3, single: 88.5, multi: 83.0, api: 90.0, pref: 90.0, summary: 90.8, special: 81.3, agent: 71.5, overall: 85.3 },
  { category: "open", model: "Qwen2.5-72B-Instruct", atom: 92.3, single: 86.0, multi: 75.0, api: 90.0, pref: 82.0, summary: 87.3, special: 77.3, agent: 52.5, overall: 79.3 },
  { category: "open", model: "DeepSeekV3", atom: 95.0, single: 90.5, multi: 91.0, api: 90.0, pref: 88.0, summary: 92.6, special: 73.3, agent: 35.0, overall: 78.5 },
  { category: "open", model: "Llama-3.1-70B-Instruct", atom: 81.3, single: 65.0, multi: 66.0, api: 84.0, pref: 70.0, summary: 75.3, special: 47.3, agent: 43.5, overall: 62.9 },
  { category: "open", model: "Qwen2.5-7B-Instruct", atom: 81.7, single: 63.5, multi: 68.0, api: 82.0, pref: 76.0, summary: 75.9, special: 44.7, agent: 12.5, overall: 57.8 },
  { category: "open", model: "DeepSeek-Coder-V2-Lite-Instruct", atom: 78.7, single: 57.5, multi: 43.0, api: 82.0, pref: 70.0, summary: 68.8, special: 41.3, agent: 1.5, overall: 51.1 },
  { category: "open", model: "Qwen2.5-Coder-7B-Instruct", atom: 78.7, single: 64.0, multi: 63.0, api: 78.0, pref: 78.0, summary: 73.5, special: 19.3, agent: 12.5, overall: 49.6 },
  { category: "open", model: "Watt-Tool-8B", atom: 87.0, single: 67.0, multi: 54.0, api: 88.0, pref: 66.0, summary: 76.3, special: 10.0, agent: 4.0, overall: 47.4 },
  { category: "open", model: "Hammer2.1-7B", atom: 76.0, single: 62.5, multi: 37.0, api: 60.0, pref: 58.0, summary: 62.7, special: 26.0, agent: 18.5, overall: 46.1 },
  { category: "open", model: "Llama-3.1-8B-Instruct", atom: 52.7, single: 30.0, multi: 28.0, api: 72.0, pref: 36.0, summary: 45.0, special: 26.7, agent: 4.0, overall: 33.8 },
  { category: "open", model: "Phi-3-Mini-128K-Instruct", atom: 48.0, single: 29.5, multi: 15.0, api: 58.0, pref: 32.0, summary: 38.9, special: 25.3, agent: 1.5, overall: 29.5 },
  { category: "open", model: "Llama-3.2-3B-Instruct", atom: 45.7, single: 9.0, multi: 9.0, api: 50.0, pref: 32.0, summary: 32.7, special: 10.0, agent: 0.0, overall: 21.6 },
  { category: "open", model: "xLAM-7B-r", atom: 25.3, single: 2.0, multi: 6.0, api: 56.0, pref: 0.0, summary: 18.7, special: 1.3, agent: 7.5, overall: 12.3 },
  { category: "open", model: "Hammer2.1-3B", atom: 12.0, single: 9.0, multi: 0.0, api: 44.0, pref: 8.0, summary: 11.8, special: 1.3, agent: 1.5, overall: 7.4 },
];

const allEnglishModels = [
  // Closed-Source Large Language Models
  { category: "closed", model: "GPT-4o", atom: 90.0, single: 78.0, multi: 68.0, api: 80.0, pref: 78.0, summary: 82.5, special: 92.7, agent: 56.0, overall: 81.1 },
  { category: "closed", model: "Gemini-2.5-Pro-05-06", atom: 83.7, single: 73.5, multi: 61.0, api: 72.0, pref: 58.0, summary: 75.1, special: 90.7, agent: 52.5, overall: 75.8 },
  { category: "closed", model: "Qwen-Max", atom: 88.0, single: 75.0, multi: 61.0, api: 74.0, pref: 82.0, summary: 79.7, special: 74.0, agent: 60.0, overall: 75.1 },
  { category: "closed", model: "GPT-4o-Mini", atom: 84.3, single: 73.5, multi: 59.0, api: 74.0, pref: 72.0, summary: 76.4, special: 76.7, agent: 27.5, overall: 68.9 },
  { category: "closed", model: "Gemini-1.5-Pro", atom: 82.3, single: 73.0, multi: 61.0, api: 74.0, pref: 72.0, summary: 75.7, special: 77.3, agent: 26.0, overall: 68.5 },
  { category: "closed", model: "Claude-3-5-Sonnet", atom: 66.7, single: 64.0, multi: 46.0, api: 58.0, pref: 68.0, summary: 62.2, special: 72.7, agent: 44.0, overall: 62.2 },
  { category: "closed", model: "Doubao-Pro-32K", atom: 75.3, single: 58.0, multi: 52.0, api: 70.0, pref: 54.0, summary: 66.3, special: 50.7, agent: 26.5, overall: 56.0 },

  // Open-Source Large Language Models
  { category: "open", model: "Kimi-k2-0711", atom: 87.0, single: 78.5, multi: 62.0, api: 70.0, pref: 74.0, summary: 78.9, special: 81.3, agent: 65.0, overall: 77.4 },
  { category: "open", model: "Qwen2.5-Coder-32B-Instruct", atom: 86.0, single: 73.5, multi: 59.0, api: 76.0, pref: 72.0, summary: 77.4, special: 80.0, agent: 50.0, overall: 73.9 },
  { category: "open", model: "DeepSeekV3", atom: 88.0, single: 77.5, multi: 63.0, api: 76.0, pref: 78.0, summary: 80.3, special: 72.7, agent: 34.0, overall: 71.1 },
  { category: "open", model: "Qwen2.5-72B-Instruct", atom: 81.3, single: 74.5, multi: 64.0, api: 76.0, pref: 80.0, summary: 76.8, special: 74.0, agent: 37.5, overall: 70.0 },
  { category: "open", model: "Llama-3.1-70B-Instruct", atom: 83.7, single: 71.5, multi: 61.0, api: 74.0, pref: 66.0, summary: 75.6, special: 29.3, agent: 41.0, overall: 57.9 },
  { category: "open", model: "Qwen2.5-7B-Instruct", atom: 70.3, single: 57.0, multi: 49.0, api: 62.0, pref: 58.0, summary: 62.8, special: 49.3, agent: 15.0, overall: 51.8 },
  { category: "open", model: "Qwen2.5-Coder-7B-Instruct", atom: 73.3, single: 63.5, multi: 52.0, api: 70.0, pref: 58.0, summary: 66.6, special: 25.3, agent: 18.5, overall: 48.1 },
  { category: "open", model: "DeepSeek-Coder-V2-Lite-Instruct", atom: 71.7, single: 58.0, multi: 50.0, api: 62.0, pref: 60.0, summary: 64.0, special: 39.3, agent: 2.5, overall: 47.9 },
  { category: "open", model: "Watt-Tool-8B", atom: 84.7, single: 71.5, multi: 57.0, api: 70.0, pref: 62.0, summary: 74.8, special: 2.0, agent: 1.5, overall: 44.0 },
  { category: "open", model: "Hammer2.1-7B", atom: 71.3, single: 62.5, multi: 43.0, api: 64.0, pref: 52.0, summary: 62.9, special: 3.3, agent: 15.0, overall: 39.6 },
  { category: "open", model: "Phi-3-Mini-128K-Instruct", atom: 66.3, single: 49.0, multi: 31.0, api: 58.0, pref: 32.0, summary: 54.0, special: 12.0, agent: 0.0, overall: 34.4 },
  { category: "open", model: "mLlama-3.1-8B-Instruct", atom: 51.0, single: 49.5, multi: 28.0, api: 60.0, pref: 56.0, summary: 48.1, special: 15.3, agent: 6.5, overall: 32.9 },
  { category: "open", model: "xLAM-7B-r", atom: 61.7, single: 42.0, multi: 32.0, api: 66.0, pref: 0.0, summary: 48.7, special: 4.0, agent: 10.0, overall: 30.8 },
  { category: "open", model: "Llama-3.2-3B-Instruct", atom: 31.7, single: 21.5, multi: 9.0, api: 34.0, pref: 32.0, summary: 26.4, special: 8.7, agent: 0.0, overall: 17.6 },
  { category: "open", model: "Hammer2.1-3B", atom: 32.7, single: 14.0, multi: 7.0, api: 36.0, pref: 32.0, summary: 25.5, special: 0.7, agent: 1.5, overall: 15.2 },
];
// 用于追踪当前活跃的数据集 (默认为总榜单)
let currentActiveModels = allCombinedModels;

// 用于追踪当前列的排序方向
let sortDirection = {};

// ===============================================
// 2. 渲染函数
// ===============================================

function renderTable(models) {
  const tbodyClosed = document.getElementById("closed-source");
  const tbodyOpen = document.getElementById("open-source");

  // 清空现有行 (保留分类标题行)
  tbodyClosed.innerHTML = `<tr><td colspan="10"><strong>Closed-Source Large Language Models</strong></td></tr>`;
  tbodyOpen.innerHTML = `<tr><td colspan="10"><strong>Open-Source Large Language Models</strong></td></tr>`;

  // 检查是否没有模型数据
  if (models.length === 0) {
    const noResultsRow = document.createElement("tr");
    noResultsRow.innerHTML = `<td colspan="10" style="text-align: center; color: #888; padding: 20px;">No models found matching the criteria.</td>`;
    tbodyClosed.appendChild(noResultsRow);
    return;
  }

  // 过滤并渲染封闭源模型
  models.filter(m => m.category === "closed").forEach(m => {
    const row = document.createElement("tr");
    row.innerHTML = `
      <td>${m.model}</td>
      <td>${m.atom.toFixed(1)}</td> <td>${m.single.toFixed(1)}</td> <td>${m.multi.toFixed(1)}</td> <td>${m.api.toFixed(1)}</td> <td>${m.pref.toFixed(1)}</td> <td>${m.summary.toFixed(1)}</td> <td>${m.special.toFixed(1)}</td> <td>${m.agent.toFixed(1)}</td> <td><strong>${m.overall.toFixed(1)}</strong></td> `;
    tbodyClosed.appendChild(row);
  });

  // 过滤并渲染开源模型
  models.filter(m => m.category === "open").forEach(m => {
    const row = document.createElement("tr");
    row.innerHTML = `
      <td>${m.model}</td>
      <td>${m.atom.toFixed(1)}</td> <td>${m.single.toFixed(1)}</td> <td>${m.multi.toFixed(1)}</td> <td>${m.api.toFixed(1)}</td> <td>${m.pref.toFixed(1)}</td> <td>${m.summary.toFixed(1)}</td> <td>${m.special.toFixed(1)}</td> <td>${m.agent.toFixed(1)}</td> <td><strong>${m.overall.toFixed(1)}</strong></td> `;
    tbodyOpen.appendChild(row);
  });
}


// ===============================================
// 3. 榜单切换逻辑
// ===============================================

function switchLeaderboard(type) {
  // 重置搜索框
  const searchInput = document.getElementById('modelSearchInput');
  if (searchInput) searchInput.value = '';

  // 切换活跃数据集
  switch (type) {
    case 'combined':
      currentActiveModels = [...allCombinedModels]; // 创建副本以便排序和搜索不影响原始数据
      break;
    case 'chinese':
      currentActiveModels = [...allChineseModels];
      break;
    case 'english':
      currentActiveModels = [...allEnglishModels];
      break;
    default:
      currentActiveModels = [...allCombinedModels];
  }

  // 渲染新的榜单
  renderTable(currentActiveModels);

  // 更新按钮的活跃状态
  document.querySelectorAll('.leaderboard-tabs button').forEach(btn => {
    btn.classList.remove('active');
  });
  document.getElementById(`show${type.charAt(0).toUpperCase() + type.slice(1)}`).classList.add('active');
}

// ===============================================
// 4. 搜索逻辑
// ===============================================

function handleSearch() {
  const searchTerm = document.getElementById('modelSearchInput').value.toLowerCase();
  let filteredModels = [];

  // 过滤当前活跃的数据集
  if (currentActiveModels) {
    filteredModels = currentActiveModels.filter(model =>
      model.model.toLowerCase().includes(searchTerm)
    );
  }

  renderTable(filteredModels);
}

// ===============================================
// 5. 排序逻辑
// ===============================================

function handleSort(key, headerElement) {
  if (!currentActiveModels || currentActiveModels.length === 0) return;

  // 切换排序方向
  sortDirection[key] = sortDirection[key] === 'asc' ? 'desc' : 'asc';

  currentActiveModels.sort((a, b) => {
    const valA = a[key];
    const valB = b[key];

    if (typeof valA === 'string') {
      return sortDirection[key] === 'asc' ? valA.localeCompare(valB) : valB.localeCompare(a[key]);
    }
    // 对于数字，由于数据是分数，通常希望从高到低排序，除非明确指定升序
    // 默认降序 (从高到低)，如果 sortDirection[key] 是 'asc' 则升序
    return sortDirection[key] === 'asc' ? valA - valB : valB - valA;
  });

  // 重新渲染，但保持搜索过滤（如果存在）
  handleSearch(); // 重新应用搜索过滤器，因为排序后可能需要重新过滤
}


// ===============================================
// 6. 初始化和事件监听
// ===============================================

document.addEventListener("DOMContentLoaded", () => {
  // 首次加载时显示总榜单
  switchLeaderboard('combined');

  // 获取 DOM 元素
  const showCombinedBtn = document.getElementById('showCombined');
  const showChineseBtn = document.getElementById('showChinese');
  const showEnglishBtn = document.getElementById('showEnglish');
  const searchButton = document.getElementById('searchButton');
  const searchInput = document.getElementById('modelSearchInput');
  const headers = document.querySelectorAll("#leaderboard th");

  // 榜单切换按钮事件
  if (showCombinedBtn) showCombinedBtn.addEventListener('click', () => switchLeaderboard('combined'));
  if (showChineseBtn) showChineseBtn.addEventListener('click', () => switchLeaderboard('chinese'));
  if (showEnglishBtn) showEnglishBtn.addEventListener('click', () => switchLeaderboard('english'));

  // 搜索事件
  if (searchButton) searchButton.addEventListener('click', handleSearch);
  if (searchInput) {
    searchInput.addEventListener('keypress', (event) => {
      if (event.key === 'Enter') {
        handleSearch();
      }
    });
  }

  // 排序事件 (需要映射到数据键名)
  headers.forEach((header) => { // 移除 index 参数，因为现在不是直接用索引映射了
    const headerText = header.textContent.trim();
    let dataKey;

    // 映射表头文本到数据对象中的键名
    switch (headerText) {
      case 'Model': dataKey = 'model'; break;
      case 'Atom': dataKey = 'atom'; break;
      case 'Single-Turn': dataKey = 'single'; break;
      case 'Multi-Turn': dataKey = 'multi'; break;
      case 'Similar API': dataKey = 'api'; break;
      case 'Preference': dataKey = 'pref'; break;
      case 'Summary': dataKey = 'summary'; break;
      case 'Special': dataKey = 'special'; break;
      case 'Agent': dataKey = 'agent'; break;
      case 'Overall': dataKey = 'overall'; break;
      // Normal 这样的父级表头不直接对应数据，不应可排序
      default: return; // 如果不是可排序的表头，则跳过
    }

    if (dataKey) { // 只有对应到数据键名的表头才可排序
      header.style.cursor = "pointer";
      header.addEventListener("click", () => handleSort(dataKey, header));
    }
  });
});