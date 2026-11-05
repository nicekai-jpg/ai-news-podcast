/* states/_shared.js — sample data + hydrate for preview
   注意：这里只放状态页需要的字段，避免与主页面 DATA 重复维护。
   字段名与 index.html 保持一致，见 DESIGN.md 第 7 章。 */
const DATA = {
  issue_number: "286",
  issue_date: "2026-10-15",

  /* 播放器错误 */
  player_error_headline: "音频加载失败",
  player_error_message: "网络连接异常，无法加载本期音频文件。请检查网络后重试，或切换到逐字稿阅读。",
  player_error_code: "NET_ERR",
  player_error_time: "2026-10-15 08:32",

  /* 日报为空 */
  empty_report_headline: "今日暂无深度日报",
  empty_report_message: "编辑部正在整理今日内容，日报将在稍后发布。您可以先收听播客或浏览信源。",
  empty_report_reason_1: "内容仍在审核中",
  empty_report_reason_2: "稍后刷新页面",

  /* 信源为空 */
  empty_sources_headline: "今日暂无信源数据",
  empty_sources_message: "信源扫描服务暂时不可用，或今日未采集到符合条件的信源。",
  empty_sources_scan_time: "2026-10-15 06:00",
  empty_sources_status: "扫描中",

  /* 雷达空态 */
  radar_headline: "今日候选池经多轮筛查，未检出达标项目",
  radar_paragraph: "本栏目每日严选 1 个具备重大架构革新且近 7 日 Star 增速稳定突破 300/天的优质开源项目。今日全网扫描了一批候选仓库，经工程稳定性、重复收录排查及实际代码审查，均处于评估观察期或版本迁移期，编辑部决定今日宁缺毋滥、轮空主推。",
  radar_reason: "宁缺毋滥、轮空主推",
  candidate_count: "12",
  excluded_recent: "12",

  /* 雷达主推态：与主页面第 286 期保持同一份数据 */
  radar_pick_repo: "microsoft/autogen",
  radar_pick_lang: "Python",
  radar_pick_stars: "42.3k",
  radar_pick_growth: "312",
  radar_pick_license: "MIT",
  radar_pick_desc: "微软开源的智能体编排框架，用对话驱动多智能体协作，支持工具调用、代码执行与人工介入。近 7 日 Star 增速持续高于收录阈值，工程活跃度稳定。",
  radar_pick_url: "https://github.com/microsoft/autogen",

  /* 主持人姓名（与主页面 host_a_name / host_b_name 一致） */
  host_a_name: "林夏",
  host_b_name: "陈默",

  /* 逐字稿片段 */
  transcript_sample_a: "听众朋友大家好，欢迎收听 今日AI看点，今天是 2026 年 10 月 15 日。我是林夏。",
  transcript_sample_b: "大家好，我是陈默。林夏今天气色不错啊。",
  transcript_sample_c: "那是，昨晚睡了个好觉。对了陈默，你最近关注智能体这块没有？"
};

function hydrate() {
  const roots = [document.head, document.body];
  roots.forEach(root => {
    if (!root) return;
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, null, false);
    const nodes = [];
    while (walker.nextNode()) nodes.push(walker.currentNode);
    nodes.forEach(node => {
      if (!node.nodeValue || !node.nodeValue.includes('{{')) return;
      node.nodeValue = node.nodeValue.replace(/\{\{\s*([a-zA-Z0-9_]+)\s*\}\}/g, (m, key) => {
        const v = DATA[key];
        return (v == null || String(v).trim() === '' || String(v).startsWith('{{')) ? m : String(v);
      });
    });
  });
  document.querySelectorAll('*').forEach(el => {
    for (const attr of Array.from(el.attributes)) {
      if (attr.value && attr.value.includes('{{')) {
        el.setAttribute(attr.name, attr.value.replace(/\{\{\s*([a-zA-Z0-9_]+)\s*\}\}/g, (m, key) => {
          const v = DATA[key];
          return (v == null || String(v).trim() === '' || String(v).startsWith('{{')) ? m : String(v);
        }));
      }
    }
  });
  if (document.title && document.title.includes('{{')) {
    document.title = document.title.replace(/\{\{\s*([a-zA-Z0-9_]+)\s*\}\}/g, (m, key) => {
      const v = DATA[key];
      return (v == null || String(v).trim() === '' || String(v).startsWith('{{')) ? m : String(v);
    });
  }
}

/* Theme: match the main page (localStorage + system preference) and offer a toggle */
(function initStateTheme() {
  var theme = 'light';
  try {
    var saved = localStorage.getItem('theme');
    var prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
    theme = saved || (prefersDark ? 'dark' : 'light');
  } catch (e) {}
  document.documentElement.setAttribute('data-theme', theme);

  function label(t) { return t === 'dark' ? '浅色' : '深色'; }

  document.addEventListener('DOMContentLoaded', function() {
    var btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'state-theme-toggle';
    btn.setAttribute('aria-label', '切换深浅主题');
    btn.textContent = label(theme);
    btn.addEventListener('click', function() {
      theme = theme === 'dark' ? 'light' : 'dark';
      document.documentElement.setAttribute('data-theme', theme);
      try { localStorage.setItem('theme', theme); } catch (e) {}
      btn.textContent = label(theme);
    });
    document.body.appendChild(btn);
  });
})();

document.addEventListener('DOMContentLoaded', hydrate);
