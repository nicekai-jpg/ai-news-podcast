(function () {
  'use strict';

  var SITE = JSON.parse(document.getElementById('site-data').textContent);
  var DATES = SITE.dates || [];
  var EPISODES = SITE.episodes || {};
  var VOICES = SITE.voices || {};

  var state = {
    date: null,
    digest: null,
    chunks: [],
    sourceGroup: 'all',
    sourceQuery: '',
    sourcesExpanded: false,
    drawerFilter: 'all',
    player: null,
  };

  var el = function (id) { return document.getElementById(id); };
  var HOST_NAMES = { A: '苏晴', B: '周航' };
  var SOURCE_PREVIEW = 12;
  var radarLoop = { running: false, raf: 0, reduceMotion: false };

  function showToast(message) {
    var toast = el('toast-pill');
    if (!toast) return;
    toast.textContent = message;
    toast.classList.add('show');
    clearTimeout(showToast._timer);
    showToast._timer = setTimeout(function () { toast.classList.remove('show'); }, 2200);
  }

  function fmtTime(seconds) {
    var total = Math.max(0, Math.floor(seconds || 0));
    var mm = String(Math.floor(total / 60)).padStart(2, '0');
    var ss = String(total % 60).padStart(2, '0');
    return mm + ':' + ss;
  }

  function escapeHtml(value) {
    return String(value == null ? '' : value)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;')
      .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  }

  function placeholder(icon, text) {
    return '<div class="state-placeholder"><div class="state-icon">' + icon + '</div><div>' +
      escapeHtml(text) + '</div></div>';
  }

  async function fetchJson(url) {
    var response = await fetch(url, { cache: 'no-cache' });
    if (!response.ok) throw new Error(url + ' ' + response.status);
    return response.json();
  }

  /* ---------- 日期轴 ---------- */
  function renderDateAxis() {
    var wrap = el('date-pills');
    wrap.innerHTML = '';
    var calendar = state.calendar || {};
    var byDate = {};
    (calendar.dates || []).forEach(function (item) { byDate[item.date] = item; });
    DATES.forEach(function (date) {
      var meta = byDate[date] || {};
      var issue = meta.issue_number ? ' · 第 ' + meta.issue_number + ' 期' : '';
      var pill = document.createElement('button');
      pill.className = 'date-pill' + (date === state.date ? ' active' : '');
      pill.setAttribute('data-date', date);
      pill.textContent = (date === DATES[0] ? date + ' (最新' + issue + ')' : date + issue);
      pill.addEventListener('click', function () { loadDate(date); });
      wrap.appendChild(pill);
    });
    var index = DATES.indexOf(state.date);
    el('date-prev').disabled = index < 0 || index >= DATES.length - 1;
    el('date-next').disabled = index <= 0;
    el('date-latest').hidden = index <= 0;
    document.querySelectorAll('.nav-anchor-link').forEach(function (link) {
      link.classList.toggle('active', link.getAttribute('href') === '#section-hero');
    });
  }

  /* ---------- 首屏静态区 ---------- */
  function renderHero() {
    var digest = state.digest;
    var podcast = digest.podcast || {};
    el('issue-badge').textContent = digest.issue && digest.issue.label
      ? digest.date + ' · ' + digest.issue.label
      : digest.date;
    el('podcast-title').textContent = podcast.title || ('AI 新闻快报 | ' + digest.date);
    el('podcast-subtitle').textContent = podcast.subtitle || '';
    el('podcast-duration').textContent = [
      podcast.duration_label || '--:--',
      podcast.segment_count ? podcast.segment_count + ' 段逐字稿' : '',
    ].filter(Boolean).join(' · ');
    el('player-title').textContent = podcast.title || '';
    el('lead-date').textContent = digest.date_label + ' · 编辑部按';
    el('lead-text').innerHTML = digest.lead ? escapeHtml(digest.lead.text) : placeholder('✨', '本期暂无导语');
    el('nav-sources').textContent = '🌐 来源速递 (' + ((digest.sources || {}).shown || 0) + ')';
    renderMetrics(digest.metrics || []);
    renderChapters(digest.chapters || []);
    renderVoiceButtons((podcast.hosts || []));
  }

  function renderMetrics(metrics) {
    el('metrics-strip').innerHTML = metrics.map(function (metric) {
      return '<div class="metric-item"><div class="metric-value">' + escapeHtml(metric.value) +
        '<span class="metric-unit">' + escapeHtml(metric.unit) + '</span></div>' +
        '<div class="metric-label">' + escapeHtml(metric.label) + '</div></div>';
    }).join('');
  }

  function renderChapters(chapters) {
    var wrap = el('chapters-row');
    if (!chapters.length) { wrap.innerHTML = ''; wrap.hidden = true; return; }
    wrap.hidden = false;
    wrap.innerHTML = '<span class="chapters-title">快速定位章节：</span>' + chapters.map(function (chapter) {
      return '<button class="chapter-chip" data-start="' + chapter.start + '">' +
        '<span>🎧</span> ' + escapeHtml(chapter.title) +
        ' <span class="chapter-time">' + escapeHtml(chapter.label) + '</span></button>';
    }).join('');
    wrap.querySelectorAll('.chapter-chip').forEach(function (chip) {
      chip.addEventListener('click', function () {
        state.player.seek(Number(chip.getAttribute('data-start')) + 0.01, true);
        showToast('已跳转至 ' + fmtTime(Number(chip.getAttribute('data-start'))));
      });
    });
  }

  function renderVoiceButtons(hosts) {
    var engine = state.player;
    hosts.forEach(function (host) {
      var container = el(host.key === 'A' ? 'host-a-voices' : 'host-b-voices');
      container.innerHTML = '';
      var selected = engine ? engine.voiceOf(host.key) : null;
      (host.variants || []).forEach(function (variant, index) {
        var isActive = selected ? selected === variant.id : index === 0;
        var button = document.createElement('button');
        button.className = 'voice-btn host-' + host.key.toLowerCase() + '-btn' +
          (isActive ? ' active' : '');
        button.textContent = variant.label;
        button.addEventListener('click', function () {
          engine.setVoice(host.key, variant.id);
          renderVoiceButtons(hosts);
          showToast('已切换 ' + host.name + ' 音色为：' + variant.label);
        });
        container.appendChild(button);
      });
      if (!host.variants || !host.variants.length) {
        container.innerHTML = placeholder('🎙️', '该主持人暂无可选音色');
      }
    });
  }

  /* ---------- 日报 ---------- */
  function renderReport(report) {
    var body = el('daily-report-body');
    var toc = el('daily-toc');
    if (!report) {
      body.innerHTML = placeholder('📰', '该日期暂无深度日报');
      toc.innerHTML = '';
      el('daily-meta').textContent = '';
      el('daily-counter').textContent = '';
      return;
    }
    el('daily-counter').textContent = '全文 ' + report.char_count + ' 字';
    el('daily-meta').textContent = '阅读时间：约 ' + report.reading_minutes + ' 分钟 · ' +
      (state.digest.date_label || '');
    var html = '';
    html += '<h1 class="daily-article-title">' + escapeHtml(report.title || '') + '</h1>';
    (report.deep_dives || []).forEach(function (dive, index) {
      html += '<div class="article-section-block" id="deep-dive-' + (index + 1) + '">' +
        '<div class="deep-dive-card">' +
        '<h3 class="deep-dive-title">' + escapeHtml(dive.heading) + '</h3>' +
        pointBlock('📌 事件背景', dive.background) +
        pointBlock('⚡ 核心突破点', dive.breakthrough) +
        pointBlock('🌐 对行业与用户的深远影响', dive.impact) +
        (dive.raw_sections || []).map(function (text) {
          return '<div class="deep-dive-point">' + escapeHtml(text) + '</div>';
        }).join('') +
        '</div></div>';
    });
    if ((report.frontier || []).length) {
      html += '<div class="article-section-block"><h2 class="article-section-header">⚡ 前沿情报</h2>' +
        '<div class="frontier-list">' + report.frontier.map(function (item) {
          return '<div class="frontier-item"><span class="frontier-bullet">🔹</span><div>' +
            '<strong>' + escapeHtml(item.title) + '</strong>：' + escapeHtml(item.summary) +
            '</div></div>';
        }).join('') + '</div></div>';
    }
    if (report.editor_verdict) {
      html += '<div class="article-section-block"><div class="editor-quote-card">' +
        '<div class="editor-quote-header"><span>💡</span> AI 小编深度点评</div>' +
        '<div class="editor-quote-body">' + escapeHtml(report.editor_verdict) + '</div>' +
        '</div></div>';
    }
    body.innerHTML = html;
    toc.innerHTML = (report.toc || []).map(function (item) {
      return '<a href="#' + escapeHtml(item.anchor) + '" class="daily-toc-link">' +
        escapeHtml(item.title.length > 12 ? item.title.slice(0, 12) + '…' : item.title) + '</a>';
    }).join('');
  }

  function pointBlock(label, text) {
    if (!text) return '';
    return '<div class="deep-dive-point"><strong>' + label + '：</strong>' +
      escapeHtml(text) + '</div>';
  }

  /* ---------- 来源 ---------- */
  function filteredSources() {
    var block = state.digest.sources || { items: [] };
    var query = state.sourceQuery.trim().toLowerCase();
    return (block.items || []).filter(function (item) {
      var groupOk = state.sourceGroup === 'all' || item.group === state.sourceGroup;
      var queryOk = !query ||
        (item.title || '').toLowerCase().indexOf(query) >= 0 ||
        (item.publisher || '').toLowerCase().indexOf(query) >= 0;
      return groupOk && queryOk;
    });
  }

  function renderSources() {
    var block = state.digest.sources || { items: [], groups: [] };
    if (!(block.items || []).length) {
      el('source-count-badge').textContent = '—';
      el('sources-grid').innerHTML = placeholder('🌐', '该日期暂无来源数据');
      el('sources-expand-wrap').hidden = true;
      el('source-filters').innerHTML = '';
      return;
    }
    el('source-count-badge').textContent = block.shown + ' 条条目已收录';
    el('source-filters').innerHTML =
      '<button class="source-filter-btn' + (state.sourceGroup === 'all' ? ' active' : '') +
      '" data-group="all">全部 (' + block.shown + ')</button>' +
      (block.groups || []).map(function (group) {
        return '<button class="source-filter-btn' +
          (state.sourceGroup === group.group ? ' active' : '') +
          '" data-group="' + escapeHtml(group.group) + '">' +
          escapeHtml(group.group) + ' (' + group.count + ')</button>';
      }).join('');
    el('source-filters').querySelectorAll('.source-filter-btn').forEach(function (button) {
      button.addEventListener('click', function () {
        state.sourceGroup = button.getAttribute('data-group');
        state.sourcesExpanded = false;
        renderSources();
      });
    });

    var items = filteredSources();
    var visible = state.sourcesExpanded ? items : items.slice(0, SOURCE_PREVIEW);
    el('sources-grid').innerHTML = visible.map(function (item) {
      return '<div class="source-card">' +
        '<div class="source-card-top"><span class="source-badge">' +
        escapeHtml(item.publisher) + '</span><span class="source-role-pill">' +
        escapeHtml(item.role_label) + '</span></div>' +
        '<a href="' + escapeHtml(item.link) + '" target="_blank" rel="noopener" ' +
        'class="source-title-link" title="' + escapeHtml(item.title) + '">' +
        escapeHtml(item.title) + '</a>' +
        '<div class="source-card-footer"><span>直达原文 ↗</span></div></div>';
    }).join('');
    var canExpand = items.length > SOURCE_PREVIEW;
    el('sources-expand-wrap').hidden = !canExpand;
    if (canExpand) {
      el('btn-expand-sources').textContent = state.sourcesExpanded
        ? '收起部分信源 ▴' : '展开全部 ' + items.length + ' 条信源 ▾';
    }
  }

  /* ---------- 雷达 ---------- */
  function renderRadar() {
    var radar = state.digest.radar || {};
    var root = el('radar-root');
    el('radar-badge').textContent = radar.available
      ? (radar.status === 'pick' ? '主推 ' + radar.pick.repo : '本期无主推项目')
      : '该期无雷达';
    if (radar.rules) {
      el('radar-rules').textContent = '门槛：≥ ' + radar.rules.min_stars + ' 星 · 近 ' +
        radar.rules.created_window_days + ' 天创建 · ' + radar.rules.recent_push_days + ' 天内活跃';
    }
    if (!radar.available) {
      root.innerHTML = placeholder('📡', '该日期暂无项目雷达');
      return;
    }
    if (radar.status === 'pick' && radar.pick) {
      var pick = radar.pick;
      var delta = pick.delta_stars == null ? '' : ' (+' + pick.delta_stars + '/日)';
      root.innerHTML = '<div class="radar-hit-view active"><div class="radar-hit-header">' +
        '<div class="radar-status-badge">● 今日主推开源项目 · 深度挖掘</div></div>' +
        '<div class="radar-hit-card"><div class="radar-hit-title">' +
        '<h3>' + escapeHtml(pick.repo) + '</h3>' +
        '<span class="source-badge">' + escapeHtml(pick.language || '—') + '</span>' +
        '<span class="radar-stars">★ ' + escapeHtml(pick.stars) + delta + '</span></div>' +
        '<p>' + escapeHtml(pick.readme_intro || pick.description) + '</p>' +
        '<div class="radar-hit-meta"><span>许可证：' + escapeHtml(pick.license || '—') +
        '</span><a href="' + escapeHtml(pick.url) + '" target="_blank" rel="noopener">' +
        '在 GitHub 上查阅仓库 ↗</a></div></div></div>';
      return;
    }
    var candidates = radar.candidates || {};
    var message = candidates.total
      ? '今日全网共扫描 <strong>' + candidates.total + ' 个候选仓库</strong>，' +
        (candidates.excluded_recent ? '其中 <strong>' + candidates.excluded_recent +
          ' 个</strong>为近 30 天已推荐或重复收录，' : '') +
        '经工程稳定性与代码审查，编辑部决定今日<strong>宁缺毋滥、轮空主推</strong>。'
      : '今日候选项不足，编辑部决定<strong>轮空主推</strong>。';
    root.innerHTML = '<div class="radar-empty-state-view"><div class="radar-visual-box">' +
      '<canvas id="radarCanvas" class="radar-sweep-canvas" width="180" height="180"></canvas>' +
      '</div><div class="radar-empty-content">' +
      '<span class="radar-status-badge"><span>●</span> 今日雷达状态：暂无主推项目</span>' +
      '<h3 class="radar-empty-headline">今日候选池经多轮筛查，未检出达标项目</h3>' +
      '<p class="radar-empty-paragraph">「AI 每日先锋」每日严选 1 个具备重大架构革新且近期 ' +
      'Star 增速稳定的开源项目。' + message + '</p>' +
      '</div></div>';
    startRadarSweep();
  }

  function startRadarSweep() {
    var canvas = el('radarCanvas');
    if (!canvas) return;
    var ctx = canvas.getContext('2d');
    var angle = 0;
    cancelAnimationFrame(radarLoop.raf);
    radarLoop.running = true;
    function draw() {
      if (!radarLoop.running) return;
      var w = canvas.width, h = canvas.height, cx = w / 2, cy = h / 2, r = cx - 6;
      ctx.clearRect(0, 0, w, h);
      ctx.strokeStyle = 'rgba(56,189,248,0.18)';
      ctx.lineWidth = 1;
      for (var rad = r / 4; rad <= r; rad += r / 4) {
        ctx.beginPath(); ctx.arc(cx, cy, rad, 0, Math.PI * 2); ctx.stroke();
      }
      ctx.beginPath();
      ctx.moveTo(cx, 0); ctx.lineTo(cx, h); ctx.moveTo(0, cy); ctx.lineTo(w, cy); ctx.stroke();
      ctx.save(); ctx.translate(cx, cy); ctx.rotate(angle);
      var grad = ctx.createRadialGradient(0, 0, 0, 0, 0, r);
      grad.addColorStop(0, 'rgba(56,189,248,0.4)');
      grad.addColorStop(1, 'rgba(56,189,248,0)');
      ctx.beginPath(); ctx.moveTo(0, 0); ctx.arc(0, 0, r, -0.4, 0); ctx.closePath();
      ctx.fillStyle = grad; ctx.fill();
      ctx.beginPath(); ctx.moveTo(0, 0); ctx.lineTo(r, 0);
      ctx.strokeStyle = '#38bdf8'; ctx.lineWidth = 2; ctx.stroke();
      ctx.restore();
      angle += 0.035;
      if (!radarLoop.reduceMotion) radarLoop.raf = requestAnimationFrame(draw);
    }
    draw();
  }

  /* ---------- 逐字稿抽屉 ---------- */
  function renderTranscript() {
    var drawer = el('drawer-chunk-list');
    if (!state.chunks.length) {
      drawer.innerHTML = placeholder('🎙️', '该日期暂无逐字稿');
      el('drawer-count').textContent = '';
      return;
    }
    el('drawer-count').textContent = state.chunks.length + ' 句对谈';
    var visible = state.chunks.filter(function (chunk) {
      return state.drawerFilter === 'all' || chunk.host === state.drawerFilter;
    });
    drawer.innerHTML = visible.map(function (chunk) {
      return '<div class="transcript-chunk-row" data-index="' +
        state.chunks.indexOf(chunk) + '">' +
        '<div class="chunk-header"><span class="chunk-speaker-badge host-' +
        (chunk.host === 'A' ? 'a' : 'b') + '">● ' + HOST_NAMES[chunk.host] +
        '</span><span class="chunk-time">' + fmtTime(chunk.start) + '</span></div>' +
        '<div class="chunk-text">' + escapeHtml(chunk.text) + '</div></div>';
    }).join('');
    drawer.querySelectorAll('.transcript-chunk-row').forEach(function (row) {
      row.addEventListener('click', function () {
        var chunk = state.chunks[Number(row.getAttribute('data-index'))];
        if (!chunk) return;
        state.player.seek(chunk.start + 0.01, true);
        highlightChunk(state.chunks.indexOf(chunk));
      });
    });
  }

  function highlightChunk(index) {
    document.querySelectorAll('.transcript-chunk-row').forEach(function (row) {
      row.classList.toggle('active', Number(row.getAttribute('data-index')) === index);
    });
    var chunk = state.chunks[index];
    if (!chunk) return;
    el('player-sentence').textContent = HOST_NAMES[chunk.host] + '：' + chunk.text;
    var badge = el('live-speaker-badge');
    badge.classList.toggle('host-b-speaking', chunk.host === 'B');
    badge.innerHTML = '<span class="badge-dot">●</span> 正在发言：' + HOST_NAMES[chunk.host];
  }

  function toggleDrawer(open) {
    var drawer = el('transcript-drawer');
    var shouldOpen = open === undefined ? !drawer.classList.contains('open') : open;
    drawer.classList.toggle('open', shouldOpen);
    el('btn-drawer-toggle').classList.toggle('active', shouldOpen);
  }

  /* ---------- 播放器 UI 绑定 ---------- */
  function bindPlayer() {
    var engine = state.player;
    engine.onState(function (playing) {
      var icon = playing ? '⏸' : '▶';
      el('hero-play-icon').textContent = icon;
      el('dock-play-icon').textContent = icon;
      el('hero-play-text').textContent = playing ? '暂停播放' : '立即收听本期';
      el('podcast-card').classList.toggle('is-playing', playing);
    });
    engine.onTime(function (current, duration) {
      el('time-current').textContent = fmtTime(current);
      el('time-duration').textContent = fmtTime(duration);
      el('scrubber-fill').style.width = (duration ? (current / duration) * 100 : 0) + '%';
      highlightChunk(engine.index);
    });
    document.querySelectorAll('#hero-play-btn, #dock-play-btn').forEach(function (button) {
      button.addEventListener('click', function () { engine.toggle(); });
    });
    el('skip-back').addEventListener('click', function () { engine.skip(-15); });
    el('skip-forward').addEventListener('click', function () { engine.skip(15); });
    el('scrubber-track').addEventListener('click', function (event) {
      var rect = event.currentTarget.getBoundingClientRect();
      engine.seek(((event.clientX - rect.left) / rect.width) * engine.duration, false);
    });
    el('speed-btn').addEventListener('click', function () {
      var label = engine.cycleRate();
      el('speed-btn').textContent = label;
      showToast('播放倍速：' + label);
    });
    el('vol-slider').addEventListener('input', function (event) {
      engine.setVolume(Number(event.target.value));
      el('mute-btn').textContent = Number(event.target.value) === 0 ? '🔇' : '🔊';
    });
    el('mute-btn').addEventListener('click', function () {
      var muted = engine.toggleMute();
      el('mute-btn').textContent = muted ? '🔇' : '🔊';
      el('vol-slider').value = muted ? 0 : engine.volume;
    });
    el('btn-drawer-toggle').addEventListener('click', function () { toggleDrawer(); });
    el('hero-transcript-btn').addEventListener('click', function () { toggleDrawer(true); });
    el('drawer-close').addEventListener('click', function () { toggleDrawer(false); });
    document.querySelectorAll('.btn-drawer-filter').forEach(function (button) {
      button.addEventListener('click', function () {
        document.querySelectorAll('.btn-drawer-filter').forEach(function (other) {
          other.classList.remove('active');
        });
        button.classList.add('active');
        state.drawerFilter = button.getAttribute('data-speaker');
        renderTranscript();
      });
    });
    el('source-search').addEventListener('input', function (event) {
      state.sourceQuery = event.target.value;
      state.sourcesExpanded = false;
      renderSources();
    });
    el('btn-expand-sources').addEventListener('click', function () {
      state.sourcesExpanded = !state.sourcesExpanded;
      renderSources();
    });
    document.addEventListener('keydown', function (event) {
      if (event.key === 'Escape') toggleDrawer(false);
    });
    el('date-prev').addEventListener('click', function () { stepDate(1); });
    el('date-next').addEventListener('click', function () { stepDate(-1); });
    el('date-latest').addEventListener('click', function () { loadDate(DATES[0]); });
  }

  function stepDate(direction) {
    var index = DATES.indexOf(state.date);
    var next = DATES[index + direction];
    if (next) loadDate(next);
  }

  /* ---------- 日期加载主流程 ---------- */
  async function loadDate(date) {
    state.date = date;
    renderDateAxis();
    var body = el('daily-report-body');
    body.classList.add('is-loading');
    try {
      var results = await Promise.all([
        fetchJson('./data/digest_' + date + '.json'),
        fetchJson('./episodes/' + date + '/playlist.json').catch(function () { return { chunks: [] }; }),
      ]);
      state.digest = results[0];
      state.chunks = (results[1].chunks || []);
    } catch (error) {
      console.warn('digest load failed', date, error);
      state.digest = {
        schema_version: 1, date: date, date_label: date,
        issue: { number: null, label: null, source: 'unavailable' },
        availability: {}, lead: null, metrics: [], chapters: [],
        report: null, sources: { items: [], groups: [], shown: 0 },
        radar: { available: false, status: 'unavailable', rules: null,
                 pick: null, runner_ups: [], candidates: {}, message: '' },
        podcast: { title: 'AI 新闻快报 | ' + date, subtitle: null, duration_label: '--:--',
                   segment_count: 0, hosts: [] },
      };
      state.chunks = [];
      showToast('该日期数据暂不可用');
    } finally {
      body.classList.remove('is-loading');
    }
    renderHero();
    renderReport(state.digest.report);
    renderSources();
    renderRadar();
    renderTranscript();
    state.player.load(state.chunks, date);
    ((state.digest.podcast || {}).hosts || []).forEach(function (host) {
      if (!state.player.voiceOf(host.key) && (host.variants || []).length) {
        state.player.setVoice(host.key, host.variants[0].id);
      }
    });
    renderVoiceButtons(((state.digest.podcast || {}).hosts || []));
    el('player-sentence').textContent = '点击播放开始收听…';
    el('live-speaker-badge').innerHTML = '<span class="badge-dot">●</span> 待播放';
  }

  async function boot() {
    state.player = new window.PodcastEngine(el('main-audio'));
    bindPlayer();
    radarLoop.reduceMotion = !!(window.matchMedia &&
      window.matchMedia('(prefers-reduced-motion: reduce)').matches);
    document.addEventListener('visibilitychange', function () {
      if (document.hidden) {
        radarLoop.running = false;
        cancelAnimationFrame(radarLoop.raf);
      } else if (el('radarCanvas') && !radarLoop.reduceMotion) {
        startRadarSweep();
      }
    });
    try {
      state.calendar = await fetchJson('./data/calendar.json');
    } catch (error) {
      state.calendar = { dates: DATES.map(function (date) {
        return { date: date, issue_number: null };
      }) };
    }
    var initial = DATES.indexOf(location.hash.replace('#', '')) >= 0
      ? location.hash.replace('#', '') : DATES[0];
    if (!initial) {
      el('date-pills').innerHTML = '';
      el('daily-report-body').innerHTML = placeholder('📰', '暂无任何已发布期数');
      return;
    }
    await loadDate(initial);
  }

  document.addEventListener('DOMContentLoaded', boot);
})();
