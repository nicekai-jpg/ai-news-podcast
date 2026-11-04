/* ============================================================
   数据装配：从 digest JSON 映射为设计稿 v23 的注入字段
   ============================================================ */
    const SITE_DATA = (function () {
      try {
        var node = document.getElementById('site-data');
        return node ? JSON.parse(node.textContent) : { dates: [] };
      } catch (e) {
        return { dates: [] };
      }
    })();

    let ISSUE_ORDER = Array.isArray(SITE_DATA.dates) ? SITE_DATA.dates.slice() : [];
    let DATA = {};
    let PLAYLIST_CHUNKS = [];
    let audioBroken = false;

    const SOURCE_BADGE_KEYS = {
      'OpenAI': 'openai',
      'Google': 'google',
      'NVIDIA': 'nvidia',
      'arXiv 论文': 'arxiv',
      'Hacker News': 'industry',
      '科技媒体': 'industry'
    };
    const POINT_LABELS = ['事件背景', '核心突破', '深远影响'];

    async function fetchJson(url) {
      const response = await fetch(url, { cache: 'no-cache' });
      if (!response.ok) throw new Error('HTTP ' + response.status);
      return response.json();
    }

    function clockText(seconds) {
      const value = Math.max(0, Math.round(Number(seconds) || 0));
      const mm = String(Math.floor(value / 60)).padStart(2, '0');
      const ss = String(value % 60).padStart(2, '0');
      return mm + ':' + ss;
    }

    function cleanReportTitle(title) {
      return String(title || '')
        .replace(/^[✨\s]+/, '')
        .replace(/\s*\|\s*/g, ' · ')
        .trim();
    }

    function cleanHeading(heading) {
      return String(heading || '')
        .replace(/^[一二三四五六七八九十]+、\s*/, '')
        .replace(/^[0-9]+[.、]\s*/, '')
        .trim();
    }

    function chapterTag(heading) {
      const text = cleanHeading(heading);
      const match = text.match(/^[A-Za-z][A-Za-z0-9 .&+·_-]{1,18}/);
      if (match) return match[0].trim();
      return text.slice(0, 6);
    }

    function chapterTopic(heading) {
      const text = cleanHeading(heading);
      let rest = text.slice(chapterTag(heading).length).replace(/^[\s，、：:·\-—]+/, '');
      const cut = rest.search(/[，。！？；：,.:;!?]/);
      if (cut > 0) rest = rest.slice(0, cut);
      return rest.length > 12 ? rest.slice(0, 12) + '…' : rest;
    }

    function shortTopic(heading) {
      const text = cleanHeading(heading);
      const cut = text.search(/[，。！？；：,.:;!?]/);
      const label = cut > 0 ? text.slice(0, cut) : text;
      return label.length > 10 ? label.slice(0, 10) + '…' : label;
    }

    function formatStars(stars) {
      const value = Number(stars);
      if (!isFinite(value)) return String(stars || '');
      if (value >= 1000) return (value / 1000).toFixed(1).replace(/\.0$/, '') + 'k';
      return String(value);
    }

    function digestToData(digest, playlist) {
      const d = digest || {};
      const date = String(d.date || '');
      const podcast = d.podcast || {};
      const hosts = podcast.hosts || [];
      const chapters = d.chapters || [];
      const report = d.report || {};
      const dives = (report.deep_dives || []).slice(0, 4);
      const editorName = (hosts[0] && hosts[0].name) || '林夏';
      const data = {
        issue_number: date,
        issue_date: date,
        issue_tagline: 'AI 前沿 · 每日一刊',
        demo_note: '',
        podcast_title: podcast.title || ('AI 新闻快报 | ' + date),
        podcast_subtitle: podcast.subtitle || '双主持对谈',
        player_subtitle: '今日AI看点 · ' + date,
        podcast_duration: podcast.duration_label || '--:--',
        podcast_audio_url: podcast.mp3_url || ('./episodes/' + date + '.mp3'),
        podcast_chapter_count: String(chapters.length || dives.length || 0),
        host_a_name: (hosts[0] && hosts[0].name) || '林夏',
        host_b_name: (hosts[1] && hosts[1].name) || '陈默',
        lead_quote: (d.lead && d.lead.text) || '',
        report_word_count: String(report.char_count || 0),
        report_read_minutes: String(report.reading_minutes || 0),
        report_pill: '深度研判',
        report_title: cleanReportTitle(report.title || ''),
        editor_quote: report.editor_verdict || (d.lead && d.lead.text) || '',
        editor_name: editorName,
        editor_role: '主编'
      };

      const metrics = d.metrics || [];
      ['sources', 'hours', 'speedup', 'items'].forEach(function (key, index) {
        const metric = metrics[index] || {};
        data['metric_' + key + '_value'] = metric.value != null ? String(metric.value) : '';
        data['metric_' + key + '_unit'] = metric.unit || '';
        data['metric_' + key + '_label'] = metric.label || '';
      });

      for (let n = 1; n <= 4; n += 1) {
        const dive = dives[n - 1] || {};
        const title = cleanHeading(dive.heading || '');
        const points = [
          dive.background || '',
          dive.breakthrough || '',
          dive.impact || ''
        ];
        if ((dive.raw_sections || []).length) {
          points[2] = (points[2] ? points[2] + ' ' : '') + dive.raw_sections.join(' ');
        }
        data['chapter_' + n + '_title'] = title;
        data['chapter_' + n + '_tag'] = title ? chapterTag(title) : '';
        data['chapter_' + n + '_topic'] = title ? chapterTopic(title) : '';
        for (let m = 1; m <= 3; m += 1) {
          data['chapter_' + n + '_point_' + m + '_label'] = points[m - 1] ? POINT_LABELS[m - 1] : '';
          data['chapter_' + n + '_point_' + m + '_text'] = points[m - 1] || '';
        }
      }

      const sourceItems = ((d.sources || {}).items || []).slice(0, 30);
      data.sources_list = JSON.stringify(sourceItems.map(function (item) {
        return {
          badge: SOURCE_BADGE_KEYS[item.group] || 'industry',
          badge_label: item.publisher || item.source_name || '',
          meta: String(item.published_at || '').slice(0, 10) || date,
          title: item.title || '',
          summary: '',
          url: item.link || ''
        };
      }));

      const radar = d.radar || {};
      const candidates = radar.candidates || {};
      const pick = radar.pick || null;
      data.radar_status_label = pick ? '有主推项目' : '无主推项目';
      data.radar_headline = pick
        ? '今日候选池命中达标项目'
        : '今日候选池经多轮筛查，未检出达标项目';
      data.radar_paragraph = pick
        ? '本栏目每日严选 1 个具备重大架构革新且增速稳定的开源项目。今日主推仓库经工程稳定性、重复收录排查与实际代码审查后入选。'
        : '本栏目每日严选 1 个具备重大架构革新且近 7 日 Star 增速稳定突破 300/天的优质开源项目。今日全网扫描了一批候选仓库，经工程稳定性、重复收录排查及实际代码审查，均处于评估观察期或版本迁移期，编辑部决定今日宁缺毋滥、轮空主推。';
      data.radar_reason = pick ? '编辑部一致通过' : '宁缺毋滥、轮空主推';
      data.candidate_count = String(candidates.total != null ? candidates.total : (radar.runner_ups || []).length || 0);
      data.excluded_recent = String(candidates.excluded_recent != null ? candidates.excluded_recent : 0);
      data.radar_pick_repo = pick ? String(pick.repo || '') : '';
      data.radar_pick_lang = pick ? String(pick.language || '') : '';
      data.radar_pick_stars = pick ? formatStars(pick.stars) : '';
      data.radar_pick_growth = pick ? String(pick.delta_stars == null ? '' : pick.delta_stars) : '';
      data.radar_pick_license = pick ? String(pick.license || '') : '';
      data.radar_pick_desc = pick ? String(pick.readme_intro || pick.description || '') : '';
      data.radar_pick_url = pick ? String(pick.url || '') : '';

      const chunks = (playlist && playlist.chunks) || [];
      data.transcript_list = JSON.stringify(chunks.map(function (chunk) {
        return {
          speaker: chunk.host || 'A',
          time: clockText(chunk.start),
          text: chunk.text || ''
        };
      }));

      return data;
    }

    function applyPostHydrateChrome() {
      const stamp = document.querySelector('.stamp-date');
      if (stamp) stamp.textContent = (DATA.issue_date || '') + ' · 今日AI看点';
    }

    function applyConditionalBlocks() {
      for (let n = 1; n <= 4; n += 1) {
        const hasTitle = !!(DATA['chapter_' + n + '_title'] || '').trim();
        const block = document.getElementById('deep-dive-' + n);
        if (block) block.hidden = !hasTitle;
        const link = document.querySelectorAll('.broadside-toc-link')[n - 1];
        if (link) link.hidden = !hasTitle;
      }
      document.querySelectorAll('.metric-item').forEach(function (item) {
        const value = item.querySelector('.metric-value');
        const text = value ? value.textContent.trim() : '';
        item.hidden = !text;
      });
    }


    function toggleTheme() {
      const html = document.documentElement;
      const current = html.getAttribute('data-theme') || 'light';
      const next = current === 'dark' ? 'light' : 'dark';
      html.setAttribute('data-theme', next);
      localStorage.setItem('theme', next);
      updateThemeUI(next);
    }
    function updateThemeUI(theme) {
      const label = document.getElementById('theme-label');
      const icon = document.getElementById('theme-icon');
      if (label) label.textContent = theme === 'dark' ? '浅色' : '深色';
      if (icon) {
        /* 图标表示「点击后切到哪套主题」：深色时显示太阳，浅色时显示月亮 */
        icon.innerHTML = theme === 'dark'
          ? '<circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/>'
          : '<path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>';
      }
    }
    /* update on load */
    (function() { updateThemeUI(document.documentElement.getAttribute('data-theme') || 'light'); })();
    let isPlaying = false;
    let sourcesExpanded = false;
    let sourceFilter = 'all';
    let playTimer = null;
    let currentTime = 0;
    let audioEl = null;
    let audioDuration = 0;

    const PLAY_ICON = '<svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"/></svg>';
    const PAUSE_ICON = '<svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg>';

    /* 有 podcast_audio_url 时走真实音频；没有时退化为模拟时钟，保证时间码与前后 15 秒可感知 */
    function audioUrl() {
      const u = DATA.podcast_audio_url;
      if (typeof u !== 'string') return '';
      const t = u.trim();
      return (!t || t.startsWith('{{')) ? '' : t;
    }
    function hasAudio() { return !!audioUrl() && !audioBroken; }

    function ensureAudio() {
      const url = audioUrl();
      if (!url || audioBroken) return null;
      if (audioEl && audioEl.dataset.src === url) return audioEl;
      if (audioEl) audioEl.pause();
      audioEl = new Audio(url);
      audioEl.dataset.src = url;
      audioEl.preload = 'metadata';
      audioEl.addEventListener('loadedmetadata', function() {
        if (isFinite(audioEl.duration)) { audioDuration = audioEl.duration; renderClock(); }
      });
      audioEl.addEventListener('timeupdate', function() {
        if (!audioEl) return;
        currentTime = audioEl.currentTime;
        renderClock();
      });
      audioEl.addEventListener('ended', function() { setPlaying(false); });
      audioEl.addEventListener('error', function() {
        /* 音频缺失（本地预览 / 网络异常）时退化为模拟时钟，播放按钮仍可用 */
        const wasPlaying = isPlaying;
        audioBroken = true;
        audioEl = null;
        if (wasPlaying) startClock();
      });
      return audioEl;
    }

    function durationSeconds() {
      if (hasAudio()) {
        if (audioDuration) return audioDuration;
        if (audioEl && isFinite(audioEl.duration)) return audioEl.duration;
      }
      const parts = String(DATA.podcast_duration || '').split(':').map(Number);
      return parts.length === 2 && parts.every(n => !isNaN(n)) ? parts[0] * 60 + parts[1] : 0;
    }
    function formatClock(sec) {
      if (!isFinite(sec)) return '00:00';
      const s = Math.max(0, Math.round(sec));
      return String(Math.floor(s / 60)).padStart(2, '0') + ':' + String(s % 60).padStart(2, '0');
    }
    function renderClock() {
      const cur = document.getElementById('player-current');
      if (cur) cur.textContent = formatClock(currentTime);
      const total = document.getElementById('player-total');
      if (total) {
        const totalSec = durationSeconds();
        total.textContent = totalSec ? formatClock(totalSec) : (DATA.podcast_duration || '00:00');
      }
    }
    function seek(delta) {
      if (audioEl && isFinite(audioEl.duration)) {
        audioEl.currentTime = Math.max(0, Math.min(audioEl.currentTime + delta, audioEl.duration));
        currentTime = audioEl.currentTime;
      } else {
        const total = durationSeconds();
        currentTime = Math.max(0, Math.min(currentTime + delta, total || currentTime));
      }
      renderClock();
    }
    function stopClock() {
      if (playTimer) { clearInterval(playTimer); playTimer = null; }
    }
    function startClock() {
      stopClock();
      playTimer = setInterval(function() {
        const total = durationSeconds();
        if (total && currentTime >= total) { setPlaying(false); return; }
        currentTime = total ? Math.min(currentTime + 1, total) : currentTime + 1;
        renderClock();
      }, 1000);
    }

    /* 首屏与底部播放器共用一个播放状态，任何一处点击后两处按钮同步 */
    function togglePlay() { setPlaying(!isPlaying); }

    function setPlaying(next) {
      isPlaying = next;

      document.querySelectorAll('.podcast-disc, .player-disc')
        .forEach(d => d.classList.toggle('is-playing', isPlaying));

      const heroBtn = document.querySelector('.btn-play-primary');
      if (heroBtn) {
        heroBtn.innerHTML = isPlaying ? PAUSE_ICON + ' 暂停' : PLAY_ICON + ' 播放本期';
        heroBtn.setAttribute('aria-pressed', String(isPlaying));
      }

      const dockBtn = document.querySelector('.player-btn.play');
      if (dockBtn) {
        dockBtn.innerHTML = isPlaying ? PAUSE_ICON : PLAY_ICON;
        dockBtn.setAttribute('aria-label', isPlaying ? '暂停' : '播放');
        dockBtn.setAttribute('aria-pressed', String(isPlaying));
      }

      if (isPlaying) {
        const a = ensureAudio();
        if (a) {
          stopClock();
          if (currentTime) a.currentTime = Math.min(currentTime, isFinite(a.duration) ? a.duration : currentTime);
          const p = a.play();
          if (p && p.catch) p.catch(function(error) {
            if (error && error.name === 'NotAllowedError') {
              /* 自动播放被拦截时保持暂停态 */
              setPlaying(false);
              return;
            }
            /* 音频缺失（本地预览 / CDN 异常）→ 模拟时钟，保证可试听交互 */
            audioBroken = true;
            audioEl = null;
            if (isPlaying) startClock();
          });
        } else {
          startClock();
        }
      } else {
        if (audioEl) audioEl.pause();
        stopClock();
      }
    }

    let drawerReturnFocus = null;
    function openDrawer() {
      drawerReturnFocus = document.activeElement;
      document.getElementById('transcript-drawer').classList.add('open');
      document.getElementById('drawer-overlay').classList.add('visible');
      document.body.style.overflow = 'hidden';
      const closeBtn = document.querySelector('.drawer-close');
      if (closeBtn) closeBtn.focus();
    }
    function closeDrawer() {
      document.getElementById('transcript-drawer').classList.remove('open');
      document.getElementById('drawer-overlay').classList.remove('visible');
      document.body.style.overflow = '';
      if (drawerReturnFocus && typeof drawerReturnFocus.focus === 'function') {
        drawerReturnFocus.focus();
      }
      drawerReturnFocus = null;
    }
    document.addEventListener('keydown', function(e) {
      if (e.key === 'Escape') closeDrawer();
    });

    function toggleExpandSources() {
      sourcesExpanded = !sourcesExpanded;
      renderSources();
    }

    /* ISSUE_ORDER 为「新 → 旧」排列：上一期 = 更旧（索引 +1），下一期 = 更新（索引 -1） */
    function stepIssue(delta) {
      const idx = ISSUE_ORDER.indexOf(DATA.issue_number);
      const next = idx + delta;
      if (idx < 0 || next < 0 || next >= ISSUE_ORDER.length) return;
      switchIssue(ISSUE_ORDER[next]);
    }

    function updateDateNav() {
      const idx = ISSUE_ORDER.indexOf(DATA.issue_number);
      const prevBtn = document.getElementById('btn-prev-issue');
      const nextBtn = document.getElementById('btn-next-issue');
      if (prevBtn) prevBtn.disabled = idx >= ISSUE_ORDER.length - 1;
      if (nextBtn) nextBtn.disabled = idx <= 0;
    }

    async function loadIssue(id) {
      if (!id || ISSUE_ORDER.indexOf(id) === -1) return;
      if (DATA.issue_date === id) return;
      let digest;
      let playlist = null;
      try {
        digest = await fetchJson('./data/digest_' + id + '.json');
      } catch (error) {
        console.warn('digest load failed', id, error);
        return;
      }
      try {
        playlist = await fetchJson('./episodes/' + id + '/playlist.json');
      } catch (error) {
        playlist = { chunks: [] };
      }
      DATA = digestToData(digest, playlist);
      PLAYLIST_CHUNKS = (playlist && playlist.chunks) || [];
      sourcesExpanded = false;
      sourceFilter = 'all';
      currentTime = 0;
      audioBroken = false;
      if (audioEl) { audioEl.pause(); audioEl.src = ''; audioEl = null; }
      audioDuration = 0;
      setPlaying(false);
      renderClock();
      hydrate();
      applyPostHydrateChrome();
      applyConditionalBlocks();
      renderDatePills();
      updateDateNav();
      renderHosts();
      renderSourceFilters();
      renderSources();
      renderTranscript();
      renderRadar();
      const reduceMotion = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
      window.scrollTo({ top: 0, behavior: reduceMotion ? 'auto' : 'smooth' });
    }

    function switchIssue(id) { return loadIssue(id); }

    /* Render date pills — 期号列表由 ISSUE_ORDER 驱动 */
    function renderDatePills() {
      const container = document.getElementById('date-pills');
      if (!container) return;
      container.innerHTML = ISSUE_ORDER.map(function (id) {
        const label = String(id).slice(5);
        const active = id === DATA.issue_date;
        return '<button class="date-pill' + (active ? ' active' : '') + '"' +
          ' onclick="selectIssue(\'' + id + '\')" aria-pressed="' + active + '">' + label + '</button>';
      }).join('');
    }

    const SOURCE_ORIGINS = {
      'Google AI Blog': 'https://blog.google/technology/ai/',
      'OpenAI Blog': 'https://openai.com/news/',
      'Meta AI Blog': 'https://ai.meta.com/blog/',
      'NVIDIA 官方博客': 'https://blogs.nvidia.com/',
      'Hacker News': 'https://news.ycombinator.com/',
      'TechCrunch': 'https://techcrunch.com/',
      'arXiv': 'https://arxiv.org/',
      google: 'https://blog.google/technology/ai/',
      openai: 'https://openai.com/news/',
      meta: 'https://ai.meta.com/blog/',
      arxiv: 'https://arxiv.org/',
      nvidia: 'https://blogs.nvidia.com/',
      anthropic: 'https://www.anthropic.com/news'
    };

    function selectIssue(id) { switchIssue(id); }

    const BADGE_LABELS = {
      openai: 'OpenAI',
      google: 'Google',
      meta: 'Meta',
      arxiv: 'arXiv',
      anthropic: 'Anthropic',
      nvidia: 'NVIDIA',
      industry: '行业媒体'
    };

    function filteredSources() {
      const all = safeParse(DATA.sources_list, []);
      return sourceFilter === 'all'
        ? all
        : all.filter(s => (s.badge || 'industry') === sourceFilter);
    }

    /* 来源分类统计 + 筛选 */
    function renderSourceFilters() {
      const bar = document.getElementById('sources-filter');
      if (!bar) return;
      const all = safeParse(DATA.sources_list, []);
      const counts = {};
      all.forEach(s => {
        const key = s.badge || 'industry';
        counts[key] = (counts[key] || 0) + 1;
      });
      const chips = [{ key: 'all', label: '全部', count: all.length }].concat(
        Object.keys(counts)
          .sort((a, b) => counts[b] - counts[a])
          .map(key => ({ key, label: BADGE_LABELS[key] || key, count: counts[key] }))
      );
      bar.innerHTML = chips.map(c =>
        '<button type="button" class="source-filter-chip' + (sourceFilter === c.key ? ' active' : '') + '"' +
        ' onclick="setSourceFilter(\'' + c.key + '\')" aria-pressed="' + (sourceFilter === c.key) + '">' +
        c.label + '<span class="chip-count">' + c.count + '</span></button>'
      ).join('');
    }

    function setSourceFilter(key) {
      sourceFilter = key;
      sourcesExpanded = false;
      renderSources();
      renderSourceFilters();
    }

    /* Render sources — 有 url 用 url，否则回落到来源站点首页 */
    function renderSources() {
      const grid = document.getElementById('sources-grid');
      if (!grid) return;
      const sources = filteredSources();
      grid.innerHTML = sources.map((s, i) => {
        const hidden = (i >= 6 && !sourcesExpanded) ? ' style="display:none"' : '';
        const href = s.url || SOURCE_ORIGINS[s.badge_label] || SOURCE_ORIGINS[s.badge] || '';
        const title = s.title || '';
        const summary = s.summary || '';
        const inner =
          '<div class="source-card-header">' +
          '<span class="source-badge ' + (s.badge || 'industry') + '">' + (s.badge_label || '') + '</span>' +
          '<span class="source-meta">' + (s.meta || '') + '</span>' +
          '</div>' +
          '<div class="source-title">' + title + '</div>' +
          (summary && summary !== title ? '<div class="source-summary">' + summary + '</div>' : '');
        if (!href) return '<article class="source-card"' + hidden + '>' + inner + '</article>';
        return '<a class="source-card" href="' + href + '" target="_blank" rel="noopener"' + hidden + '>' + inner + '</a>';
      }).join('');

      const btn = document.getElementById('btn-expand-sources');
      if (btn) btn.textContent = sourcesExpanded ? '收起信源' : '展开全部';

      const label = document.getElementById('sources-count-label');
      if (label) label.textContent = sources.length + ' 条信源';
    }

    /* 雷达：有主推数据时渲染主推卡，否则保留轮空态 */
    function renderRadar() {
      const empty = document.getElementById('radar-empty');
      const pick = document.getElementById('radar-pick');
      if (!empty || !pick) return;
      const hasPick = !!(DATA.radar_pick_repo && String(DATA.radar_pick_repo).trim());
      empty.hidden = hasPick;
      pick.hidden = !hasPick;
      if (hasPick) stopRadar(); else startRadar();
    }

    /* 主持人由数据驱动，roster 与逐字稿头像共用同一份姓名 */
    function hostInitial(name) {
      return String(name || '').trim().slice(0, 1) || '·';
    }
    function renderHosts() {
      const list = document.getElementById('host-roster-list');
      if (!list) return;
      list.innerHTML = [
        { cls: 'host-a', name: DATA.host_a_name },
        { cls: 'host-b', name: DATA.host_b_name }
      ].filter(h => h.name && String(h.name).trim()).map(h =>
        '<span class="host-identity">' +
        '<span class="host-avatar ' + h.cls + '">' + hostInitial(h.name) + '</span>' +
        '<span class="host-name">' + h.name + '</span>' +
        '</span>'
      ).join('');
    }

    /* Render transcript */
    function renderTranscript() {
      const body = document.getElementById('transcript-body');
      if (!body) return;
      const lines = safeParse(DATA.transcript_list, []);
      body.innerHTML = lines.map(l => {
        const isA = l.speaker === 'A';
        const name = isA ? DATA.host_a_name : DATA.host_b_name;
        return '<div class="transcript-line">' +
          '<span class="transcript-speaker ' + (isA ? 'host-a' : 'host-b') + '">' + hostInitial(name) + '</span>' +
          '<div class="transcript-text"><span class="timestamp">' + (l.time || '') + '</span>' + (l.text || '') + '</div>' +
          '</div>';
      }).join('');
    }

    /* Radar canvas animation */
    let radarRAF = null;
    let radarAngle = 0;
    let radarCanvas = null;
    let radarCtx = null;

    function radarReducedMotion() {
      return window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    }

    function drawRadarFrame() {
      if (!radarCtx || !radarCanvas) return;
      const ctx = radarCtx;
      const canvas = radarCanvas;
      const cx = canvas.width / 2, cy = canvas.height / 2;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      // rings
      for (let r = 20; r <= 60; r += 20) {
        ctx.beginPath();
        ctx.arc(cx, cy, r, 0, Math.PI * 2);
        ctx.strokeStyle = 'rgba(212, 165, 116, 0.15)';
        ctx.lineWidth = 1;
        ctx.stroke();
      }
      // cross
      ctx.strokeStyle = crossColor;
      ctx.beginPath();
      ctx.moveTo(cx, 0); ctx.lineTo(cx, canvas.height);
      ctx.moveTo(0, cy); ctx.lineTo(canvas.width, cy);
      ctx.stroke();
      // sweep
      const grad = ctx.createConicGradient(radarAngle, cx, cy);
      grad.addColorStop(0, 'rgba(217, 74, 43, 0.35)');
      grad.addColorStop(0.15, 'rgba(217, 74, 43, 0)');
      grad.addColorStop(1, 'rgba(217, 74, 43, 0)');
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.arc(cx, cy, 60, radarAngle - 0.5, radarAngle);
      ctx.closePath();
      ctx.fill();
      // center dot
      ctx.beginPath();
      ctx.arc(cx, cy, 3, 0, Math.PI * 2);
      ctx.fillStyle = '#d94a2b';
      ctx.fill();
      radarAngle += 0.02;
    }

    function radarTick() {
      drawRadarFrame();
      radarRAF = requestAnimationFrame(radarTick);
    }

    function startRadar() {
      if (!radarCanvas || radarRAF) return;
      if (radarReducedMotion()) { drawRadarFrame(); return; }
      radarRAF = requestAnimationFrame(radarTick);
    }

    function stopRadar() {
      if (radarRAF) { cancelAnimationFrame(radarRAF); radarRAF = null; }
    }

    function initRadar() {
      radarCanvas = document.getElementById('radarCanvas');
      if (!radarCanvas) return;
      radarCtx = radarCanvas.getContext('2d');
      /* 主推态时轮空画布是隐藏的，不启动动画 */
      const empty = document.getElementById('radar-empty');
      if (empty && empty.hidden) return;
      startRadar();
    }

    function safeParse(str, fallback) {
      if (typeof str !== 'string') return (str == null ? fallback : str);
      const s = str.trim();
      if (!s || s.startsWith('{{')) return fallback;
      try { return JSON.parse(s); } catch (e) { return fallback; }
    }


    /* Hydrate {{...}} placeholders in DOM with DATA values (preview mode)
       首次运行时缓存模板文本，切换期号后可重复渲染 */
    let titleTemplate = null;

    function fieldValue(key) {
      const v = DATA[key];
      if (v == null) return null;
      const text = String(v);
      /* 空字符串表示「这一项确实没有」，渲染为空；未注入的模板占位符保持原样 */
      if (text.trim().startsWith('{{')) return null;
      return text;
    }

    function hydrate() {
      const roots = [document.head, document.body];
      roots.forEach(root => {
        if (!root) return;
        const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, null, false);
        const nodes = [];
        while (walker.nextNode()) nodes.push(walker.currentNode);
        nodes.forEach(node => {
          if (node.__tpl == null) {
            if (!node.nodeValue || node.nodeValue.indexOf('{{') === -1) return;
            node.__tpl = node.nodeValue;
          }
          node.nodeValue = node.__tpl.replace(/\{\{\s*([a-zA-Z0-9_]+)\s*\}\}/g, (m, key) => {
            const v = fieldValue(key);
            return v === null ? m : v;
          });
        });
      });

      document.querySelectorAll('*').forEach(el => {
        if (el.__attrTpl == null) {
          const tpl = {};
          for (const attr of Array.from(el.attributes)) {
            if (attr.value && attr.value.indexOf('{{') !== -1) tpl[attr.name] = attr.value;
          }
          if (!Object.keys(tpl).length) return;
          el.__attrTpl = tpl;
        }
        Object.keys(el.__attrTpl).forEach(name => {
          el.setAttribute(name, el.__attrTpl[name].replace(/\{\{\s*([a-zA-Z0-9_]+)\s*\}\}/g, (m, key) => {
            const v = fieldValue(key);
            return v === null ? m : v;
          }));
        });
      });

      if (titleTemplate === null && document.title.indexOf('{{') !== -1) {
        titleTemplate = document.title;
      }
      if (titleTemplate) {
        document.title = titleTemplate.replace(/\{\{\s*([a-zA-Z0-9_]+)\s*\}\}/g, (m, key) => {
          const v = fieldValue(key);
          return v === null ? m : v;
        });
      }

      /* 可选内容块：data-requires 里的字段全部为空时隐藏，避免残留占位符或空行 */
      document.querySelectorAll('[data-requires]').forEach(el => {
        const keys = el.getAttribute('data-requires').split(',');
        const hasContent = keys.some(k => {
          const v = fieldValue(k.trim());
          return v !== null && v.trim() !== '';
        });
        el.hidden = !hasContent;
      });
    }

    /* Init：从 site-data 的日期列表启动，默认加载最新一期 */
    document.addEventListener('DOMContentLoaded', async function () {
      try {
        const saved = localStorage.getItem('theme');
        if (saved === 'dark' || saved === 'light') {
          document.documentElement.setAttribute('data-theme', saved);
        }
        updateThemeUI(document.documentElement.getAttribute('data-theme') || 'light');
        hydrate();
        applyPostHydrateChrome();
        applyConditionalBlocks();
        const hashDate = location.hash.replace('#', '');
        const initial = ISSUE_ORDER.indexOf(hashDate) >= 0 ? hashDate : ISSUE_ORDER[0];
        if (initial) {
          await loadIssue(initial);
        } else {
          renderDatePills();
          updateDateNav();
          renderHosts();
          renderSourceFilters();
          renderSources();
          renderTranscript();
          renderRadar();
        }
      } finally {
        initRadar();
        renderClock();
      }
    });
