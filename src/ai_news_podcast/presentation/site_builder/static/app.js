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
    let dataReady = false;
    let PLAYLIST_CHUNKS = [];
    let audioBroken = false;

    /* 所有动态文本进入 innerHTML 前必须转义：信源标题/摘要来自外部 RSS，属不可信输入 */
    function escapeHtml(value) {
      return String(value == null ? '' : value)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
    }

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
        /* 报告 H1 的日期是零填充（2026年11月05日），统一成 2026年11月5日 */
        .replace(/(\d{4})年0?(\d{1,2})月0?(\d{1,2})日/, function (m, y, mo, d) {
          return y + '年' + Number(mo) + '月' + Number(d) + '日';
        })
        .trim();
    }

    /* 历史报告的 H1 是通用报头「科技新闻日报 · 日期」，和版块标题「深度日报」重复，直接不显示；
       新报告的大标题是当日主题标题，正常展示 */
    function displayReportTitle(title) {
      const clean = cleanReportTitle(title);
      return /科技新闻日报/.test(clean) ? '' : clean;
    }

    function cleanHeading(heading) {
      return String(heading || '')
        .replace(/^[一二三四五六七八九十]+、\s*/, '')
        .replace(/^[0-9]+[.、]\s*/, '')
        .trim();
    }

    /* 章节短标签：英文标题按词边界截断，避免出现 "Third Moment Resear" 这类半截单词 */
    function truncateTag(tag, limit) {
      const text = String(tag || '').trim();
      if (text.length <= limit) return text;
      const cut = text.slice(0, limit);
      const boundary = cut.lastIndexOf(' ');
      return (boundary > 0 ? cut.slice(0, boundary) : cut).replace(/[\s.&+·_-]+$/, '');
    }

    function chapterTag(heading) {
      const text = cleanHeading(heading);
      /* 英文/带变音符的拉丁开头：取开头连续拉丁词（到中文/标点为止），按词边界截断 */
      const latin = text.match(/^[A-Za-z\u00C0-\u024F][A-Za-z0-9\u00C0-\u024F .&+·_-]{1,24}/);
      if (latin) return truncateTag(latin[0], 16);
      /* 中文标题：取冒号前的短标签，最多 6 字（与旧实现一致，避免胶囊过长） */
      const colon = text.search(/[：:]/);
      if (colon > 0) return text.slice(0, colon).trim().slice(0, 6);
      return text.slice(0, 6);
    }

    function chapterTopic(heading) {
      const text = cleanHeading(heading);
      const colon = text.search(/[：:]/);
      let rest;
      if (colon > 0) {
        /* 有冒号时取冒号后的话题短语（如 "Third Moment Research：1 万智能体…" → "1 万智能体"） */
        rest = text.slice(colon + 1);
      } else {
        rest = text.slice(chapterTag(heading).length).replace(/^[\s，、：:·\-—]+/, '');
      }
      rest = rest.trim();
      const cut = rest.search(/[，。！？；：,.:;!?、]/);
      if (cut > 0) rest = rest.slice(0, cut);
      rest = rest.trim();
      /* 剩余过短（如 "ch"）时不渲染，避免出现无意义的话题标签 */
      if (rest.length < 2) return '';
      return rest.length > 12 ? rest.slice(0, 12) + '…' : rest;
    }

    function shortTopic(heading) {
      const text = cleanHeading(heading);
      const cut = text.search(/[，。！？；：,.:;!?]/);
      const label = cut > 0 ? text.slice(0, cut) : text;
      return label.length > 10 ? label.slice(0, 10) + '…' : label;
    }

    /* 新闻分类兜底：新报告由 LLM 在标题里给 [分类]；历史报告按标题关键词归类 */
    const NEWS_CATEGORY_RULES = [
      { label: '安全合规', pattern: /安全|漏洞|攻击|隐私|监管|合规|恶意|风险/ },
      { label: '研究论文', pattern: /论文|arXiv|研究|基准|实验|数据集|评测/ },
      { label: '开源项目', pattern: /GitHub|仓库|开源项目|stars?/i },
      { label: '行业动态', pattern: /投资|融资|收购|合作|政策|政府|市场|竞争|战略|押注|发布财报/ },
      { label: '技术突破', pattern: /突破|架构|算法|推理|加速|优化|压缩|剪枝/ },
      { label: '模型发布', pattern: /模型|发布|推出|上线|开源|训练|参数|智能体/ },
      { label: '产品工具', pattern: /产品|工具|应用|客户端|插件|功能|体验|助手/ }
    ];

    function _chapterMeta(chapters, podcast) {
      const realChapters = (chapters || []).filter(function (c) { return c && c.source === 'script'; });
      if (realChapters.length) return realChapters.length + ' 章节';
      const segments = Number(podcast && podcast.segment_count) || 0;
      return segments ? segments + ' 段对谈' : '';
    }

    function _podcastMetaText(podcast, chapters) {
      const segments = Number(podcast && podcast.segment_count) || 0;
      const realChapters = (chapters || []).filter(function (c) { return c && c.source === 'script'; });
      return [
        String((podcast && podcast.duration_label) || '--:--'),
        segments ? segments + ' 段对谈' : '',
        realChapters.length ? realChapters.length + ' 章节' : ''
      ].filter(Boolean).join(' · ');
    }

    /* 主持人默认音色标签：取 variants 里标记 default 且已合成的，否则取第一个可用 */
    function _defaultVoiceLabel(host) {
      const variants = (host && host.variants) || [];
      const usable = variants.filter(function (variant) {
        return variant && variant.available !== false && variant.label;
      });
      const def = usable.filter(function (variant) { return variant.default; })[0] || usable[0];
      return def ? String(def.label) : '';
    }

    function newsCategory(heading) {
      const text = String(heading || '');
      for (let i = 0; i < NEWS_CATEGORY_RULES.length; i += 1) {
        if (NEWS_CATEGORY_RULES[i].pattern.test(text)) return NEWS_CATEGORY_RULES[i].label;
      }
      return text ? '行业动态' : '';
    }

    /* 轻量 Markdown 渲染：空行分段、`1. ` 行转有序列表、**加粗** 转 strong。
       用于编辑导语 / 编辑短评这类「评语」——多条观点时按序号或段落展示。 */
    function inlineRichText(text) {
      return escapeHtml(text).replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    }

    function renderRichText(text) {
      const blocks = String(text || '')
        .split(/\n{2,}/)
        .map(function (block) { return block.trim(); })
        .filter(function (block) { return !!block; });
      return blocks.map(function (block) {
        const lines = block.split('\n').map(function (line) { return line.trim(); }).filter(Boolean);
        const isList = lines.length > 0 && lines.every(function (line) {
          return /^\d+[.、]\s*/.test(line);
        });
        if (isList) {
          return '<ol>' + lines.map(function (line) {
            return '<li>' + inlineRichText(line.replace(/^\d+[.、]\s*/, '')) + '</li>';
          }).join('') + '</ol>';
        }
        return '<p>' + lines.map(inlineRichText).join('<br>') + '</p>';
      }).join('');
    }

    function renderRichFields() {
      const lead = document.querySelector('.lead-quote');
      if (lead) lead.innerHTML = renderRichText(DATA.lead_quote);
      const quote = document.querySelector('.editor-quote-text');
      if (quote) quote.innerHTML = renderRichText(DATA.editor_quote);
    }

    function formatStars(stars) {
      const value = Number(stars);
      if (!isFinite(value)) return String(stars || '');
      if (value >= 1000) return (value / 1000).toFixed(1).replace(/\.0$/, '') + 'k';
      return String(value);
    }

    /* README 首段是 Markdown/HTML 混排（含 badge、&nbsp;、**强调**），落成纯文本再截断 */
    function cleanReadmeText(value) {
      let text = String(value || '');
      text = text.replace(/<[^>]+>/g, ' ');
      text = text.replace(/&nbsp;/gi, ' ').replace(/&amp;/gi, '&')
        .replace(/&lt;/gi, '<').replace(/&gt;/gi, '>').replace(/&quot;/gi, '"');
      text = text.replace(/!\[[^\]]*\]\([^)]*\)/g, ' ');
      text = text.replace(/\[([^\]]*)\]\([^)]*\)/g, '$1');
      text = text.replace(/[*_`#>]+/g, '');
      text = text.replace(/\s+/g, ' ').trim();
      if (!text) return '';
      return text.length > 220 ? text.slice(0, 220) + '…' : text;
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
      /* 设计稿的播客卡标题是「A 与 B」式短标题（如「Third Moment 与 Agent Kernel」）：
         取当日前两个章节 tag 组合；不足两个时回退到首章整句标题 */
      const headline = cleanHeading((dives[0] && dives[0].heading) || '');
      const topTags = dives
        .map(function (dive) { return chapterTag((dive && dive.heading) || ''); })
        .filter(function (tag) { return !!tag; })
        .slice(0, 2);
      const shortTitle = topTags.length >= 2 ? topTags[0] + ' 与 ' + topTags[1] : '';
      const data = {
        issue_number: date,
        issue_date: date,
        issue_tagline: 'AI 前沿 · 每日一刊',
        demo_note: '',
        podcast_title: shortTitle || headline || podcast.title || ('今日AI看点 | ' + date),
        podcast_subtitle: '双主持对谈 · 今日焦点解读',
        player_subtitle: '今日AI看点 · ' + date,
        podcast_duration: podcast.duration_label || '--:--',
        podcast_segment_count: String(podcast.segment_count || (playlist && playlist.chunks ? playlist.chunks.length : 0) || ''),
        host_a_voice: (hosts[0] && _defaultVoiceLabel(hosts[0])) || '',
        host_b_voice: (hosts[1] && _defaultVoiceLabel(hosts[1])) || '',
        podcast_audio_url: podcast.mp3_url || ('./episodes/' + date + '.mp3'),
        podcast_chapter_count: String(chapters.length || dives.length || 0),
        /* 01 板块头：有真实 [CHAPTER] 章节显示「N 章节」，历史期（cue 启发式）显示「N 段对谈」 */
        podcast_chapter_meta: _chapterMeta(chapters, podcast),
        /* 播客卡信息行：时长 · 段数 · 章节（不重复） */
        podcast_meta_text: _podcastMetaText(podcast, chapters),
        host_a_name: (hosts[0] && hosts[0].name) || '林夏',
        host_b_name: (hosts[1] && hosts[1].name) || '陈默',
        lead_quote: (d.lead && d.lead.text) || '',
        report_word_count: String(report.char_count || 0),
        report_read_minutes: String(report.reading_minutes || 0),
        report_pill: '深度研判',
        report_title: displayReportTitle(report.title),
        editor_quote: report.editor_verdict || (d.lead && d.lead.text) || '',
        editor_name: editorName,
        editor_role: '主编'
      };

      /* 指标按 digest 的 metric.key 映射到设计四槽位；四键齐全时用键映射，否则退回位置映射 */
      const METRIC_SLOT_BY_KEY = {
        deduped_items: 'sources',
        main_stories: 'hours',
        podcast_segments: 'speedup',
        media_outlets: 'items'
      };
      const METRIC_SLOTS = ['sources', 'hours', 'speedup', 'items'];
      const metrics = d.metrics || [];
      const metricBySlot = {};
      metrics.forEach(function (metric) {
        const slot = METRIC_SLOT_BY_KEY[metric && metric.key];
        if (slot) metricBySlot[slot] = metric;
      });
      const keyedMetrics = Object.keys(metricBySlot).length === METRIC_SLOTS.length;
      METRIC_SLOTS.forEach(function (slot, index) {
        const metric = (keyedMetrics ? metricBySlot[slot] : metrics[index]) || {};
        data['metric_' + slot + '_value'] = metric.value != null ? String(metric.value) : '';
        data['metric_' + slot + '_unit'] = metric.unit || '';
        data['metric_' + slot + '_label'] = metric.label || '';
      });

      for (let n = 1; n <= 4; n += 1) {
        const dive = dives[n - 1] || {};
        const title = cleanHeading(dive.heading || '');
        /* 要点：新报告带 {label,text}（1-3 条，按新闻性质自由选择）；老报告退回三段式 */
        let points = [];
        if (Array.isArray(dive.points) && dive.points.length) {
          points = dive.points.map(function (point) {
            return {
              label: String((point && point.label) || '').trim(),
              text: String((point && point.text) || '').trim()
            };
          }).filter(function (point) { return !!point.text; });
        } else {
          points = [
            { label: POINT_LABELS[0], text: dive.background || '' },
            { label: POINT_LABELS[1], text: dive.breakthrough || '' },
            { label: POINT_LABELS[2], text: dive.impact || '' }
          ];
          /* 老报告里未被结构化的段落：填进空缺要点位，避免整段塞进第 3 点 */
          (Array.isArray(dive.raw_sections) ? dive.raw_sections : []).forEach(function (section) {
            const text = String(section || '').trim();
            if (!text) return;
            const emptyIndex = points.findIndex(function (point) { return !point.text; });
            if (emptyIndex >= 0) points[emptyIndex].text = text;
          });
          points = points.filter(function (point) { return !!point.text; });
        }
        const storyTag = title ? chapterTag(title) : '';
        data['chapter_' + n + '_title'] = title;
        /* 左上角小字 = 新闻分类（模型发布/安全合规/…）；右侧小字 = 事件短名 */
        data['chapter_' + n + '_tag'] = title ? (dive.category || newsCategory(title)) : '';
        data['chapter_' + n + '_topic'] = storyTag;
        for (let m = 1; m <= 3; m += 1) {
          const point = points[m - 1] || {};
          data['chapter_' + n + '_point_' + m + '_label'] = point.text ? (point.label || POINT_LABELS[m - 1]) : '';
          data['chapter_' + n + '_point_' + m + '_text'] = point.text || '';
        }
      }

      /* 信源按设计稿展示 30 条精选（digest 入库上限 150），页头标注「30 / 150 条信源」 */
      const SOURCE_DISPLAY_LIMIT = 30;
      const allSourceItems = (d.sources || {}).items || [];
      const sourceItems = allSourceItems.slice(0, SOURCE_DISPLAY_LIMIT);
      data.sources_total_all = String((d.sources || {}).total_all || allSourceItems.length || 0);
      data.sources_list = JSON.stringify(sourceItems.map(function (item) {
        return {
          badge: SOURCE_BADGE_KEYS[item.group] || 'industry',
          badge_label: item.source_name || item.publisher || '',
          meta: date,
          title: item.title || '',
          summary: item.summary || '',
          url: item.link || ''
        };
      }));

      /* 本期话题：取重磅解读（带锚点）+ 前沿情报（跳到深度日报），点击可定位 */
      const topics = [];
      dives.slice(0, 2).forEach(function (dive, index) {
        const label = chapterTag((dive && dive.heading) || '');
        if (label) topics.push({ label: label, anchor: '#deep-dive-' + (index + 1) });
      });
      (report.frontier || []).forEach(function (item) {
        if (topics.length >= 3) return;
        const label = shortTopic((item && item.title) || '');
        if (label) topics.push({ label: label, anchor: '#section-report' });
      });
      data.topics_list = JSON.stringify(topics.slice(0, 3));

      const radar = d.radar || {};
      const candidates = radar.candidates || {};
      const pick = radar.pick || null;
      const radarAvailable = radar.available !== false;
      data.radar_available = radarAvailable;
      data.radar_status_label = !radarAvailable
        ? '该期无雷达'
        : (pick ? '有主推项目' : '无主推项目');
      if (!radarAvailable) {
        /* 该期没有雷达数据：走独立空态，不伪装成「轮空主推」 */
        data.radar_headline = String(radar.message || '该期暂无项目雷达');
        data.radar_paragraph = '项目雷达自 2026 年 10 月上线，此前各期不含该项目数据。';
        data.radar_reason = '';
        data.candidate_count = '';
        data.excluded_recent = '';
      } else {
        data.radar_headline = pick
          ? '今日候选池命中达标项目'
          : '今日候选池经多轮筛查，未检出达标项目';
        data.radar_paragraph = pick
          ? '本栏目每日严选 1 个具备重大架构革新且增速稳定的开源项目。今日主推仓库经工程稳定性、重复收录排查与实际代码审查后入选。'
          : '本栏目每日严选 1 个具备重大架构革新且近 7 日 Star 增速稳定突破 300/天的优质开源项目。今日全网扫描了一批候选仓库，经工程稳定性、重复收录排查及实际代码审查，均处于评估观察期或版本迁移期，编辑部决定今日宁缺毋滥、轮空主推。';
        data.radar_reason = pick ? '编辑部一致通过' : '宁缺毋滥、轮空主推';
        data.candidate_count = String(candidates.total != null ? candidates.total : (radar.runner_ups || []).length || 0);
        data.excluded_recent = String(candidates.excluded_recent != null ? candidates.excluded_recent : 0);
      }
      data.radar_candidates_total = data.candidate_count;
      data.radar_candidates_semantics = String(candidates.semantics || '');
      data.radar_candidates_preview = JSON.stringify(
        Array.isArray(candidates.preview) ? candidates.preview.slice(0, 20) : []
      );
      data.radar_candidates_detail_available = candidates.detail_available === true;
      data.radar_pick_repo = pick ? String(pick.repo || '') : '';
      data.radar_pick_lang = pick ? String(pick.language || '') : '';
      data.radar_pick_stars = pick ? formatStars(pick.stars) : '';
      const growth = pick && pick.delta_stars != null
        ? '近 7 日 ★ +' + pick.delta_stars + '/天'
        : '';
      /* 首日收录的项目没有增速基线：明说，而不是留一个「+/天」的空位 */
      data.radar_pick_growth_text = growth || (pick && pick.is_new ? '首日收录 · 暂无增速基线' : '');
      data.radar_pick_license = pick ? String(pick.license || '') : '';
      /* 优先用 GitHub 仓库描述（干净一句话）；readme 首段只做兜底并清成纯文本 */
      data.radar_pick_desc = pick
        ? String(pick.description || '').trim() || cleanReadmeText(pick.readme_intro)
        : '';
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
      if (stamp) stamp.textContent = DATA.issue_date || '';
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
      /* 要点/话题为空时整块隐藏：日报结构不完整时不再留空高亮框 */
      document.querySelectorAll('.deep-dive-point').forEach(function (point) {
        const text = point.querySelector('.point-text');
        point.hidden = !(text && text.textContent.trim());
      });
      document.querySelectorAll('.deep-dive-topic, .deep-dive-tag').forEach(function (node) {
        node.hidden = !node.textContent.trim();
      });
      /* 雷达主推卡：语言/Star/许可证缺省时隐藏，避免只剩「· ·」 */
      document.querySelectorAll('.radar-pick-meta > span:not(.sep)').forEach(function (node) {
        node.hidden = !node.textContent.trim();
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

    /* 音频代际：切期释放旧 audio 时递增，旧元素的异步事件不再污染新一期的状态 */
    let audioGeneration = 0;

    function releaseAudio() {
      audioGeneration += 1;
      if (!audioEl) return;
      const stale = audioEl;
      audioEl = null;
      stale.pause();
      stale.removeAttribute('src');
      try { stale.load(); } catch (e) { /* 旧浏览器忽略 */ }
    }

    function ensureAudio() {
      const url = audioUrl();
      if (!url || audioBroken) return null;
      if (audioEl && audioEl.dataset.src === url) return audioEl;
      releaseAudio();
      const generation = audioGeneration;
      const el = new Audio(url);
      audioEl = el;
      el.dataset.src = url;
      el.preload = 'metadata';
      const isCurrent = function () { return generation === audioGeneration && audioEl === el; };
      el.addEventListener('loadedmetadata', function() {
        if (!isCurrent()) return;
        if (isFinite(el.duration)) { audioDuration = el.duration; renderClock(); }
      });
      el.addEventListener('timeupdate', function() {
        if (!isCurrent()) return;
        currentTime = el.currentTime;
        renderClock();
      });
      el.addEventListener('ended', function() { if (isCurrent()) setPlaying(false); });
      el.addEventListener('error', function() {
        /* 音频缺失（本地预览 / 网络异常）时退化为模拟时钟，播放按钮仍可用 */
        if (!isCurrent()) return;
        const wasPlaying = isPlaying;
        audioBroken = true;
        audioEl = null;
        if (wasPlaying) startClock();
      });
      return el;
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
    function drawerElement() { return document.getElementById('transcript-drawer'); }

    function openDrawer() {
      drawerReturnFocus = document.activeElement;
      const drawer = drawerElement();
      if (!drawer) return;
      drawer.classList.add('open');
      drawer.setAttribute('aria-hidden', 'false');
      document.getElementById('drawer-overlay').classList.add('visible');
      document.body.style.overflow = 'hidden';
      const closeBtn = document.querySelector('.drawer-close');
      if (closeBtn) closeBtn.focus();
    }
    function closeDrawer() {
      const drawer = drawerElement();
      if (!drawer || !drawer.classList.contains('open')) return;
      drawer.classList.remove('open');
      drawer.setAttribute('aria-hidden', 'true');
      document.getElementById('drawer-overlay').classList.remove('visible');
      document.body.style.overflow = '';
      if (drawerReturnFocus && typeof drawerReturnFocus.focus === 'function') {
        drawerReturnFocus.focus();
      }
      drawerReturnFocus = null;
    }
    function trapDrawerFocus(event) {
      const drawer = drawerElement();
      if (!drawer || !drawer.classList.contains('open')) return;
      const focusable = Array.from(drawer.querySelectorAll(
        'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
      )).filter(function(node) { return !node.disabled && node.offsetParent !== null; });
      if (!focusable.length) return;
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    }
    document.addEventListener('keydown', function(e) {
      if (e.key === 'Escape') { closeDrawer(); return; }
      if (e.key === 'Tab') trapDrawerFocus(e);
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

    /* 切期请求令牌：快速连点时只应用最后一次请求的结果 */
    let loadToken = 0;

    async function loadIssue(id) {
      if (!id || ISSUE_ORDER.indexOf(id) === -1) return;
      if (DATA.issue_date === id) return;
      const token = ++loadToken;
      let digest;
      let playlist = null;
      try {
        digest = await fetchJson('./data/digest_' + id + '.json');
      } catch (error) {
        console.warn('digest load failed', id, error);
        return;
      }
      if (token !== loadToken) return;
      try {
        playlist = await fetchJson('./episodes/' + id + '/playlist.json');
      } catch (error) {
        playlist = { chunks: [] };
      }
      if (token !== loadToken) return;
      DATA = digestToData(digest, playlist);
      dataReady = true;
      PLAYLIST_CHUNKS = (playlist && playlist.chunks) || [];
      sourcesExpanded = false;
      sourceFilter = 'all';
      currentTime = 0;
      audioBroken = false;
      releaseAudio();
      audioDuration = 0;
      setPlaying(false);
      renderClock();
      hydrate();
      applyPostHydrateChrome();
      applyConditionalBlocks();
      renderRichFields();
      closeArchive();
      renderDatePills();
      updateDateNav();
      renderHosts();
      renderTopics();
      renderSourceFilters();
      renderSources();
      renderTranscript();
      renderRadar();
      const reduceMotion = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
      window.scrollTo({ top: 0, behavior: reduceMotion ? 'auto' : 'smooth' });
    }

    function switchIssue(id) { return loadIssue(id); }

    /* Render date pills — 最近 6 期直接展示，其余收进「历史归档」下拉（30 期不撑爆日期轴） */
    const DATE_PILL_LIMIT = 6;

    function issuePill(id) {
      const label = String(id).slice(5);
      const active = id === DATA.issue_date;
      return '<button class="date-pill' + (active ? ' active' : '') + '"' +
        ' onclick="selectIssue(\'' + escapeHtml(id) + '\')" aria-pressed="' + active + '">' +
        escapeHtml(label) + '</button>';
    }

    function renderDatePills() {
      const container = document.getElementById('date-pills');
      if (!container) return;
      let visible = ISSUE_ORDER.slice(0, DATE_PILL_LIMIT);
      /* 当前期不在最近 6 期时也保留一颗激活胶囊，否则用户看不到自己在哪一期 */
      if (DATA.issue_date && ISSUE_ORDER.indexOf(DATA.issue_date) >= DATE_PILL_LIMIT) {
        visible = [DATA.issue_date].concat(visible.slice(0, DATE_PILL_LIMIT - 1));
      }
      const archive = ISSUE_ORDER.length > visible.length
        ? '<button type="button" class="date-pill date-archive" id="date-archive"' +
          ' aria-expanded="false" aria-haspopup="true" onclick="toggleArchive(event)">历史归档 ▾</button>'
        : '';
      container.innerHTML = visible.map(issuePill).join('') + archive;

      const panel = document.getElementById('date-archive-panel');
      if (panel) {
        panel.innerHTML = ISSUE_ORDER.map(function (id) {
          return '<button type="button" class="archive-item' +
            (id === DATA.issue_date ? ' active' : '') + '" data-date="' + escapeHtml(id) + '"' +
            ' onclick="selectIssue(\'' + escapeHtml(id) + '\')">' + escapeHtml(id) + '</button>';
        }).join('');
      }
    }

    function closeArchive() {
      const panel = document.getElementById('date-archive-panel');
      const button = document.getElementById('date-archive');
      if (panel) panel.hidden = true;
      if (button) button.setAttribute('aria-expanded', 'false');
    }

    function toggleArchive(event) {
      if (event) event.stopPropagation();
      const panel = document.getElementById('date-archive-panel');
      const button = document.getElementById('date-archive');
      if (!panel || !button) return;
      const open = panel.hidden;
      panel.hidden = !open;
      button.setAttribute('aria-expanded', String(open));
    }

    document.addEventListener('click', function (event) {
      const panel = document.getElementById('date-archive-panel');
      const button = document.getElementById('date-archive');
      if (!panel || panel.hidden) return;
      if (panel.contains(event.target) || (button && button.contains(event.target))) return;
      closeArchive();
    });

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

    function selectIssue(id) {
      closeArchive();
      return switchIssue(id);
    }

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
        ' onclick="setSourceFilter(\'' + escapeHtml(c.key) + '\')" aria-pressed="' + (sourceFilter === c.key) + '">' +
        escapeHtml(c.label) + '<span class="chip-count">' + escapeHtml(c.count) + '</span></button>'
      ).join('');
    }

    function setSourceFilter(key) {
      sourceFilter = key;
      sourcesExpanded = false;
      renderSources();
      renderSourceFilters();
    }

    /* 外链白名单：只放行 http(s)，避免 RSS 里出现 javascript: 之类的伪协议 */
    function safeHref(url) {
      const value = String(url || '').trim();
      return /^https?:\/\//i.test(value) ? value : '';
    }

    const SOURCE_BADGE_CLASSES = ['openai', 'anthropic', 'google', 'meta', 'arxiv', 'nvidia', 'industry'];
    const SOURCE_PREVIEW = 6;

    function sourceBadgeClass(badge) {
      const key = String(badge || 'industry');
      return SOURCE_BADGE_CLASSES.indexOf(key) >= 0 ? key : 'industry';
    }

    /* Render sources — 有 url 用 url，否则回落到来源站点首页 */
    function renderSources() {
      const grid = document.getElementById('sources-grid');
      if (!grid) return;
      const sources = filteredSources();
      grid.innerHTML = sources.map((s, i) => {
        const hidden = (i >= SOURCE_PREVIEW && !sourcesExpanded) ? ' style="display:none"' : '';
        const href = safeHref(s.url) || SOURCE_ORIGINS[s.badge_label] || SOURCE_ORIGINS[s.badge] || '';
        const title = s.title || '';
        const summary = s.summary || '';
        const inner =
          '<div class="source-card-header">' +
          '<span class="source-badge ' + sourceBadgeClass(s.badge) + '">' +
          escapeHtml(s.badge_label) + '</span>' +
          '<span class="source-meta">' + escapeHtml(s.meta) + '</span>' +
          '</div>' +
          '<div class="source-title">' + escapeHtml(title) + '</div>' +
          (summary && summary !== title
            ? '<div class="source-summary">' + escapeHtml(summary) + '</div>'
            : '');
        if (!href) return '<article class="source-card"' + hidden + '>' + inner + '</article>';
        return '<a class="source-card" href="' + escapeHtml(href) + '" target="_blank" rel="noopener"' +
          hidden + '>' + inner + '</a>';
      }).join('');

      const btn = document.getElementById('btn-expand-sources');
      if (btn) {
        btn.textContent = sourcesExpanded ? '收起信源' : '展开全部';
        btn.setAttribute('aria-expanded', String(sourcesExpanded));
        const bar = btn.closest('.sources-expand-bar');
        if (bar) bar.hidden = sources.length <= SOURCE_PREVIEW;
      }

      const label = document.getElementById('sources-count-label');
      if (label) {
        const total = Number(DATA.sources_total_all) || sources.length;
        label.textContent = (total > sources.length && sourceFilter === 'all')
          ? sources.length + ' / ' + total + ' 条信源'
          : sources.length + ' 条信源';
      }
    }

    const RADAR_REASON_LABELS = {
      selected: '本期主推',
      score_rank: '评分排名靠后',
      direction_gate: '方向闸门过滤',
      recent_pick: '近 30 天已推荐',
      probe_limit: '超出探测上限'
    };

    /* 候选审计明细：新数据渲染表格，老数据只给口径说明（fail-soft，不影响轮空态） */
    function renderRadarCandidates() {
      const actions = document.getElementById('radar-empty-actions');
      const panel = document.getElementById('radar-candidates-panel');
      const button = document.getElementById('btn-radar-candidates');
      if (!actions || !panel || !button) return;
      const preview = safeParse(DATA.radar_candidates_preview, []);
      const rows = Array.isArray(preview) ? preview.filter(item => item && item.repo) : [];
      const total = Number(DATA.radar_candidates_total) || rows.length;
      const excluded = Number(DATA.excluded_recent) || 0;
      const available = DATA.radar_candidates_detail_available === true && rows.length > 0;

      actions.hidden = DATA.radar_available === false;
      button.hidden = false;
      button.setAttribute('aria-expanded', 'false');
      panel.hidden = true;
      if (actions.hidden) return;

      if (available) {
        button.innerHTML = '<span aria-hidden="true">📋</span> 查看今日参与评估的 ' +
          escapeHtml(total) + ' 个候选项目明细';
        panel.innerHTML = '<table><thead><tr><th>仓库</th><th>Stars</th><th>日增</th>' +
          '<th>语言</th><th>未收录原因</th></tr></thead><tbody>' +
          rows.map(function (item) {
            const delta = item.delta_stars == null ? '—' : '+' + item.delta_stars;
            const reason = RADAR_REASON_LABELS[item.exclusion_reason] || item.exclusion_reason || '—';
            return '<tr><td class="radar-candidate-repo">' + escapeHtml(item.repo) +
              '</td><td>' + escapeHtml(item.stars) + '</td><td>' + escapeHtml(delta) +
              '</td><td>' + escapeHtml(item.language || '—') +
              '</td><td><span class="radar-candidate-reason">' + escapeHtml(reason) +
              '</span></td></tr>';
          }).join('') +
          '</tbody></table>';
      } else {
        button.innerHTML = '<span aria-hidden="true">📋</span> 查看今日候选池口径';
        panel.innerHTML = '<p class="radar-candidates-note">该日期早于候选审计上线，暂无逐项明细。' +
          '本期共扫描 ' + escapeHtml(total) + ' 个候选仓库，其中 ' + escapeHtml(excluded) +
          ' 个为近 30 天已推荐或重复收录。</p>';
      }
    }

    function toggleRadarCandidates() {
      const panel = document.getElementById('radar-candidates-panel');
      const button = document.getElementById('btn-radar-candidates');
      if (!panel || !button) return;
      const open = panel.hidden;
      panel.hidden = !open;
      button.setAttribute('aria-expanded', String(open));
    }

    /* 雷达：有主推数据时渲染主推卡，否则保留轮空态（轮空态带候选审计入口） */
    function renderRadar() {
      const empty = document.getElementById('radar-empty');
      const pick = document.getElementById('radar-pick');
      if (!empty || !pick) return;
      const hasPick = !!(DATA.radar_pick_repo && String(DATA.radar_pick_repo).trim());
      empty.hidden = hasPick;
      pick.hidden = !hasPick;
      const actions = document.getElementById('radar-empty-actions');
      const panel = document.getElementById('radar-candidates-panel');
      if (hasPick) {
        if (actions) actions.hidden = true;
        if (panel) panel.hidden = true;
        stopRadar();
      } else {
        renderRadarCandidates();
        startRadar();
      }
    }

    /* 主持人由数据驱动，roster 与逐字稿头像共用同一份姓名 */
    function hostInitial(name) {
      return String(name || '').trim().slice(0, 1) || '·';
    }
    function renderHosts() {
      const list = document.getElementById('host-roster-list');
      if (!list) return;
      list.innerHTML = [
        { cls: 'host-a', name: DATA.host_a_name, voice: DATA.host_a_voice },
        { cls: 'host-b', name: DATA.host_b_name, voice: DATA.host_b_voice }
      ].filter(h => h.name && String(h.name).trim()).map(h =>
        '<span class="host-identity">' +
        '<span class="host-avatar ' + h.cls + '">' + escapeHtml(hostInitial(h.name)) + '</span>' +
        '<span class="host-text">' +
        '<span class="host-name">' + escapeHtml(h.name) + '</span>' +
        (h.voice ? '<span class="host-voice">' + escapeHtml(h.voice) + '</span>' : '') +
        '</span>' +
        '</span>'
      ).join('');
    }

    /* 本期话题 chips：点击定位到对应章节 / 深度日报 */
    function renderTopics() {
      const wrap = document.getElementById('podcast-topics');
      const box = document.getElementById('podcast-topic-chips');
      if (!wrap || !box) return;
      const topics = safeParse(DATA.topics_list, []);
      const items = (Array.isArray(topics) ? topics : []).filter(function (item) {
        return item && item.label;
      });
      if (!items.length) {
        wrap.hidden = true;
        return;
      }
      wrap.hidden = false;
      box.innerHTML = items.map(function (item) {
        return '<button type="button" class="podcast-topic-chip" data-anchor="' +
          escapeHtml(item.anchor || '#section-report') + '">' + escapeHtml(item.label) + '</button>';
      }).join('');
      box.querySelectorAll('.podcast-topic-chip').forEach(function (chip) {
        chip.addEventListener('click', function () {
          const target = document.querySelector(chip.getAttribute('data-anchor'));
          if (target) target.scrollIntoView({ behavior: 'smooth', block: 'start' });
        });
      });
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
          '<span class="transcript-speaker ' + (isA ? 'host-a' : 'host-b') + '">' +
          escapeHtml(hostInitial(name)) + '</span>' +
          '<div class="transcript-text"><span class="timestamp">' + escapeHtml(l.time) + '</span>' +
          escapeHtml(l.text) + '</div>' +
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
      ctx.strokeStyle = 'rgba(212, 165, 116, 0.1)';
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
            return v === null ? (dataReady ? '' : m) : v;
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
            return v === null ? (dataReady ? '' : m) : v;
          }));
        });
      });

      /* 标签页标题：品牌 + 期号日期（切期同步；首屏由服务端直接写好，无占位符闪烁） */
      const pageTitle = DATA.issue_date ? '今日AI看点 · ' + DATA.issue_date : '今日AI看点';
      if (document.title !== pageTitle) document.title = pageTitle;

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
