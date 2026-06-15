    const DATES = {dates_json};
    const EPISODES = {episodes_map_json};
    const WEEKDAYS = ['日','一','二','三','四','五','六'];
    const SPEEDS = [1, 1.25, 1.5, 1.75, 2];
    let speedIdx = 0;
    let currentDate = null;
    let currentEpId = null;
    let lastVol = 0.8;
    let isMuted = false;
    const audio = document.getElementById('main-audio');
    audio.volume = lastVol;

    let playbackMode = 'full'; // 'full' | 'sentence'
    let currentPlaylist = null; // List of chunks from playlist.json
    let currentChunkIndex = 0;
    const bgmAudio = document.getElementById('bgm-audio');
    bgmAudio.volume = 0.05; // 低音量背景音乐

    let currentMode = 'podcast';
    let scriptTotalChars = 0;
    let scriptParagraphs = [];

    // 人机共存滚动参数
    let userScrolling = false;
    let userScrollTimer = null;
    let isInternalScroll = false;

    function switchMode(mode) {
      if (currentMode === mode) return;
      currentMode = mode;

      document.getElementById('btn-mode-report').classList.toggle('active', mode === 'report');
      document.getElementById('btn-mode-podcast').classList.toggle('active', mode === 'podcast');

      const heroDesc = document.getElementById('hero-desc');
      if (mode === 'report') {
        document.body.classList.add('theme-report');
        heroDesc.textContent = '科技日报 · 聚合每日 AI 领域深度前沿动态';
      } else {
        document.body.classList.remove('theme-report');
        heroDesc.textContent = 'AI 播客电台 · 每日 AI 资讯的声音解说与剧本展示';
      }

      const panelReport = document.getElementById('panel-report');
      const panelPodcast = document.getElementById('panel-podcast');

      if (mode === 'report') {
        panelPodcast.style.display = 'none';
        panelReport.style.display = 'flex';
      } else {
        panelReport.style.display = 'none';
        panelPodcast.style.display = 'grid';
      }

      buildDatePills();
    }

    function buildDatePills() {
      const wrap = document.getElementById('date-pills');
      wrap.innerHTML = '';

      const filteredDates = DATES.filter(function(d) {
        if (currentMode === 'podcast') {
          return EPISODES[d] && EPISODES[d].mp3;
        }
        return true;
      });

      filteredDates.forEach(function(d) {
        var dt = new Date(d + 'T00:00:00');
        var pill = document.createElement('div');
        pill.className = 'date-pill';
        pill.setAttribute('data-date', d);
        pill.onclick = function() { loadDate(d); };
        var mon = dt.getMonth() + 1;
        var day = dt.getDate();
        var wk = WEEKDAYS[dt.getDay()];
        pill.innerHTML = '<span class="pill-month">' + mon + '月</span><span class="pill-day">' + day + '</span><span class="pill-weekday">周' + wk + '</span>';
        wrap.appendChild(pill);
      });

      if (filteredDates.length > 0) {
        if (currentDate && filteredDates.includes(currentDate)) {
          loadDate(currentDate);
        } else {
          loadDate(filteredDates[0]);
        }
      } else {
        if (currentMode === 'report') {
          setEmpty(document.getElementById('report-panel-body'), '📰', '该模式下暂无日期数据');
        } else {
          setEmpty(document.getElementById('cast-panel-body'), '🎙️', '暂无播客电台数据');
        }
      }
    }

    function setActiveDate(d) {
      currentDate = d;
      document.querySelectorAll('.date-pill').forEach(function(p) {
        p.classList.toggle('active', p.getAttribute('data-date') === d);
      });
    }

    async function loadDate(d) {
      var isFirstLoad = (currentDate === null);
      setActiveDate(d);
      var ep = EPISODES[d] || {};
      var dt = new Date(d + 'T00:00:00');
      var label = dt.getFullYear() + '年' + (dt.getMonth()+1) + '月' + dt.getDate() + '日';
      document.getElementById('hero-title').textContent = ep.title || ('AI 新闻快报 | ' + d);

      // 渲染引证参考源卡片
      var sourcesCard = document.getElementById('sources-card');
      var sourcesList = document.getElementById('sources-list-body');
      if (currentMode === 'podcast') {
        if (ep.desc && ep.desc.trim()) {
          sourcesList.innerHTML = ep.desc;
          sourcesCard.style.display = 'flex';
        } else {
          sourcesCard.style.display = 'none';
        }
      }

      if (currentMode === 'report') {
        document.getElementById('report-date-tag').textContent = label;
        var repBody = document.getElementById('report-panel-body');
        setLoading(repBody, '正在加载日报…');
        try {
          var r = await fetch('./reports/daily_report_' + d + '.md');
          if (!r.ok) throw new Error();
          var md = await r.text();
          repBody.innerHTML = '<div class="report-content">' + marked.parse(md) + '</div>';

          extractEditorVerdict();
        } catch(e) {
          if (ep.desc) {
            repBody.innerHTML = '<div style="font-size:.88rem;color:#d1d5db;line-height:1.75">' + ep.desc + '</div>';
          } else {
            setEmpty(repBody, '📰', '该日期暂无日报');
          }
          document.getElementById('editor-verdict-wrapper').style.display = 'none';
        }
      } else {
        document.getElementById('podcast-date-tag').textContent = label;
        document.getElementById('side-podcast-title').textContent = ep.title || ('AI 新闻快报 | ' + d);
        var castBody = document.getElementById('cast-panel-body');
        setLoading(castBody, '正在加载剧本…');
        var loaded = false;
        try {
          var r2 = await fetch('./episodes/' + d + '.txt');
          if (!r2.ok) throw new Error();
          var txt = await r2.text();
          castBody.innerHTML = parseTranscript(txt);
          loaded = true;
        } catch(e2) {}
        if (!loaded) {
          if (ep.desc) {
            castBody.innerHTML = '<div style="font-size:.88rem;color:#d1d5db;line-height:1.75">' + ep.desc + '</div>';
          } else {
            setEmpty(castBody, '🎙️', '该日期暂无剧本');
          }
        }

        currentPlaylist = null;
        try {
          var rPl = await fetch('./episodes/' + d + '/playlist.json');
          if (rPl.ok) {
            var plData = await rPl.json();
            currentPlaylist = plData.chunks;
            document.getElementById('playback-btn-sentence').style.display = 'block';
          } else {
            document.getElementById('playback-btn-sentence').style.display = 'none';
            setPlaybackMode('full');
          }
        } catch (ePl) {
          document.getElementById('playback-btn-sentence').style.display = 'none';
          setPlaybackMode('full');
        }

        if (playbackMode === 'sentence' && currentPlaylist && currentPlaylist.length > 0) {
          currentChunkIndex = 0;
          if (currentEpId !== d) {
            currentEpId = d;
            loadChunk(currentChunkIndex, !isFirstLoad);
            bgmAudio.src = 'assets/bgm_placeholder.wav';
          }
        } else {
          if (ep.mp3 && currentEpId !== d) {
            loadAudio(d, ep.mp3, ep.title || ('AI 新闻快报 | ' + d), !isFirstLoad);
          }
        }
      }
    }

    function extractEditorVerdict() {
      var repBody = document.getElementById('report-panel-body');
      var quoteWrapper = document.getElementById('editor-verdict-wrapper');
      var quoteBody = document.getElementById('editor-quote-body');

      if (!repBody || !quoteWrapper || !quoteBody) return;

      var bq = repBody.querySelector('blockquote');
      if (bq) {
        quoteBody.innerHTML = bq.innerHTML;
        quoteWrapper.style.display = 'block';
        bq.style.display = 'none';
      } else {
        quoteWrapper.style.display = 'none';
      }
    }

    function setLoading(el, msg) {
      el.innerHTML = '<div class="state-placeholder"><div class="loading-spinner"></div><p>' + msg + '</p></div>';
    }
    function setEmpty(el, icon, msg) {
      el.innerHTML = '<div class="state-placeholder"><div style="font-size:2.5rem;opacity:.5">' + icon + '</div><p>' + msg + '</p></div>';
    }

    function parseTranscript(text) {
      if (!text || !text.trim()) return '<div class="state-placeholder"><p>剧本内容为空</p></div>';
      var lines = [];
      var rawParagraphs = [];

      // 1. 使用浏览器的 DOM 树解析器进行强健解析 SSML 标签
      try {
        var parser = new DOMParser();
        var doc = parser.parseFromString(text, "text/html");
        var voiceTags = doc.querySelectorAll("voice");
        if (voiceTags && voiceTags.length > 0) {
          voiceTags.forEach(function(voiceTag, idx) {
            var c = voiceTag.textContent.trim();
            if (!c) return;
            var voiceName = (voiceTag.getAttribute("name") || "").toLowerCase();
            var start = parseFloat(voiceTag.getAttribute("start") || "-1");
            var duration = parseFloat(voiceTag.getAttribute("duration") || "-1");

            var isXx = voiceName.indexOf('xiaoxiao') >= 0 || voiceName.indexOf('host-b') >= 0 || voiceName.indexOf('host b') >= 0;
            rawParagraphs.push({
              role: isXx ? 'B' : 'A',
              text: c,
              start: start >= 0 ? start : null,
              duration: duration >= 0 ? duration : null
            });
          });
        }
      } catch (eDOM) {}

      // 2. 如果 DOM 解析未成功或无 voice 标签，尝试正则兼容旧 SSML 格式
      if (rawParagraphs.length === 0) {
        var re = /<voice\s+name\s*=\s*["']([^"']+)["']\s*>([\s\S]*?)<\/voice>/gi;
        var m;
        while ((m = re.exec(text)) !== null) {
          var c = m[2].trim();
          if (!c) continue;
          var voiceName = m[1].toLowerCase();
          var isXx = voiceName.indexOf('xiaoxiao') >= 0 || voiceName.indexOf('host-b') >= 0 || voiceName.indexOf('host b') >= 0;
          rawParagraphs.push({ role: isXx ? 'B' : 'A', text: c, start: null, duration: null });
        }
      }

      // 3. 尝试匹配中括号格式 [Host A] / [Host B]
      if (rawParagraphs.length === 0) {
        var re2 = /\[Host\s*([AB])\]\s*([^\[]*)/gi;
        var m2;
        while ((m2 = re2.exec(text)) !== null) {
          var c2 = m2[2].trim();
          if (!c2) continue;
          rawParagraphs.push({ role: m2[1].toUpperCase(), text: c2, start: null, duration: null });
        }
      }

      // 4. 再次尝试：解析冒号前缀格式 (如 "博文：" 或 "晓晓:")
      if (rawParagraphs.length === 0) {
        var textLines = text.split('\n');
        textLines.forEach(function(line) {
          var trimmed = line.trim();
          if (!trimmed) return;
          var colonIdx = trimmed.search(/[:：]/);
          if (colonIdx > 0 && colonIdx < 10) {
            var name = trimmed.substring(0, colonIdx).trim();
            var content = trimmed.substring(colonIdx + 1).trim();
            if (content) {
              var isB = name.indexOf('晓晓') >= 0 || name.toLowerCase().indexOf('b') >= 0 || name.indexOf('女') >= 0;
              rawParagraphs.push({ role: isB ? 'B' : 'A', text: content, start: null, duration: null });
            }
          }
        });
      }

      // 5. 段落交替兜底：如果没有结构化数据，去掉XML标签，以空白行切分并交替赋予角色
      if (rawParagraphs.length === 0) {
        var cleanText = text.replace(/<[^>]+>/g, '').trim();
        var textLines2 = cleanText.split(/\n+/);
        textLines2.forEach(function(line, idx) {
          var trimmed = line.trim();
          if (!trimmed) return;
          rawParagraphs.push({ role: (idx % 2 === 0) ? 'A' : 'B', text: trimmed, start: null, duration: null });
        });
      }

      // 6. 渲染页面并计算字符偏移量
      scriptTotalChars = 0;
      scriptParagraphs = [];
      rawParagraphs.forEach(function(p) {
        scriptTotalChars += p.text.length;
      });

      var currentOffset = 0;
      rawParagraphs.forEach(function(p, index) {
        var cls = p.role === 'B' ? 'host-b' : 'host-a';
        var name = p.role === 'B' ? '晓晓' : '博文';
        var avatarChar = p.role === 'B' ? '晓' : '博';
        var len = p.text.length;

        var rowHtml = '<div class="transcript-row" id="trans-row-' + index + '" ' +
                      'data-offset="' + currentOffset + '" ' +
                      'data-len="' + len + '" ' +
                      (p.start !== null ? 'data-start="' + p.start + '" data-duration="' + p.duration + '" ' : '') +
                      'onclick="seekToParagraph(' + index + ')">' +
                      '<div class="speaker-meta">' +
                        '<div class="speaker-avatar ' + cls + '">' + avatarChar + '</div>' +
                        '<div class="speaker-name">' + name + '</div>' +
                        '<div class="seek-play-icon">▶</div>' +
                      '</div>' +
                      '<div class="transcript-text">' + esc(p.text) + '</div>' +
                    '</div>';
        lines.push(rowHtml);

        scriptParagraphs.push({
          index: index,
          offset: currentOffset,
          len: len,
          start: p.start,
          duration: p.duration
        });
        currentOffset += len;
      });

      return lines.join('');
    }

    function esc(s) {
      return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
    }

    function setPlaybackMode(mode) {
      if (playbackMode === mode) return;
      playbackMode = mode;
      document.getElementById('playback-btn-full').classList.toggle('active', mode === 'full');
      document.getElementById('playback-btn-sentence').classList.toggle('active', mode === 'sentence');

      audio.pause();
      bgmAudio.pause();
      var ep = EPISODES[currentDate] || {};
      if (mode === 'sentence' && currentPlaylist && currentPlaylist.length > 0) {
        currentChunkIndex = 0;
        loadChunk(currentChunkIndex, false);
        bgmAudio.src = 'assets/bgm_placeholder.wav';
      } else {
        if (ep.mp3) {
          loadAudio(currentDate, ep.mp3, ep.title || ('AI 新闻快报 | ' + currentDate), false);
        }
      }
    }

    function loadChunk(index, autoplay) {
      if (!currentPlaylist || index < 0 || index >= currentPlaylist.length) return;
      currentChunkIndex = index;
      var chunk = currentPlaylist[index];
      audio.src = './episodes/' + currentDate + '/' + chunk.audio;
      if (autoplay) {
        audio.play().catch(function(){});
        if (playbackMode === 'sentence') {
          bgmAudio.play().catch(function(){});
        }
      } else {
        audio.load();
      }

      // 智能句读模式下切换片段，需要重置高亮和同步滚动
      // 我们的 syncScrollTo(true) 会由 timeupdate 触发，或者直接在这里触发
      document.querySelectorAll('.transcript-row').forEach(function(el) {
        el.classList.remove('speaking');
      });
      var activeEl = document.getElementById('trans-row-' + index);
      if (activeEl) {
        activeEl.classList.add('speaking');
        if (autoplay || !userScrolling) {
          var container = document.getElementById('cast-panel-body');
          var relativeTop = activeEl.offsetTop - container.offsetTop;
          var containerHeight = container.clientHeight;
          var activeHeight = activeEl.clientHeight;
          var targetScroll = relativeTop - (containerHeight / 2) + (activeHeight / 2);
          isInternalScroll = true;
          container.scrollTo({ top: targetScroll, behavior: 'smooth' });
        }
      }
    }

    function loadAudio(epId, url, title, autoplay) {
      currentEpId = epId;
      audio.src = url;
      if (autoplay) {
        audio.play().catch(function(){});
      } else {
        audio.load();
      }
      speedIdx = 0; audio.playbackRate = 1;
      document.getElementById('console-speed-btn').textContent = '1.0x';

      // 载入新音频时，恢复自动滚动
      resumeSyncScroll();
    }

    function seekToParagraph(index) {
      if (playbackMode === 'sentence' && currentPlaylist) {
        var isPlaying = !audio.paused;
        loadChunk(index, isPlaying);
        resumeSyncScroll();
        return;
      }
      if (!audio.duration) return;
      var p = scriptParagraphs[index];
      if (!p) return;

      var targetTime;
      if (p.start !== null && p.start !== undefined) {
        targetTime = p.start;
      } else {
        var ratio = p.offset / scriptTotalChars;
        targetTime = audio.duration * ratio;
        targetTime = Math.max(0, targetTime - 0.2);
      }

      audio.currentTime = targetTime;
      if (audio.paused) {
        audio.play().catch(function(){});
      }

      // 用户点击跳转后，默认自动同步该句
      resumeSyncScroll();
    }

    function toggleAudio() {
      if (!currentEpId) return;
      if (audio.paused) {
        audio.play();
      } else {
        audio.pause();
      }
    }

    function skipAudio(s) {
      if (playbackMode === 'sentence' && currentPlaylist) {
        var totalDuration = currentPlaylist.reduce((acc, c) => acc + (c.duration || 0), 0);
        var curVirtualTime = (currentPlaylist[currentChunkIndex].start || 0) + audio.currentTime;
        var targetTime = Math.max(0, Math.min(totalDuration, curVirtualTime + s));

        // 寻找对应的音频片段
        for (var i = 0; i < currentPlaylist.length; i++) {
          var chunk = currentPlaylist[i];
          if (targetTime >= chunk.start && targetTime <= chunk.start + chunk.duration) {
            var isPlaying = !audio.paused;
            loadChunk(i, isPlaying);
            audio.currentTime = targetTime - chunk.start;
            break;
          }
        }
        resumeSyncScroll();
        return;
      }

      audio.currentTime = Math.max(0, Math.min(audio.duration||0, audio.currentTime + s));
      resumeSyncScroll();
    }

    function seekAudio(e) {
      if (playbackMode === 'sentence' && currentPlaylist) {
        var totalDuration = currentPlaylist.reduce((acc, c) => acc + (c.duration || 0), 0);
        if (!totalDuration) return;
        var track = document.getElementById('console-progress-track');
        var rect = track.getBoundingClientRect();
        var pct = (e.clientX - rect.left) / rect.width;
        var targetTime = totalDuration * pct;

        // 寻找对应的音频片段并跳转
        for (var i = 0; i < currentPlaylist.length; i++) {
          var chunk = currentPlaylist[i];
          if (targetTime >= chunk.start && targetTime <= chunk.start + chunk.duration) {
            var isPlaying = !audio.paused;
            loadChunk(i, isPlaying);
            audio.currentTime = targetTime - chunk.start;
            break;
          }
        }
        resumeSyncScroll();
        return;
      }

      if (!audio.duration) return;
      var track = document.getElementById('console-progress-track');
      var rect = track.getBoundingClientRect();
      var pct = (e.clientX - rect.left) / rect.width;
      audio.currentTime = audio.duration * pct;
      resumeSyncScroll();
    }

    function cycleSpeed() {
      speedIdx = (speedIdx + 1) % SPEEDS.length;
      audio.playbackRate = SPEEDS[speedIdx];
      bgmAudio.playbackRate = SPEEDS[speedIdx];
      var s = SPEEDS[speedIdx];
      document.getElementById('console-speed-btn').textContent = (s % 1 === 0 ? s + '.0' : s) + 'x';
    }

    function toggleMute() {
      isMuted = !isMuted;
      audio.muted = isMuted;
      bgmAudio.muted = isMuted;
      var icon = document.getElementById('volume-icon');
      var slider = document.getElementById('volume-slider');
      if (isMuted) {
        icon.innerHTML = '<path d="M11 5L6 9H2v6h4l5 4V5z"/><path d="M23 9l-6 6M17 9l6 6"/>';
        slider.value = 0;
      } else {
        icon.innerHTML = '<path d="M11 5L6 9H2v6h4l5 4V5z"/><path d="M15.54 8.46a5 5 0 0 1 0 7.07"/>';
        slider.value = lastVol;
      }
    }

    function changeVolume(v) {
      audio.volume = v;
      bgmAudio.volume = v * 0.05; // 背景音乐音量缩放
      lastVol = v;
      isMuted = (v == 0);
      audio.muted = isMuted;
      bgmAudio.muted = isMuted;
      var icon = document.getElementById('volume-icon');
      if (isMuted) {
        icon.innerHTML = '<path d="M11 5L6 9H2v6h4l5 4V5z"/><path d="M23 9l-6 6M17 9l6 6"/>';
      } else {
        icon.innerHTML = '<path d="M11 5L6 9H2v6h4l5 4V5z"/><path d="M15.54 8.46a5 5 0 0 1 0 7.07"/>';
      }
    }

    function fmtTime(s) {
      if (!s || isNaN(s)) return '0:00';
      var m = Math.floor(s/60), sec = Math.floor(s%60);
      return m + ':' + (sec<10?'0':'') + sec;
    }

    // 监听用户滚动剧本以实现人机共存控制
    function handleTranscriptScroll() {
      if (isInternalScroll) {
        isInternalScroll = false;
        return;
      }

      userScrolling = true;
      document.getElementById('back-to-sync-btn').classList.add('show');

      // 6秒内如果没有操作，自动恢复同步
      clearTimeout(userScrollTimer);
      userScrollTimer = setTimeout(function() {
        resumeSyncScroll();
      }, 6000);
    }

    function resumeSyncScroll() {
      userScrolling = false;
      document.getElementById('back-to-sync-btn').classList.remove('show');
      clearTimeout(userScrollTimer);
      forceSyncScroll();
    }

    function forceSyncScroll() {
      syncScrollTo(true);
    }

    function syncScrollTo(force) {
      if ((playbackMode !== 'sentence' && !audio.duration) || scriptParagraphs.length === 0) return;

      var currentTime = audio.currentTime;
      if (playbackMode === 'sentence' && currentPlaylist) {
        currentTime = (currentPlaylist[currentChunkIndex].start || 0) + audio.currentTime;
      }
      var activeIndex = 0;

      var hasTimestamps = scriptParagraphs[0].start !== null && scriptParagraphs[0].start !== undefined;
      if (hasTimestamps) {
        for (var i = 0; i < scriptParagraphs.length; i++) {
          var p = scriptParagraphs[i];
          if (currentTime >= p.start && currentTime < (p.start + p.duration)) {
            activeIndex = p.index;
            break;
          }
          if (currentTime < p.start) {
            activeIndex = Math.max(0, i - 1);
            break;
          }
          if (i === scriptParagraphs.length - 1) {
            activeIndex = i;
          }
        }
      } else {
        if (scriptTotalChars > 0) {
          var currentRatio = currentTime / audio.duration;
          var currentOffset = currentRatio * scriptTotalChars;
          for (var i = 0; i < scriptParagraphs.length; i++) {
            var p = scriptParagraphs[i];
            if (currentOffset >= p.offset && currentOffset < (p.offset + p.len)) {
              activeIndex = p.index;
              break;
            }
            if (currentOffset < p.offset) {
              activeIndex = Math.max(0, i - 1);
              break;
            }
            if (i === scriptParagraphs.length - 1) {
              activeIndex = i;
            }
          }
        }
      }

      document.querySelectorAll('.transcript-row').forEach(function(el) {
        el.classList.remove('speaking');
      });

      var activeEl = document.getElementById('trans-row-' + activeIndex);
      if (activeEl) {
        activeEl.classList.add('speaking');

        if (force || !userScrolling) {
          var container = document.getElementById('cast-panel-body');
          var relativeTop = activeEl.offsetTop - container.offsetTop;
          var containerScrollTop = container.scrollTop;
          var containerHeight = container.clientHeight;
          var activeHeight = activeEl.clientHeight;

          var margin = 60; // 60px padding from top and bottom edges
          var isFullyVisible = (relativeTop >= containerScrollTop + margin) &&
                               ((relativeTop + activeHeight) <= (containerScrollTop + containerHeight - margin));

          if (!isFullyVisible) {
            var targetScroll = relativeTop - (containerHeight / 2) + (activeHeight / 2);
            isInternalScroll = true;
            container.scrollTo({
              top: targetScroll,
              behavior: 'smooth'
            });
          }
        }
      }
    }

    audio.addEventListener('ended', function() {
      if (playbackMode === 'sentence' && currentPlaylist) {
        if (currentChunkIndex + 1 < currentPlaylist.length) {
          loadChunk(currentChunkIndex + 1, true);
        } else {
          audio.pause();
          bgmAudio.pause();
        }
      }
    });

    audio.addEventListener('play', function() {
      document.getElementById('console-btn-play').textContent = '⏸';
      document.getElementById('vinyl-disc').classList.add('spinning');
      if (playbackMode === 'sentence' && currentPlaylist) {
        bgmAudio.play().catch(function(){});
      }
    });
    audio.addEventListener('pause', function() {
      document.getElementById('console-btn-play').textContent = '▶';
      document.getElementById('vinyl-disc').classList.remove('spinning');
      if (playbackMode === 'sentence') {
        bgmAudio.pause();
      }
    });

    audio.addEventListener('timeupdate', function() {
      var curTime = audio.currentTime;
      var totalDur = audio.duration || 0;
      if (playbackMode === 'sentence' && currentPlaylist) {
        if (currentPlaylist[currentChunkIndex]) {
          curTime = (currentPlaylist[currentChunkIndex].start || 0) + audio.currentTime;
        }
        totalDur = currentPlaylist.reduce((acc, c) => acc + (c.duration || 0), 0);
      } else {
        if (!audio.duration) return;
      }

      var pct = totalDur > 0 ? (curTime / totalDur * 100).toFixed(1) : 0;

      var consoleFill = document.getElementById('console-progress-fill');
      if (consoleFill) consoleFill.style.width = pct + '%';

      var curTimeEl = document.getElementById('current-time');
      if (curTimeEl) curTimeEl.textContent = fmtTime(curTime);

      var totalTimeEl = document.getElementById('total-time');
      if (totalTimeEl && totalDur) totalTimeEl.textContent = fmtTime(totalDur);

      // 同步高亮及滚动
      syncScrollTo(false);
    });

    window.addEventListener('DOMContentLoaded', buildDatePills);
