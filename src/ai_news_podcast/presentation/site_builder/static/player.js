(function () {
  'use strict';

  var SPEEDS = [1.0, 1.25, 1.5, 1.75, 2.0];
  var DEFAULT_VOLUME = 0.8;

  class PodcastEngine {
    constructor(audio) {
      this.audio = audio;
      this.chunks = [];
      this.index = 0;
      this.voices = { A: null, B: null };
      this.rateIndex = 0;
      this.volume = DEFAULT_VOLUME;
      this.lastVolume = DEFAULT_VOLUME;
      this._handlers = { state: [], time: [], duration: [] };
      audio.volume = this.volume;
      audio.addEventListener('play', () => this._emitState(true));
      audio.addEventListener('pause', () => this._emitState(false));
      audio.addEventListener('timeupdate', () => this._emitTime());
      audio.addEventListener('loadedmetadata', () => this._emitTime());
      audio.addEventListener('ended', () => this._onChunkEnded());
    }

    /* ---------- 事件订阅 ---------- */
    onState(handler) { this._handlers.state.push(handler); }
    onTime(handler) { this._handlers.time.push(handler); }
    onDuration(handler) { this._handlers.duration.push(handler); }
    _emitState(playing) { this._handlers.state.forEach(function (fn) { fn(playing); }); }
    _emitTime() {
      var current = this.currentTime;
      var duration = this.duration;
      this._handlers.time.forEach(function (fn) { fn(current, duration); });
    }

    /* ---------- 状态 ---------- */
    get duration() {
      if (!this.chunks.length) return 0;
      var last = this.chunks[this.chunks.length - 1];
      return Number(last.start || 0) + Number(last.duration || 0);
    }

    get currentTime() {
      var chunk = this.chunks[this.index];
      if (!chunk) return 0;
      return Number(chunk.start || 0) + (this.audio.currentTime || 0);
    }

    get playing() { return !this.audio.paused && !!this.audio.currentSrc; }

    /* ---------- 数据装载 ---------- */
    load(chunks, episodeDate) {
      this.chunks = chunks || [];
      this._episodeDate = episodeDate || '';
      this.index = 0;
      this.audio.pause();
      this.audio.removeAttribute('src');
      this.audio.load();
      if (this.chunks.length && this._episodeDate) this._setChunk(0, false);
      this._emitTime();
    }

    /* ---------- 播放控制 ---------- */
    toggle() { if (this.playing) { this.pause(); } else { this.play(); } }

    play() {
      if (!this.chunks.length) return Promise.resolve();
      return this.audio.play().catch(function (error) {
        console.warn('audio play blocked or failed', error);
      });
    }

    pause() { this.audio.pause(); }

    seek(globalSeconds, autoplay) {
      if (!this.chunks.length) return;
      var target = Math.max(0, Math.min(this.duration, Number(globalSeconds) || 0));
      var index = this._indexAt(target);
      var offset = target - Number(this.chunks[index].start || 0);
      var shouldPlay = autoplay === undefined ? this.playing : autoplay;
      this._setChunk(index, false);
      this.audio.addEventListener('loadedmetadata', () => {
        this.audio.currentTime = Math.max(0, offset);
      }, { once: true });
      this.audio.currentTime = Math.max(0, offset);
      if (shouldPlay) this.play();
      this._emitTime();
    }

    skip(delta) {
      if (!this.chunks.length) return;
      this.seek(this.currentTime + Number(delta), this.playing);
    }

    /* ---------- 音色 / 倍速 / 音量 ---------- */
    setVoice(host, variant) {
      this.voices[host] = variant;
      if (this.chunks[this.index] && this.chunks[this.index].host === host) {
        var wasPlaying = this.playing;
        this._setChunk(this.index, false);
        if (wasPlaying) this.play();
      }
    }

    voiceOf(host) {
      var current = this.chunks[this.index];
      if (current && current.host === host && this.voices[host]) return this.voices[host];
      return this.voices[host];
    }

    cycleRate() {
      this.rateIndex = (this.rateIndex + 1) % SPEEDS.length;
      this.audio.playbackRate = SPEEDS[this.rateIndex];
      return SPEEDS[this.rateIndex].toFixed(this.rateIndex % 2 === 0 ? 1 : 2) + 'x';
    }

    setVolume(volume) {
      this.volume = Math.max(0, Math.min(1, Number(volume) || 0));
      this.audio.volume = this.volume;
      if (this.volume > 0) this.lastVolume = this.volume;
    }

    toggleMute() {
      if (this.audio.volume > 0) {
        this.lastVolume = this.audio.volume;
        this.audio.volume = 0;
        return true;
      }
      this.audio.volume = this.lastVolume || DEFAULT_VOLUME;
      return false;
    }

    /* ---------- 内部 ---------- */
    _indexAt(globalSeconds) {
      for (var i = this.chunks.length - 1; i >= 0; i -= 1) {
        if (globalSeconds >= Number(this.chunks[i].start || 0)) return i;
      }
      return 0;
    }

    _audioFile(chunk, index) {
      var variant = this.voices[chunk.host];
      var audios = chunk.audios || {};
      var file = (variant && audios[variant]) || audios.professional || Object.values(audios)[0];
      if (file) return file;
      if (chunk.audio) return chunk.audio;
      return 'chunk_' + String(index + 1).padStart(3, '0') + '.mp3';
    }

    _setChunk(index, autoplay) {
      if (!this.chunks.length) return;
      this.index = Math.max(0, Math.min(this.chunks.length - 1, index));
      var chunk = this.chunks[this.index];
      this.audio.src = './episodes/' + this._episodeDate + '/' + this._audioFile(chunk, this.index);
      this.audio.playbackRate = SPEEDS[this.rateIndex];
      if (autoplay) this.play();
      this._emitTime();
    }

    _onChunkEnded() {
      if (this.index < this.chunks.length - 1) {
        this._setChunk(this.index + 1, true);
      } else {
        this._emitState(false);
      }
    }

  }

  window.PodcastEngine = PodcastEngine;
})();
