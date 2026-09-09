# 主线加固 (Mainline Hardening) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复主线流程的结构性缺陷与脆点——report 不再阻塞发布、失败必有告警、失败自动重跑一次、发布前后双向对账、TTS 时长可观测——使"当天断播且无人知晓"不再可能。

**Architecture:** 只动 GitHub Actions 层(`daily.yml`/`prune_pages.yml`)与文档,不碰 Python 源码(零测试影响)。新增 `notify` 汇总 job(`if: always()`)做健康报告 + 开/关 issue 告警 + 自动重跑一次;publish 前加产物校验、部署后加 gh-pages 对账;TTS 与脚本质量以指标进入 Step Summary(观测先行,不盲改)。

**Tech Stack:** GitHub Actions / `gh` CLI(runner 预装)/ bash / python3(仅校验脚本)。

**范围裁剪(明确不做,及原因):**
- 单 LLM 双供应商降级(`llm_backends` 注册表已支持,但需要用户提供第二家 API key)——待定,等用户给 key。
- 素材源质量反馈回路(P2-8)——需要先积累数据,本期不做。
- TTS 合成策略改动——仅调查与观测,任何改动需用户基于调查结论另行拍板(敏感区)。
- 项目雷达——计划已存档于 `2026-09-09-project-radar.md`,主线修完再启动。

**已核实的事实:** `tts_backends/cosyvoice2.py:129,166` 显示分片风格变量来自 refs 配置;gh-pages 分片目录同时含 `chunk_XXX_lively.mp3` 与 `chunk_XXX_professional.mp3`,两者疑似对应两位主持人各自台词(A=苏晴/B=周航),即并非同一句话合成两遍——但需 Task 6 读取完整合成与拼装流程后定论。

---

## File Structure

| 文件 | 动作 | 职责 |
|---|---|---|
| `.github/workflows/daily.yml` | 修改 | 解耦 report/publish;发布前校验;部署后对账;新增 notify job(告警+自动重跑);TTS 时长指标;脚本质量统计 |
| `.github/workflows/prune_pages.yml` | 修改 | 失败告警 issue |
| `AGENTS.md` | 修改 | job 链语义、告警/重跑纪律说明 |

每个 workflow 任务用「python 断言」作为验收(先跑断言看失败 → 改 YAML → 跑断言看通过),保持 TDD 节奏。

---

### Task 1: 解耦 report 与 publish(P0-1)

**Files:**
- Modify: `.github/workflows/daily.yml`

- [ ] **Step 1: 写失败断言**

```bash
uv run python - <<'EOF'
import yaml
wf = yaml.safe_load(open(".github/workflows/daily.yml"))
pub = wf["jobs"]["publish"]
assert pub["needs"] == ["tts"], f"publish needs = {pub['needs']}"
EOF
```
Expected: FAIL —— `publish needs = ['tts', 'report']`

- [ ] **Step 2: 修改 daily.yml**

三处改动:

1) publish job 头部:
```yaml
  publish:
    name: Stage 5 (publish site)
    needs: [tts]
```

2) `Publish feed and site` step:
```yaml
        run: |
          EPISODE="${{ needs.tts.outputs.episode_id }}"
          uv run podcast-publish --date "$EPISODE"
```

3) `Commit site` step:
```yaml
          EPISODE="${{ needs.tts.outputs.episode_id }}"
          git add -u site/ data/episodes.json
          git add -f "site/reports/daily_report_${EPISODE}.md" 2>/dev/null || echo "No daily report for ${EPISODE}, skipped"
          git add -f "site/episodes/${EPISODE}/playlist.json" 2>/dev/null || echo "No playlist.json for ${EPISODE}, skipped"
```

(原 `needs.tts.outputs.episode_id || needs.report.outputs.episode_id` 双源表达式两处一并替换。)

- [ ] **Step 3: 跑断言确认通过**

Run: 同 Step 1 的断言。
Expected: PASS(无输出即通过)

- [ ] **Step 4: 验证 YAML 可解析并提交**

```bash
uv run python -c "import yaml; yaml.safe_load(open('.github/workflows/daily.yml')); print('OK')"
git add .github/workflows/daily.yml && git commit -m "fix(ci): decouple daily report from podcast publish"
```
Expected: `OK`,提交成功

---

### Task 2: 发布前产物校验(P2-9)

**Files:**
- Modify: `.github/workflows/daily.yml`(publish job)

- [ ] **Step 1: 写失败断言**

```bash
uv run python - <<'EOF'
import yaml
steps = yaml.safe_load(open(".github/workflows/daily.yml"))["jobs"]["publish"]["steps"]
assert any(s.get("name") == "Validate episode artifacts" for s in steps), "missing validation step"
EOF
```
Expected: FAIL —— `missing validation step`

- [ ] **Step 2: 插入校验 step**

在 `Publish feed and site` 之后、`Commit site` 之前插入:

```yaml
      - name: Validate episode artifacts
        env:
          TZ: Asia/Shanghai
        run: |
          EPISODE="${{ needs.tts.outputs.episode_id }}"
          python3 - "$EPISODE" <<'PYEOF'
          import json
          import sys
          from pathlib import Path

          ep = sys.argv[1]
          mp3 = Path(f"site/episodes/{ep}.mp3")
          assert mp3.exists(), f"missing {mp3}"
          size = mp3.stat().st_size
          assert size > 100_000, f"mp3 too small: {size} bytes"
          feed = Path("site/feed.xml").read_text(encoding="utf-8")
          assert f"episodes/{ep}.mp3" in feed, "feed.xml missing episode enclosure"
          index = json.loads(Path("data/episodes.json").read_text(encoding="utf-8"))
          eps = index if isinstance(index, list) else index.get("episodes", [])
          assert any(str(e.get("id")) == ep for e in eps), "episodes.json missing entry"
          print(f"validated {ep}: mp3={size} bytes, feed+index ok")
          PYEOF
```

- [ ] **Step 3: 跑断言确认通过 + YAML 解析**

```bash
uv run python - <<'EOF'
import yaml
steps = yaml.safe_load(open(".github/workflows/daily.yml"))["jobs"]["publish"]["steps"]
assert any(s.get("name") == "Validate episode artifacts" for s in steps)
print("OK")
EOF
uv run python -c "import yaml; yaml.safe_load(open('.github/workflows/daily.yml')); print('parse OK')"
```
Expected: `OK` + `parse OK`

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/daily.yml && git commit -m "fix(ci): validate episode artifacts before publish"
```

---

### Task 3: 部署后对账(P1-5)

**Files:**
- Modify: `.github/workflows/daily.yml`(publish job)

- [ ] **Step 1: 写失败断言**

```bash
uv run python - <<'EOF'
import yaml
steps = yaml.safe_load(open(".github/workflows/daily.yml"))["jobs"]["publish"]["steps"]
assert any(s.get("name") == "Verify gh-pages deployment" for s in steps), "missing reconciliation step"
EOF
```
Expected: FAIL

- [ ] **Step 2: 插入对账 step(置于 Deploy GitHub Pages 之后)**

```yaml
      - name: Verify gh-pages deployment
        env:
          GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          GH_REPO: ${{ github.repository }}
        run: |
          EPISODE="${{ needs.tts.outputs.episode_id }}"
          sleep 10
          SIZE=$(gh api "repos/$GH_REPO/contents/episodes/$EPISODE.mp3?ref=gh-pages" --jq '.size' 2>/dev/null || echo 0)
          echo "gh-pages $EPISODE.mp3 size: $SIZE"
          if [ "$SIZE" -le 100000 ]; then
            echo "::error::$EPISODE.mp3 missing or too small on gh-pages"
            exit 1
          fi
          PLAYLIST=$(gh api "repos/$GH_REPO/contents/episodes/$EPISODE/playlist.json?ref=gh-pages" --jq '.size' 2>/dev/null || echo 0)
          if [ "$PLAYLIST" -le 0 ]; then
            echo "::error::$EPISODE/playlist.json missing on gh-pages"
            exit 1
          fi
          echo "gh-pages reconciliation OK"
```

- [ ] **Step 3: 跑断言确认通过 + YAML 解析**(同 Task 2 Step 3 模式,断言 `Verify gh-pages deployment`)
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/daily.yml && git commit -m "fix(ci): reconcile gh-pages deployment after publish"
```

---

### Task 4: notify 汇总 job — 健康报告 + issue 告警 + 自动重跑一次(P0-2、P0-3)

**Files:**
- Modify: `.github/workflows/daily.yml`

- [ ] **Step 1: 写失败断言**

```bash
uv run python - <<'EOF'
import yaml
wf = yaml.safe_load(open(".github/workflows/daily.yml"))
assert "notify" in wf["jobs"], "missing notify job"
n = wf["jobs"]["notify"]
assert n.get("if") == "always()"
assert "issues" in (n.get("permissions") or {})
EOF
```
Expected: FAIL —— `missing notify job`

- [ ] **Step 2: workflow_dispatch 增加 retry 输入**

```yaml
workflow_dispatch:
  inputs:
    date:
      description: "Episode date (YYYY-MM-DD), default is today"
      required: false
      type: string
    retry:
      description: "Internal: auto-retry pass (do not set manually)"
      required: false
      type: boolean
      default: false
```

- [ ] **Step 3: 末尾追加 notify job**

```yaml
  notify:
    name: Health check & alert
    needs: [stage1, writer, report, tts, publish]
    if: always()
    runs-on: ubuntu-latest
    permissions:
      contents: read
      issues: write
      actions: write
    steps:
      - name: Summarize, alert and auto-retry
        env:
          GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          GH_REPO: ${{ github.repository }}
          RUN_URL: ${{ github.server_url }}/${{ github.repository }}/actions/runs/${{ github.run_id }}
          EPISODE: ${{ needs.stage1.outputs.episode_id }}
          S1: ${{ needs.stage1.result }}
          WRITER: ${{ needs.writer.result }}
          REPORT: ${{ needs.report.result }}
          TTS: ${{ needs.tts.result }}
          PUBLISH: ${{ needs.publish.result }}
          RETRY: ${{ github.event.inputs.retry }}
        run: |
          EPISODE="${EPISODE:-$(date -u +%F)}"
          FAILED=""
          if [ "$S1" != "success" ]; then FAILED="$FAILED stage1=$S1"; fi
          if [ "$WRITER" = "failure" ] || [ "$WRITER" = "cancelled" ]; then FAILED="$FAILED writer=$WRITER"; fi
          if [ "$REPORT" = "failure" ] || [ "$REPORT" = "cancelled" ]; then FAILED="$FAILED report=$REPORT"; fi
          if [ "$TTS" = "failure" ] || [ "$TTS" = "cancelled" ]; then FAILED="$FAILED tts=$TTS"; fi
          if [ "$PUBLISH" = "failure" ] || [ "$PUBLISH" = "cancelled" ]; then FAILED="$FAILED publish=$PUBLISH"; fi

          {
            echo "## Daily Podcast 健康报告 — $EPISODE"
            echo ""
            echo "| job | 结果 |"
            echo "|---|---|"
            echo "| stage1 | $S1 |"
            echo "| writer | $WRITER |"
            echo "| report | $REPORT |"
            echo "| tts | $TTS |"
            echo "| publish | $PUBLISH |"
            echo ""
            echo "Run: $RUN_URL"
          } >> "$GITHUB_STEP_SUMMARY"

          TITLE="⚠️ Daily pipeline failed: $EPISODE"
          if [ -n "$FAILED" ]; then
            BODY="日期: $EPISODE
          失败环节: $FAILED
          Run: $RUN_URL
          重试策略: $([ "$RETRY" = "true" ] && echo "本次已是自动重跑,不再继续重试,需要人工介入" || echo "将自动重跑一次")"
            FOUND=$(gh api "search/issues?q=repo:$GH_REPO+in:title+state:open+\"$TITLE\"" --jq '.items[0].number' 2>/dev/null || true)
            if [ -n "$FOUND" ]; then
              gh issue comment "$FOUND" --body "$BODY"
            else
              gh issue create --title "$TITLE" --body "$BODY"
            fi
            if [ "$RETRY" != "true" ]; then
              gh workflow run daily.yml --ref main -f date="$EPISODE" -f retry=true
              echo "已触发自动重跑: date=$EPISODE retry=true"
            fi
          else
            for N in $(gh api "search/issues?q=repo:$GH_REPO+in:title+state:open+\"Daily pipeline failed\"" --jq '.items[].number' 2>/dev/null || true); do
              gh issue close "$N" -c "今日运行全部成功,自动关闭。Run: $RUN_URL"
            done
          fi
```

要点:`actions: write` 是 `gh workflow run` 的前提;并发组保证重跑排在当前 run 结束之后;`retry=true` 的 run 失败时只告警不再递归。

- [ ] **Step 4: 跑断言确认通过 + YAML 解析**(断言 notify job / if: always() / permissions)
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add .github/workflows/daily.yml && git commit -m "feat(ci): add health notify with issue alert and one auto-retry"
```

---

### Task 5: TTS 时长指标 + 脚本质量统计(P1-4 观测、P2-7 轻量)

**Files:**
- Modify: `.github/workflows/daily.yml`

- [ ] **Step 1: 写失败断言**

```bash
uv run python - <<'EOF'
import yaml
steps = {s.get("name") for s in yaml.safe_load(open(".github/workflows/daily.yml"))["jobs"]["tts"]["steps"]}
assert any("Report synthesis duration" in n for n in steps if n), "missing tts duration metric"
wsteps = yaml.safe_load(open(".github/workflows/daily.yml"))["jobs"]["writer"]["steps"]
assert any(s.get("name") == "Script quality stats" for s in wsteps), "missing script stats"
print("OK")
EOF
```
Expected: FAIL

- [ ] **Step 2: tts job 末尾(Synthesize step 之后、Upload artifact 之前)加指标 step**

```yaml
      - name: Report synthesis duration
        if: always()
        run: |
          echo "TTS 用时: $((SECONDS / 60)) 分 $((SECONDS % 60)) 秒(episode ${{ needs.writer.outputs.episode_id }})" >> "$GITHUB_STEP_SUMMARY"
```

- [ ] **Step 3: writer job 的 Commit script 之后加质量统计 step(只警告不失败)**

```yaml
      - name: Script quality stats
        if: always()
        run: |
          EPISODE="${{ needs.stage1.outputs.episode_id }}"
          python3 - "$EPISODE" <<'PYEOF'
          import os
          import re
          import sys
          from pathlib import Path

          path = Path(f"site/episodes/{sys.argv[1]}.txt")
          if not path.exists():
              print(f"script missing: {path}")
              sys.exit(0)
          txt = path.read_text(encoding="utf-8")
          a = len(re.findall(r"^\[Host A\]", txt, re.M))
          b = len(re.findall(r"^\[Host B\]", txt, re.M))
          chars = len(re.sub(r"\s", "", txt))
          ratio = round(b / a, 2) if a else 0
          warns = []
          if chars < 1500:
              warns.append("too short")
          if a == 0 or b == 0:
              warns.append("missing host")
          if ratio and (ratio < 0.5 or ratio > 2.0):
              warns.append("host imbalance")
          line = f"script stats: {chars} chars, HostA={a}, HostB={b}, ratio={ratio}, warnings={warns or 'none'}"
          print(line)
          summary = os.environ.get("GITHUB_STEP_SUMMARY")
          if summary:
              with open(summary, "a", encoding="utf-8") as f:
                  f.write(f"- {line}\n")
          PYEOF
```

- [ ] **Step 4: 跑断言确认通过 + YAML 解析**
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add .github/workflows/daily.yml && git commit -m "feat(ci): surface tts duration and script quality stats"
```

---

### Task 6: TTS 合成流程调查(只调查,不改合成)

**Files:**
- Read: `src/ai_news_podcast/pipeline/tts_backends/cosyvoice2.py`、`scripts/gha_tts_cosyvoice.py`
- Read: gh-pages 任一期的 `playlist.json`(本地 `site/episodes/2026-09-07/playlist.json` 即可)
- Create: `docs/superpowers/plans/2026-09-09-tts-investigation-notes.md`(调查结论)

- [ ] **Step 1: 回答四个问题并落成文档**
  1. 每个 chunk 是否对每位主持人只合成一份?`_lively/_professional` 各对应谁?
  2. 最终 mp3 的拼装路径:哪些 chunk 文件被混入,顺序由什么决定(playlist.json?)。
  3. 时间都花在哪:模型加载(每次 run 冷启动?)、每 chunk 推理、后处理(loudnorm/BGM)——按 `2h28m` 的 run 估算各段占比。
  4. 可行的降时长选项清单(如:缓存命中时跳过重装、去掉冗余探针、并发合成),每项标注风险与预期收益。

- [ ] **Step 2: 向用户汇报调查结论,TTS 改动等待用户拍板**

不修改任何 TTS 代码。此项的产出是决策依据。

---

### Task 7: prune_pages 失败告警(P0-2 补充)

**Files:**
- Modify: `.github/workflows/prune_pages.yml`

- [ ] **Step 1: 写失败断言**

```bash
uv run python - <<'EOF'
import yaml
wf = yaml.safe_load(open(".github/workflows/prune_pages.yml"))
perms = wf.get("permissions") or {}
steps = wf["jobs"]["prune"]["steps"]
assert any(s.get("name") == "Alert on failure" for s in steps), "missing alert step"
assert "issues" in perms, "missing issues permission"
print("OK")
EOF
```
Expected: FAIL(先看断言失败再改)

- [ ] **Step 2: 修改**

顶层 permissions 改为:

```yaml
permissions:
  contents: write
  issues: write
```

job 末尾追加:

```yaml
      - name: Alert on failure
        if: failure()
        env:
          GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          GH_REPO: ${{ github.repository }}
        run: |
          TITLE="⚠️ gh-pages prune failed: $(date -u +%F)"
          BODY="Run: ${{ github.server_url }}/${{ github.repository }}/actions/runs/${{ github.run_id }}"
          FOUND=$(gh api "search/issues?q=repo:$GH_REPO+in:title+state:open+\"$TITLE\"" --jq '.items[0].number' 2>/dev/null || true)
          if [ -n "$FOUND" ]; then
            gh issue comment "$FOUND" --body "$BODY"
          else
            gh issue create --title "$TITLE" --body "$BODY"
          fi
```

- [ ] **Step 3: 跑断言确认通过 + YAML 解析**
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/prune_pages.yml && git commit -m "feat(ci): alert on gh-pages prune failure"
```

---

### Task 8: 文档同步与全量验证

**Files:**
- Modify: `AGENTS.md`

- [ ] **Step 1: AGENTS.md 同步**

`Gotchas` 中 job 链描述更新(现在写着 `stage1 → writer → report → tts → publish`),改为:

```markdown
- `.github/workflows/daily.yml` job 链:主线为 `stage1 → writer → tts → publish`;
  report 是旁路(失败只告警、不阻塞发布)。末尾 `notify` job(`if: always()`)
  汇总健康报告到 Step Summary,失败时开/评论 GitHub issue 并**自动重跑一次**
  (`-f retry=true`,重跑再失败则只告警);全部成功时自动关闭历史失败 issue。
  publish 在提交前做产物校验(mp3 尺寸/feed 条目/索引),部署后对账 gh-pages 上
  的 mp3 与 playlist.json。
```

- [ ] **Step 2: 全量验证**

```bash
uv run python -c "import yaml; [yaml.safe_load(open(f)) for f in ['.github/workflows/daily.yml', '.github/workflows/prune_pages.yml']]; print('yaml OK')"
uv run ruff check src/ tests/ scripts/ && uv run ruff format --check src/ tests/ scripts/ && uv run lint-imports && uv run pytest tests/ -q && uv run pre-commit run --all-files
```
Expected: 全部通过(源码未动,271 个测试不变)

- [ ] **Step 3: Commit**

```bash
git add AGENTS.md && git commit -m "docs: sync AGENTS.md with hardened daily pipeline"
```

- [ ] **Step 4: 真实验证路径披露(向用户说明)**

本计划全部改动只在 GHA 层,本地无法完整执行;首个真实验证是下一次定时运行(或手动 dispatch 一次观察 notify 输出)。今晚的 run 将首次出现健康报告与告警机制。

---

## Self-Review 记录

1. **问题覆盖**:P0-1 解耦(Task 1 ✓)、P0-2 告警(Task 4/7 ✓)、P0-3 自动重跑一次(Task 4 ✓,防递归已处理)、P1-4 观测+调查(Task 5/6 ✓,不盲改)、P1-5 双向对账(Task 2/3 ✓)、P2-7 轻量统计(Task 5 ✓)、P2-9 发布前校验(Task 2 ✓);P1-6 双 LLM 供应商与 P2-8 素材反馈明确裁剪并说明原因 ✓。
2. **占位符**:无;所有 YAML/bash/python 代码完整给出。
3. **类型/引用一致性**:`needs.tts.outputs.episode_id` 在 Task 1/2/3 一致;notify 的 needs 列表与 job 名一致;retry 输入名在 dispatch 与 notify 引用一致(`retry`)。
4. **风险点**:notify 的 issue 搜索走 title 匹配,同一日期重复失败会追加评论而非开新 issue(符合预期);`gh workflow run` 需要 `actions: write` 已在 job permissions 中;发布前校验失败会触发自动重跑,若 mp3 本身损坏则重跑仍失败并最终只告警(符合"重跑一次"语义)。
