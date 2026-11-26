# 标注金标抽检（30 条）

> 按「规则与 LLM 金标的分歧」分组，每组内按分歧度降序。
> 每条在「判定」处填 `对`（金标正确）/ `错`（金标有误）/ `部分`，并写一句备注。
> **你只需判断「LLM 金标这一栏是否合理」，不必逐字校对规则。**
> 填完运行 `uv run python scripts/label_spotcheck.py --apply` 回写。

## 分组概览

| 分组 | 条数 | 含义 |
|---|---|---|
| 主体完全错配 | 12 | 规则标的主体与金标完全不重合，通常是规则误命中 |
| 主体漏标 | 5 | 规则没标出主体 |
| 技术过度标注 | 9 | 规则标的技术比金标多 3 个以上，多为裸词误命中 |
| 类别过度标注 | 3 | 规则类别比金标多 3 个以上 |
| 其他分歧 | 1 | 差异较小或方向混合 |

## 主体完全错配（12 条）

### 1. 苹果罕见公布人形机器人进展,宇树 G1 背着一台跑千问的 Mac mini

- **ID**：`054374d2`　**分歧度**：10
- **摘要**：终于不是后空翻、跳舞、跑步了 #欢迎关注爱范儿官方微信公众号:爱范儿(微信号:ifanr),更多精彩内容第一时间为您奉上。

| | 主体 | 技术 | 类别 |
|---|---|---|---|
| **LLM 金标** | 苹果, HALO, 宇树 G1, Mac mini M4 Pro, 千问模型 | 具身智能, 视觉-语言-动作（VLA）, 运动扩散模型, 仿真合成数据, 程序化场景生成 | 研究成果, 人形机器人, 合成训练数据 |
| **规则产出** | NVIDIA, 阿里巴巴, Apple, 通义千问, 宇树科技 | robot, llm, training, data, chip, vision, inference, open_source, reasoning | research, product, technology, industry |

- **判定**：　　（填 `对` / `错` / `部分`）
- **备注**：

### 2. 重磅!微软游说白宫搁置光模块禁令

- **ID**：`0999b939`　**分歧度**：10
- **摘要**：一家美国科技巨头,正为保住中国供应链而奔走。 微软,美国科技霸权的代表企业之一,正在向特朗普政府施压—要求搁置一项禁止进口中国 光模块 的计划。 据路透社 11 月 10 日独家报道,三名知情人士称,微软今年 9 月已与白宫科技政策办公室(OSTP)及国家安全委员会(NSC)官员会面,警告潜在禁令可能引发的供应链问题。理由很直接:禁了中国光模块,美国 AI …

| | 主体 | 技术 | 类别 |
|---|---|---|---|
| **LLM 金标** | 微软, 白宫, OSTP, NSC, FCC | 光模块, AI数据中心, AI芯片, 光纤互联, 光通信 | 监管政策, 游说, 供应链, 中美科技管制, 光模块禁令 |
| **规则产出** | Microsoft, NVIDIA | safety, data, chip, inference | policy, safety, business, capital, research, product, technology, industry |

- **判定**：　　（填 `对` / `错` / `部分`）
- **备注**：

### 3. 曾号称订单超百万,累计仅卖 6.3 万辆,量产不到四年的特斯拉 Cybertruck 官宣停产

- **ID**：`0564958a`　**分歧度**：9
- **摘要**：长得像未来,但没能活到未来。 #欢迎关注爱范儿官方微信公众号:爱范儿(微信号:ifanr),更多精彩内容第一时间为您奉上。

| | 主体 | 技术 | 类别 |
|---|---|---|---|
| **LLM 金标** | 特斯拉, Cybertruck, 马斯克, SpaceX, Cybercab | 纯电动汽车, 自动驾驶, 48V低压架构, 线控转向, 4680电池 | 产品停产, 销量下滑, 法规合规, 供应链问题, 商业表现 |
| **规则产出** | Elon Musk, Tesla | autonomous_driving, vision, safety, data, robot | safety, business, capital, product, technology, industry |

- **判定**：　　（填 `对` / `错` / `部分`）
- **备注**：

### 4. 森泰股份:控股股东、实际控制人增持计划实施完成,累计增持56.42万股

- **ID**：`135b494b`　**分歧度**：9
- **摘要**：森泰股份公告,公司控股股东、实际控制人唐道远、张勇、王斌此前计划自2026年5月13日起6个月内以自有资金增持公司股份,合计增持金额600万元至1200万元。截至2026年11月13日,本次增持计划已实施完成。2026年5月14日至2026年11月13日期间,上述增持主体通过集中竞价交易方式累计增持公司股份56.42万股,占公司总股本的0.48%,增持股份金…

| | 主体 | 技术 | 类别 |
|---|---|---|---|
| **LLM 金标** | 森泰股份, 唐道远, 张勇, 王斌 | （空） | 股份增持, 公司公告 |
| **规则产出** | OpenAI, 阿里巴巴, Elon Musk | robot, cloud, safety | policy, safety, capital, research, product, industry |

- **判定**：　　（填 `对` / `错` / `部分`）
- **备注**：

### 5. 英国萨里大学等提出NitroFusion2:首次将对抗判别引入MMDiT文本流,刷新1步极速图像生成纪录

- **ID**：`05e39210`　**分歧度**：8
- **摘要**：多模态文本流判别新范式

| | 主体 | 技术 | 类别 |
|---|---|---|---|
| **LLM 金标** | 英国萨里大学, 图灵精英研究实验室, NitroFusion2, NitroFusion | 多模态扩散 Transformer（MMDiT）, 少步扩散模型蒸馏, 对抗蒸馏, 文本流对抗蒸馏（TSAD）, 联合注意力 | 研究成果, 图像生成, 扩散模型蒸馏 |
| **规则产出** | NVIDIA, 通义千问, Figure AI | vision, multimodal | policy, safety, research, hardware, open_source, product, technology, application |

- **判定**：　　（填 `对` / `错` / `部分`）
- **备注**：

### 6. Agent 从 Demo 到生产,究竟还差几步?4周实战课来了

- **ID**：`058acbf9`　**分歧度**：7
- **摘要**：造一个 Agent,正在变得越来越容易。 借助大模型、开发框架和各类工具,开发者可以快速搭建一个能够理解指令、调用工具、执行任务的 Agent。 但 Agent 能跑起来,就意味着可以投入生产了吗? 模型升级后,如何判断效果是变好了还是变差了?面对复杂任务,如何保证运行安全?当调用量持续增长,Token 成本又该如何控制? 从一个能运行的 Demo,到稳定、…

| | 主体 | 技术 | 类别 |
|---|---|---|---|
| **LLM 金标** | 亚马逊云科技, Agent 创造季之 Agent 速造局 4×1 实战课 | Agent, 大模型, 工具调用, Prompt | Agent工程实践, 课程推广, 安全防护, 成本治理 |
| **规则产出** | Amazon | agent, llm, safety, inference, data | safety, business, product, technology |

- **判定**：　　（填 `对` / `错` / `部分`）
- **备注**：

### 7. Docker 呼吁通过隔离与治理保障智能体安全

- **ID**：`10f8805b`　**分歧度**：7
- **摘要**：Docker 发布了一篇指南,介绍当 AI 智能体在人工监督有限的情况下浏览网页、运行代码、调用工具时会产生的五大安全风险。Docker 首席产品营销经理 Srini Sekaran 与产品经理 Eric Jia 联合撰文指出:让智能体具备价值的自主能力同时也导致传统应用安全控制措施变得不足。Docker 将隔离和治理(而非仅靠模型改进)作为主要的防御手段。…

| | 主体 | 技术 | 类别 |
|---|---|---|---|
| **LLM 金标** | Docker, Srini Sekaran, Eric Jia, OWASP | AI 智能体, 工具调用, 代码执行, 提示词注入, 隔离与治理 | 安全问题, 安全指南, 风险分析 |
| **规则产出** | Google, Amazon, DeepMind, Meta | agent, safety, llm, data | safety, product, technology, application |

- **判定**：　　（填 `对` / `错` / `部分`）
- **备注**：

### 8. v0.2.2-rc.3

- **ID**：`04e106d0`　**分歧度**：6
- **摘要**：中文 | English 新增功能 新增实验性「DeepSeek 会话导入」插件,支持将 DeepSeek App 或网页版的历史会话导入工作区,并继续对话。 @imccyu 问题修复 修复 Windows 桌面版应用内更新可能失败、无法自动重新打开的问题。 @winewill New Features Add an experimental DeepSee…

| | 主体 | 技术 | 类别 |
|---|---|---|---|
| **LLM 金标** | v0.2.2-rc.3, DeepSeek App, Windows, @imccyu, @winewill | LLM, 插件, 会话导入 | 版本发布, 功能更新, 问题修复 |
| **规则产出** | DeepSeek, Anthropic | agent, safety | product, research, safety, application |

- **判定**：　　（填 `对` / `错` / `部分`）
- **备注**：

### 9. What Codex Unlocks for Holiday Extras

- **ID**：`0507060b`　**分歧度**：6
- **摘要**：What happens when the people who understand your customers can build the software to help them? At Holiday Extras, Codex is helping turn that ambition into reality—from rebuilding …

| | 主体 | 技术 | 类别 |
|---|---|---|---|
| **LLM 金标** | Holiday Extras, Codex, Matthew Pack, David Lee, Liam Thompson | AI coding assistant, LLM, Agent | Enterprise AI adoption, Software development |
| **规则产出** | OpenAI | safety | business, product |

- **判定**：　　（填 `对` / `错` / `部分`）
- **备注**：

### 10. Runway's open-source Praxis-1 turns video AI into a robot controller

- **ID**：`05df9e0e`　**分歧度**：6
- **摘要**：AI company Runway has released its robot model Praxis-1 with open weights and source code. The model uses what it learned from video to control robots and shows promising results i…

| | 主体 | 技术 | 类别 |
|---|---|---|---|
| **LLM 金标** | Runway, Praxis-1 | Multimodal, World Model, Video Model, Fine-tuning, Robot Control | Model Release, Open-Source Release, Robotics |
| **规则产出** | Google, NVIDIA, Hugging Face, Arm | open_source, robot | research, open_source, technology |

- **判定**：　　（填 `对` / `错` / `部分`）
- **备注**：

### 11. 100轮图像编辑这道「毙刷题」,新出的Seedream 5.5稳上来了!

- **ID**：`09116097`　**分歧度**：6
- **摘要**：速度提了,API价格降了

| | 主体 | 技术 | 类别 |
|---|---|---|---|
| **LLM 金标** | Seedream 5.5, 字节, 豆包, GPT-Image 2.6, Seedream 5.0 Pro | 图像生成, 图像编辑, 多轮一致性, 多语言文本渲染, 透明背景生成 | 模型发布, 模型评测, API降价 |
| **规则产出** | 字节跳动 | llm, vision, world_model, multimodal | product, technology, application, industry |

- **判定**：　　（填 `对` / `错` / `部分`）
- **备注**：

### 12. 北汽极狐贝塔和宁德时代合作换电版车型上市

- **ID**：`0f4f28bd`　**分歧度**：6
- **摘要**：36氪获悉,北汽极狐贝塔和宁德时代合作换电版车型上市,S3车型5.83万元起,T1车型5.2万元起。家庭包月租399元、含5000公里额度;畅行包月租599元,适合里程需求大的用户。

| | 主体 | 技术 | 类别 |
|---|---|---|---|
| **LLM 金标** | 北汽极狐贝塔, 宁德时代, S3车型, T1车型 | 换电技术 | 车型上市, 商业合作 |
| **规则产出** | NVIDIA, 阿里巴巴, Elon Musk, 黄仁勋 | inference, chip, cloud, safety | capital |

- **判定**：　　（填 `对` / `错` / `部分`）
- **备注**：

## 主体漏标（5 条）

### 13. 官宣承接日系光学名企整厂,盖泽科技能换来产能吗?

- **ID**：`0498d89e`　**分歧度**：8
- **摘要**：近日,盖泽精密科技(苏州)有限(简称“盖泽科技”)官微对外披露,公司全盘承接日本 光学 巨头USHIO(牛尾)苏州厂区,待厂区升级改造完成后,这片厂区将成为盖泽科技的核心产业载体。 盖泽科技是一家怎样的光学企业?为什么在这个节点盘下一座日系工厂? 盖泽科技成立于2021年,2025年完成近亿元B轮产业融资。据公司披露,其核心光学部件与量测系统实现自研,FTI…

| | 主体 | 技术 | 类别 |
|---|---|---|---|
| **LLM 金标** | 盖泽科技, USHIO（牛尾） | FTIR, 膜厚量测, 光学部件, 晶圆量测, 精密仪器制造 | 产能扩张, 并购重组, 融资, 国产替代, 量产交付 |
| **规则产出** | （空） | chip | business, capital, hardware, product, infrastructure, technology, application, industry |

- **判定**：　　（填 `对` / `错` / `部分`）
- **备注**：

### 14. Towards Better Legal Reasoning LLMs: Signal Balancing and Reward Scheduling

- **ID**：`00b8d75e`　**分歧度**：6
- **摘要**：arXiv:2611.15346v1 Announce Type: new Abstract: In the legal domain, LLMs can improve reliability and interpretability by generating explicit reasoning traces. However, training LL…

| | 主体 | 技术 | 类别 |
|---|---|---|---|
| **LLM 金标** | JurisCoT-SFT, Lifecycle-Aware Backtrackable Policy Optimization | LLM, supervised fine-tuning, reinforcement learning, Group Relative Policy Optimization, reward scheduling | research findings, legal reasoning, LLM training |
| **规则产出** | （空） | reasoning, training, reinforcement | policy, technology |

- **判定**：　　（填 `对` / `错` / `部分`）
- **备注**：

### 15. 倒计时一周|量子位2026人工智能年度榜单征集即将截止!

- **ID**：`00c861ac`　**分歧度**：6
- **摘要**：量子位在寻找“能干”的答案

| | 主体 | 技术 | 类别 |
|---|---|---|---|
| **LLM 金标** | 量子位 | 人工智能 | 榜单征集, 评选活动 |
| **规则产出** | （空） | （空） | business, capital, product, technology, application, industry |

- **判定**：　　（填 `对` / `错` / `部分`）
- **备注**：

### 16. 倒计时一周|量子位2026人工智能年度榜单征集即将截止!

- **ID**：`059db3b1`　**分歧度**：6
- **摘要**：量子位在寻找“能干”的答案

| | 主体 | 技术 | 类别 |
|---|---|---|---|
| **LLM 金标** | 量子位 | 人工智能 | 榜单征集, 评选活动 |
| **规则产出** | （空） | （空） | business, capital, product, technology, application, industry |

- **判定**：　　（填 `对` / `错` / `部分`）
- **备注**：

### 17. 成都站-Schrödinger计算化学驱动的药物发现全流程实战研修班

- **ID**：`0a8d7323`　**分歧度**：6
- **摘要**：前言 Schrödinger 套件 是行业广泛使用的 集成化计算药物设计平台 ,其独特优势在于 深度融合了基于物理的精确算法(如FEP+自由能微扰、Desmond分子动力学、Glide对接)与数据驱动的机器学习模型 ,同时覆盖从靶点结构预测(同源建模、口袋检测)到虚拟筛选(对接、药效团、形状匹配),再到先导优化(Ligand Designer、骨架跃迁)和最…

| | 主体 | 技术 | 类别 |
|---|---|---|---|
| **LLM 金标** | Schrödinger 套件, 源资信息科技（上海）有限公司, 成都生物与医药产业孵化园（天河园）, Ligand Designer, IPA | 计算药物设计, 机器学习, 分子动力学, 自由能微扰, 虚拟筛选 | 培训活动, 药物发现, 计算药物设计, 分子对接, 智能分子生成 |
| **规则产出** | （空） | data, llm | research, technology, application, industry |

- **判定**：　　（填 `对` / `错` / `部分`）
- **备注**：

## 技术过度标注（9 条）

### 18. Introducing Claude Fable 5.2 and Claude Mythos 5.2

- **ID**：`0278d449`　**分歧度**：15
- **摘要**：Introduction Performance and efficiency Coding Cybersecurity Science and engineering Safety and alignment Data privacy Availability November 9, 2026 Claude Fable 5.2 and Mythos 5.2…

| | 主体 | 技术 | 类别 |
|---|---|---|---|
| **LLM 金标** | Claude Fable 5.2, Claude Mythos 5.2, Anthropic | LLM, Agent, 多智能体协作, 形式化验证, 递归自我改进 | 模型发布, 性能与效率, 研究成果, 安全对齐, 智能体工程 |
| **规则产出** | Anthropic, Google, Microsoft, NVIDIA, 月之暗面 | agent, inference, open_source, reasoning, robot, safety, coding, llm, training, chip, speech, data, vision | business, research, technology, safety, capital |

- **判定**：　　（填 `对` / `错` / `部分`）
- **备注**：

### 19. 英伟达首席执行官黄仁勋辞职

- **ID**：`091c1726`　**分歧度**：12
- **摘要**：36氪获悉,英伟达董事会今日宣布,黄仁勋(Jensen Huang)已辞去总裁、首席执行官以及董事职务,辞呈即刻生效。 公司现任执行副总裁兼首席财务官科莱特·克雷斯(Colette Kress)将出任临时首席执行官。在正式继任者确定前,克雷斯将同时兼任执行副总裁、首席财务官与临时CEO。董事会已启动常设首席执行官的遴选工作,内部、外部候选人都将纳入考量范围。…

| | 主体 | 技术 | 类别 |
|---|---|---|---|
| **LLM 金标** | 英伟达, 黄仁勋, 科莱特·克雷斯, 哈维·C·琼斯 | 人工智能芯片 | 人事变动 |
| **规则产出** | NVIDIA, Microsoft, 黄仁勋 | chip, world_model, llm, cloud, safety | hardware, policy, safety, capital, research, product, technology, industry |

- **判定**：　　（填 `对` / `错` / `部分`）
- **备注**：

### 20. Cohere Expands Its Sovereign AI Platform with AISLE and cubic

- **ID**：`0e08870b`　**分歧度**：12
- **摘要**：Expanding the sovereign AI platform with new capabilities for cyber defense, secure software development, and mission-critical enterprise workflows Over the past year, we've seen a…

| | 主体 | 技术 | 类别 |
|---|---|---|---|
| **LLM 金标** | Cohere, AISLE, cubic | AI, sovereign AI, autonomous systems, AI-powered code review | Acquisition, Platform expansion, Cybersecurity, Enterprise AI |
| **规则产出** | Cohere | edge, data, training, coding | policy, safety, business, capital, research, infrastructure, technology, industry |

- **判定**：　　（填 `对` / `错` / `部分`）
- **备注**：

### 21. 都2026了,腾讯居然又做了个搜索引擎?!

- **ID**：`066d3e3b`　**分歧度**：11
- **摘要**：广告可以折叠,权威信源优先

| | 主体 | 技术 | 类别 |
|---|---|---|---|
| **LLM 金标** | 腾讯, 元宝搜索, 李锐章 | AI, Agent, 网页搜索 | 产品发布, AI搜索, 搜索引擎 |
| **规则产出** | 腾讯 | agent, data, open_source, speech, llm | policy, business, research, open_source, product, technology, application, industry |

- **判定**：　　（填 `对` / `错` / `部分`）
- **备注**：

### 22. Fable 5.2登场:解出两道黑洞物理难题,30小时完成费马大定理形式化

- **ID**：`0098975a`　**分歧度**：9
- **摘要**：当地时间 11 月 9 日,Anthropic 发布 Claude Fable 5.2 与 Claude Mythos 5.2。 两者沿用此前“同一底座、两套防护”的格局:Fable 5.2 面向所有付费用户和开发者开放,内置网络安全与生物学安全分类器;Mythos 5.2 放宽这两个领域的限制,只通过生命科学和网络安全两项审核计划提供给经过审查的机构。 距…

| | 主体 | 技术 | 类别 |
|---|---|---|---|
| **LLM 金标** | Anthropic, Claude Fable 5.2, Claude Mythos 5.2, Claude Opus 5.5, GPT-6 Astra | 大语言模型（LLM）, 推理模型, 智能体（Agent）, 代码生成, 安全分类器 | 模型发布, 研究成果, 基准测试, 定价调整, IPO |
| **规则产出** | Anthropic, NVIDIA, OpenAI | safety, llm, robot, open_source, reasoning, data, agent, speech, inference, training | safety, capital, research, product, technology |

- **判定**：　　（填 `对` / `错` / `部分`）
- **备注**：

### 23. 黄仁勋辞职:把扁平化推到极致的创始人,体面退场的代价

- **ID**：`0e0cfc5f`　**分歧度**：9
- **摘要**：当地时间11月12日,英伟达宣布黄仁勋即日起辞去总裁、首席执行官及董事会成员职务;克雷斯继续兼任执行副总裁兼首席财务官,董事会已启动永久CEO遴选,将考虑内部及外部候选人。 注意这个措辞——“辞去”,而非“被罢免”。董事会最终没有走到投票撤换那一步,但结果一样:那个三十多年一手把英伟达拉扯大的人,走了。 而他能以“辞任”而不是“罢免”体面离场,根子上恰恰要追…

| | 主体 | 技术 | 类别 |
|---|---|---|---|
| **LLM 金标** | 英伟达, 黄仁勋, 克雷斯 | AI算力, 芯片 | 人事变动, 公司治理, 组织架构, 监管政策 |
| **规则产出** | NVIDIA, 黄仁勋 | chip, safety, llm, training, inference, coding, robot | technology, application |

- **判定**：　　（填 `对` / `错` / `部分`）
- **备注**：

### 24. 智能体 AI 架构中记忆、知识与上下文的系统化方案

- **ID**：`04f3045e`　**分歧度**：8
- **摘要**：随着行业从聊天机器人向自主智能体演进,架构层面的瓶颈正在发生转移。难点不再仅仅是数据检索,而是决定智能体应该知道什么、何时应该知道、应该记住多久,以及应该遗忘什么。和人类一样,“知道”是一回事,“记住”是另一回事。由此有了记忆与知识的概念。根据任务上下文判断何时动用现有的各种“知道”并将其调入记忆才是关键。例如,如果我只是想煎个蛋饼,那么此刻并不需要记住如何…

| | 主体 | 技术 | 类别 |
|---|---|---|---|
| **LLM 金标** | （空） | 智能体, 大语言模型, 上下文工程, 记忆系统, 检索增强生成 | 智能体架构, 上下文工程, 大语言模型幻觉, 知识与记忆管理 |
| **规则产出** | （空） | agent, robot, data, reasoning, llm, safety | product, technology, industry |

- **判定**：　　（填 `对` / `错` / `部分`）
- **备注**：

### 25. AMD 亮出本地编程平台 Instinct Coder 355:一台机器塞进 98% 前沿模型

- **ID**：`07d22ad4`　**分歧度**：6
- **摘要**：把大模型从云端拽回机房,AMD 这次做的是企业开发者的生意。当地时间 11 月 9 日,AMD 携手 Spectro Cloud 推出企业级 AI 辅助编程平台 Instinct Coder 355,它能在本地跑起 GLM-5.3 推理,且据称逼近前沿模型 98% 的性能。 八张卡撑起一台编程服务器 这套系统的硬核在于堆料:核心是 8 张 AMD Insti…

| | 主体 | 技术 | 类别 |
|---|---|---|---|
| **LLM 金标** | AMD, Spectro Cloud, Instinct Coder 355, GLM-5.3, AMD Instinct MI355X | 大语言模型（LLM）, AI辅助编程, 本地推理, 私有化部署, 推理加速 | 产品发布, 企业级AI基础设施, AI辅助编程, 本地化部署, 性能评测 |
| **规则产出** | NVIDIA, AMD | coding, llm, reasoning, chip, data, inference | policy, product, technology |

- **判定**：　　（填 `对` / `错` / `部分`）
- **备注**：

### 26. 可用内存多3GB!华为详解麒麟9050系全新灵犀CPU:多核暴涨52%

- **ID**：`0f2b099e`　**分歧度**：6
- **摘要**：快科技11月12日消息,华为麒麟官方公众号今天发文,详解麒麟9050 Pro搭载的新一代灵犀CPU架构。 这套全新设计跳出传统手机SoC核组思路,以用户场景为先重新规划CPU调度,带来9核16线程的规格,首次在手机端引入SMT超线程。 灵犀CPU采用差异化核簇排布,1颗Prime超大核专门扛重载场景,游戏运行、大文件加载、应用启动这类高压力任务优先交给它,搭…

| | 主体 | 技术 | 类别 |
|---|---|---|---|
| **LLM 金标** | 华为, 麒麟9050 Pro, 灵犀CPU | SMT超线程, CPU任务调度, 缓存架构, FFRT并行架构, 内存压缩 | 芯片架构, 产品发布, 性能优化 |
| **规则产出** | 华为 | edge, chip, data | research, hardware, product, technology, application |

- **判定**：　　（填 `对` / `错` / `部分`）
- **备注**：

## 类别过度标注（3 条）

### 27. 谷歌最强模型 Gemini 4 Carbon 全面开放:一夜重写 OpenCode 桌面应用、价格仅为 Fable 5.2 的 1/4

- **ID**：`023b5b3d`　**分歧度**：8
- **摘要**：AI 大模型的竞争,正在从谁更会回答问题,转向谁能真正完成一项复杂工作。 刚刚,Google CEO Pichai 在 x 上宣布,旗下旗舰模型 Gemini 4 Carbon 正式向开发者、企业、科研人员和消费者扩大开放。 距离这款模型首次亮相,才刚不到两周。 10 月 28 日,Google 首次公布 Carbon 时,重点展示了其在网络安全、软件工程和…

| | 主体 | 技术 | 类别 |
|---|---|---|---|
| **LLM 金标** | Google, Gemini 4 Carbon, Pichai, OpenCode, Fable 5.2 | 大模型, Agent, 代码生成, 长上下文推理, 工作流自动化 | 模型发布, 模型开放, 基准评测, 产品定价 |
| **规则产出** | Google, Perplexity | llm, safety, reasoning, coding, agent | safety, business, technology, application |

- **判定**：　　（填 `对` / `错` / `部分`）
- **备注**：

### 28. 前端工程师转 AI 全栈开发:从学习路线到 Agent 工程落地的完整指南

- **ID**：`0c0a3174`　**分歧度**：6
- **摘要**：一、为什么前端工程师适合转 AI 全栈? 前端工程师转型 AI 全栈开发,并非从零开始,而是已有能力的迁移和升维。核心优势体现在四个方面: 1. 产品和交互经验直接复用 界面设计、交互逻辑、状态管理、用户反馈——这些前端工程师最擅长的能力,在 AI Agent 产品中同样至关重要。无论是 Chatbot 的对话界面,还是 Agent 的任务编排可视化,都需要…

| | 主体 | 技术 | 类别 |
|---|---|---|---|
| **LLM 金标** | （空） | Agent, Chatbot, Docker, CI/CD, Kubernetes | 职业转型, AI 全栈开发, 学习路线, Agent 工程 |
| **规则产出** | （空） | agent, data, inference | product, technology, application, industry |

- **判定**：　　（填 `对` / `错` / `部分`）
- **备注**：

### 29. 谷歌首批 Gemini 智能眼镜震撼开启预售:395 美元起售,开创性引入“双向隐私”条款

- **ID**：`0089977b`　**分歧度**：5
- **摘要**：人工智能硬件市场迎来全新重磅产品。当地时间11月12日,据媒体报道,谷歌正式开启了 Intelligent Eyewear 系列智能眼镜的预购。作为谷歌在具身智能与穿戴设备领域的 最新 力作,该产品由谷歌联合三星、Gentle Monster 以及 Warby Parker 共同打造,起售价为395美元。 豪华大厂联手打造,定价395美元起 在产品生态与设计…

| | 主体 | 技术 | 类别 |
|---|---|---|---|
| **LLM 金标** | 谷歌, Gemini, Intelligent Eyewear, 三星, Warby Parker | 生成式 AI, 具身智能, 端侧 AI, 多模态交互, 可穿戴设备 | 产品发布, 硬件预售, 隐私安全, 服务条款 |
| **规则产出** | Google, 三星 | robot, edge | policy, safety, business, hardware, product, industry |

- **判定**：　　（填 `对` / `错` / `部分`）
- **备注**：

## 其他分歧（1 条）

### 30. NVIDIA/auto-ontology

- **ID**：`11826844`　**分歧度**：7
- **摘要**：Auto Ontology是基于数据构建语义图推理层的Python框架,支持确定性查询生成、优质上下文检索及可解释AI。

| | 主体 | 技术 | 类别 |
|---|---|---|---|
| **LLM 金标** | NVIDIA, Auto Ontology | Python, 语义图推理, 知识图谱, Agent, 文本转SQL | 开源发布, 企业数据分析 |
| **规则产出** | NVIDIA | data, inference | technology |

- **判定**：　　（填 `对` / `错` / `部分`）
- **备注**：

