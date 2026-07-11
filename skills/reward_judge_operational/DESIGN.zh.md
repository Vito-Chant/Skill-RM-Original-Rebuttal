# reward_judge_operational 设计说明

`reward_judge_operational` 是 Skill-RM 的 resource-rich skill artifact。它不是把样本内容塞进 system prompt 的长 prompt，而是一个 agent-skill 资源接口：模型可以自行选择加载 skill，查看当前样本可见的资源索引，调用少量工具或资源，然后通过统一的 final-answer contract 给出 reward judgment。

当前 release 只面向三个 benchmark：

```text
RewardBench2
RM-Bench
JudgeBench
```

PPE-ZH 的历史实验结果不进入当前 release 配置和主线复现实验。

## 文件结构

```text
skills/reward_judge_operational/
├── SKILL.md
├── resources.yaml
├── references/
├── rubrics/
├── scripts/
└── verifiers/
```

所有 benchmark 都使用这同一个 skill 目录，不按 benchmark 切换 skill artifact。

## 调用链

1. runner 加载样本，构造当前可见的 user prompt 和 candidate responses。
2. baseline 直接让 base model judge；skill settings 则额外开放 `use_skill`、`view_resource`、`python_sandbox`、`run_resource` 和 `final_answer` 等 tool。
3. 模型如果认为 skill 有帮助，可以调用 `use_skill`。
4. `use_skill` 返回 `SKILL.md`、当前样本可见资源索引、推荐资源和工具使用约束。
5. 模型可以读取 rubric/principle/reference/checklist，或运行可用 verifier / same-base-model external pipeline。
6. 模型必须通过 `final_answer` 或可解析 JSON 输出最终 verdict。

## 资源边界

Operational setting 可以使用：

- benchmark/task metadata；
- sample-visible reference、ground truth、answer key 或 checklist；
- benchmark-specific rubric / principle；
- deterministic verifier protocol；
- `python_sandbox`；
- same-base-model external prompt resources，例如 RewardBench2 listwise 或 OpenRS-style pairwise。

Operational setting 不允许使用：

- hidden chosen/rejected origin；
- hidden gold label；
- test label；
- previous baseline prediction fallback；
- per-sample oracle routing；
- post-hoc metric hacking。

## python_sandbox

`python_sandbox` 只注入：

```python
prompt
candidates
sample = {"prompt": prompt, "candidates": candidates}
```

它不能访问本地文件、网络、hidden labels、原始 dataset record 或 chosen/rejected 来源。典型用途是格式检查、答案抽取、简单数学计算、计数字符/单词/列表项、检查 JSON 或代码片段的可见行为。

## Benchmark 资源

### RewardBench2

主要资源：

- `rubric.rewardbench2`
- `principle.rewardbench2`
- visible reference / ground truth / checklist when present
- `verifier.reference_match`
- `tool.python_sandbox`
- `external.rewardbench2_official_listwise_qwen`

任务形态是 official-compatible best-of-four listwise ranking，并保留 Ties rating path。

### RM-Bench

主要资源：

- `rubric.rmbench`
- `principle.rmbench`
- visible reference when present
- `verifier.reference_match`
- `tool.python_sandbox`
- `external.openrs_pairwise_qwen`

任务形态是 response-matrix pairwise judging。主指标用 full-set win accuracy：`win / total`。

### JudgeBench

主要资源：

- `rubric.judgebench`
- `principle.judgebench`
- visible answer key / reference when exposed
- `verifier.reference_match`
- `tool.python_sandbox`
- `external.openrs_pairwise_qwen`

任务形态是 reverse-order pairwise judging。汇总脚本按官方式 reverse-order 聚合：两个 order 映射回 chosen/rejected 后，correct vote 计 `+1`，wrong vote 计 `-1`，tie/abstain 计 `0`；总分 `>0` 为 correct，`<0` 为 wrong，`=0` 为 same。

## 和 skill_fair 的区别

`skill_fair` 只允许 prompt + candidate responses + generic rubric/principle + visible-text-only `python_sandbox`。它不暴露 benchmark metadata、reference、ground truth、checklist 或 external pipeline。

`skill_operational` 允许 resource-rich pipeline，因此应在论文中明确标注为 operational / resource-rich setting，而不是信息公平 RM 对比。
