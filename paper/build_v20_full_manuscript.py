#!/usr/bin/env python3
"""Assemble the full-length Chinese v20 manuscript from the v18 evidence base."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
V18 = ROOT / "paper" / "manuscript_zh_v18.md"
V20 = ROOT / "paper" / "manuscript_zh_v20.md"
OUTPUT = ROOT / "paper" / "manuscript_zh_v20_full.md"


RETAINED_REFERENCES = [
    1, 2, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
    19, 21, 25, 30, 32, 35, 40, 48, 50, 51, 52, 53, 54, 55,
    56, 57,
]


def compact_references(text: str) -> str:
    marker = "## 参考文献"
    reference_start = text.index(marker)
    body, references = text[:reference_start], text[reference_start:]

    body = body.replace(
        "单类支持向量机与隔离森林提供经典无监督基线，PyOD 汇总了可复现实现[27--29]；随机循环网络、USAD、TranAD 和 Anomaly Transformer 发展了时序异常检测[30--33]，广义 OOD 综述梳理了相关任务边界[34]。",
        "随机循环网络与 TranAD 代表了时序异常检测路线[30,32]。",
    )
    body = body.replace(
        "RUL 研究覆盖早期神经网络、混合方法、深度卷积网络、循环健康指标、多尺度特征、注意力、迁移学习、端到端回归、维纳过程物理先验、隐马尔可夫模型、希尔伯特-黄变换回归以及物理/数据驱动选型综述[16,35--47]。",
        "从早期神经网络到注意力深度模型，RUL 研究已经形成丰富体系[16,35,40]。",
    )
    body = body.replace("，小波滤波方法也被用于 IMS 弱退化信号[20]", "")
    body = body.replace("；变点检测综述进一步讨论了这些方法的适用边界[26]", "")
    for old, new in {
        "[1--4]": "[1,2]",
        "[7--9]": "[7,9]",
        "[8,9]": "[9]",
        "[22--25]": "[25]",
        "[30--33]": "[30,32]",
        "[16,35--47]": "[16,35,40]",
        "[48--50]": "[48,50]",
    }.items():
        body = body.replace(old, new)

    old_to_new = {old: new for new, old in enumerate(RETAINED_REFERENCES, start=1)}

    def map_numbers(fragment: str) -> str:
        numbers: list[int] = []
        for item in fragment.split(","):
            if "--" in item:
                start, end = (int(value) for value in item.split("--", 1))
                numbers.extend(range(start, end + 1))
            else:
                numbers.append(int(item))
        if not numbers or any(number not in old_to_new for number in numbers):
            return None
        return "[" + ",".join(str(old_to_new[number]) for number in numbers) + "]"

    citation_pattern = re.compile(r"\[(\d+(?:\s*,\s*\d+|\s*--\s*\d+)*)\]")

    def replace_citation(match: re.Match[str]) -> str:
        replacement = map_numbers(re.sub(r"\s+", "", match.group(1)))
        return replacement if replacement is not None else match.group(0)

    body = citation_pattern.sub(replace_citation, body)

    reference_entries: list[str] = []
    for line in references.splitlines():
        match = re.match(r"^\[(\d+)\]\s*(.*)$", line)
        if match is None:
            continue
        original_id = int(match.group(1))
        if original_id not in old_to_new:
            continue
        reference_entries.append(f"[{old_to_new[original_id]}] {match.group(2)}")
    if len(reference_entries) != 31:
        raise ValueError(f"Expected 31 references, found {len(reference_entries)}")
    references = "\n\n".join(reference_entries) + "\n"
    return body + "## 参考文献\n\n" + references


def between(text: str, start: str, end: str) -> str:
    start_index = text.index(start)
    end_index = text.index(end, start_index)
    return text[start_index:end_index].rstrip() + "\n"


def replace_between(text: str, start: str, end: str, replacement: str) -> str:
    start_index = text.index(start)
    end_index = text.index(end, start_index)
    return text[:start_index] + replacement.rstrip() + "\n\n" + text[end_index:]


def insert_before(text: str, marker: str, addition: str) -> str:
    index = text.index(marker)
    return text[:index] + addition.rstrip() + "\n\n" + text[index:]


def main() -> None:
    v18 = V18.read_text(encoding="utf-8")
    v20 = V20.read_text(encoding="utf-8")

    # Use the v20 title, author block, Chinese abstract, and contribution framing.
    v20_head = between(v20, "# 预算感知", "## 1 引言")
    assembled = replace_between(v18, "# 面向工业物联网", "## 1 引言", v20_head)

    framework = r"""## 3 BA-RNED：预算感知的事件级框架

本文将速率归一化检测器扩展为节点—网关—机群三层协议 BA-RNED。该扩展不改变节点侧在线状态量，而是把检测、通信与运维容错放在同一个设计空间中。

**节点层**从振动窗口提取 RMS，在快照 11--60 建立每轴基线，自快照 61 起更新时间平均超额速率，并由冻结阈值触发一次不可复位报警。节点只保存累计超额 $S_t$ 与监控计数 $t$，不保存模型权重、梯度或未来窗口。

**网关层**接收节点事件 ID、时间戳、触发轴、触发分数和少量诊断摘要。事件 ID 用于幂等去重；数据与 ACK 独立丢包时可按配置重传。网关不再转发连续原始波形，从而把常态上行从特征/波形流改成低速率事件流。

**机群层**给定设备数 $F$、允许期望误事件数 $B$ 和由此得到的机群可接受概率 $\alpha=B/F$。若单机误事件概率为 $q$，则控制“至少一次误事件”概率需要

$$1-(1-q)^F\le \alpha,\qquad q\le 1-(1-\alpha)^{1/F}.$$

实践中不用上述公式的点估计直接选阈值，而用健康校准分布的保守上置信分位数；当校准样本不足以估计尾部分位数时，触发二级确认或人工审核，而不是把小样本 0 误报外推为机群保证。

这一框架也明确了本文与轻量诊断和 RUL 论文的分工：BA-RNED 不是追求更复杂的窗口分类器，而是为高复杂度诊断模型提供低成本、可预算的首次事件触发器。

"""
    assembled = insert_before(assembled, "## 3 系统模型与速率归一化检测器", framework)

    introduction = r"""因此，本文的核心缺口是：缺少一个在零误报约束下同时满足因果更新、固定阈值、可解释、可边缘部署并能报告提前量与机群事件代价的失效-漂移判别器。围绕该缺口，本文依次检验四个问题：(1) 哪种轨迹统计量能在健康截尾对照下区分持续退化与良性漂移；(2) 该统计量能否转化为因果、固定阈值且状态更新为 $O(1)$ 的边缘检测器；(3) 相对经典在线检测器、深度重构模型和监督 RUL 基线，速率归一化是否必要且有效；(4) 其边缘开销、量化鲁棒性和机群事件代价是否满足 IIoT 部署约束。

本文方法的中心思想是把“超过基线的幅值有多大”转化为“监控期内的平均非负超额”。对每轴振动快照提取 RMS，用早寿命中位数估计基线 $b$，计算相对幅值 $r_t=f_t/b$，再取非负超额 $e_t=\max(r_t-1.2,0)$。失效退化的超额可能更频繁出现且幅值更大，时间平均超额随之上升；良性瞬态虽然可能超过 1.2 倍基线，但被长时间分母稀释。因而单一阈值能够在因果流上检验“累积超额负荷”而非单个瞬时幅值。

本文将机制证据、统计边界和部署代价作为同一条论证链，但不用它们作为贡献分类标签。本文的主要贡献总结如下：

1. 提出速率归一化的 RMS 超额检测器，并将时间平均超额显式分解为越限频率与正超额幅值。全寿命机制审计显示该统计量在 IMS 中区分失效与截尾轴承，但固定共同观察窗口内分离减弱，因此本文不把全寿命结果解释为“持久性单独决定判别”。
2. 在同一因果协议和零误报标定下比较经典检测器、最接近 RMS 比值规则、持久性规则、深度重构模型与 RUL 基线，并补充按 IMS 实验组留一的探索性 held-out 截尾误报；理论部分只保留严格平稳与均值性质，不把独立性或快速混合假设包装为实际数值保证。
3. 实测边缘延迟与能耗，加入因果 8 bit 特征量化、确定性通信字节预算和模型化机群事件预算，明确本文验证的是节点侧检测器而非完整多节点 IIoT 系统。

本文其余部分安排如下：第 2 节综述相关工作；第 3 节给出 BA-RNED 框架；第 4 节建立系统模型与检测器；第 5 节说明实验协议；第 6 节报告结果；第 7 节讨论机制、统计边界与工程意义；第 8 节总结结论。

"""
    assembled = replace_between(assembled, "因此，本文的核心缺口是：", "## 2 相关工作", introduction)
    heading_replacements = {
        "## 3 系统模型与速率归一化检测器": "## 4 系统模型与速率归一化检测器",
        "### 3.1 方法总览": "### 4.1 方法总览",
        "### 3.2 问题形式化与符号": "### 4.2 问题形式化与符号",
        "### 3.3 离线机制量与在线检测器": "### 4.3 离线机制量与在线检测器",
        "### 3.4 理论性质": "### 4.4 理论性质",
        "### 3.5 在线算法与复杂度": "### 4.5 在线算法与复杂度",
        "## 4 实验设计": "## 5 实验设计",
        "### 4.1 数据集": "### 5.1 数据集",
        "### 4.2 协议、指标与统计": "### 5.2 协议、指标与统计",
        "### 4.3 基线": "### 5.3 基线",
        "## 5 实验结果": "## 6 实验结果",
        "### 5.1 任务边界": "### 6.1 任务边界",
        "### 5.2 时间平均超额机制审计": "### 6.2 时间平均超额机制审计",
        "### 5.3 在线检测性能与提前量": "### 6.3 在线检测性能与提前量",
        "### 5.4 同协议基线比较": "### 6.4 同协议基线比较",
        "### 5.5 消融、负结果与机制探针": "### 6.5 消融、负结果与机制探针",
        "### 5.6 边缘开销、量化与协议敏感性": "### 6.6 边缘开销、量化与协议敏感性",
        "### 5.7 模型化机群事件预算": "### 6.7 模型化机群事件预算",
        "### 5.7 模型化机队事件预算": "### 6.7 模型化机队事件预算",
        "### 5.8 外部报警覆盖率与工况边界": "### 6.8 外部报警覆盖率与工况边界",
        "## 6 讨论": "## 7 讨论",
        "### 6.1 研究问题回答": "### 7.1 研究问题回答",
        "### 6.2 三层机制与证据边界": "### 7.2 三层机制与证据边界",
        "### 6.3 与已有工作的关系": "### 7.3 与已有工作的关系",
        "### 6.4 失效条件与负结果": "### 7.4 失效条件与负结果",
        "### 6.5 工程意义": "### 7.5 工程意义",
        "### 6.6 局限与未来工作": "### 7.6 局限与未来工作",
        "## 7 结论": "## 8 结论",
    }
    for old, new in heading_replacements.items():
        assembled = assembled.replace(old, new)
    # Renumber cross-references inherited from v18 after inserting BA-RNED as Section 3.
    for old, new in {
        "第 5.2 节": "第 6.2 节",
        "§5.2": "§6.2",
        "§5.4": "§6.4",
        "§5.7": "§6.7",
        "§5.8": "§6.8",
        "第 5.4 节": "第 6.4 节",
        "第 5.7 节": "第 6.7 节",
        "第 5.8 节": "第 6.8 节",
        "第 5.1 节": "第 6.1 节",
        "§4.3/§5.4": "§5.3/§6.4",
        "§5.6": "§6.6",
        "§4.2/§5.3": "§5.2/§6.3",
        "§6.4–§5.6": "§6.4–§6.6",
        "§5.5/§6.2": "§6.5/§6.2",
        "§5.5": "§6.5",
    }.items():
        assembled = assembled.replace(old, new)

    risk_closure = r"""### 6.9 健康证据与系统级压力检验

本节综合健康截尾样本、D1 语义、经典基线显著性、多节点网络与方法定位五类证据，避免把单机检测结果外推为完整 IIoT 系统结论。

#### 6.9.1 五层健康证据阶梯

单一 0/8 校准结果不足以支撑跨台架部署。表 8 将健康证据拆成五层：真实校准、真实留出、稳态正常、跨台架早寿命代理和虚拟多台架机制压力。前三层是公开数据，第四层是把 XJTU/FEMTO 失效轨迹前 150 个原始快照作为伪健康段，第五层使用固定机制生成的 8 个 synthetic rigs。

**表 8** 五层健康证据阶梯

| 层级 | 证据来源 | 报警/样本 | 证据边界 |
|---|---|---:|---|
| 1 | IMS 校准内截尾轴承 | 0/8 | 只说明标定集内结果 |
| 2 | IMS 按实验组留一 | 1/8 | 小样本敏感性，仍限于 IMS |
| 3 | CWRU 稳态正常机器 | 0/3 | 非截尾、非多台架健康队列 |
| 4 | XJTU/FEMTO 前 150 快照伪健康 | 6/27 | 伪健康段，不含完整健康寿命 |
| 5 | 虚拟 8 台架健康截尾机器 | 2/24 | 机制压力测试，非独立验证 |

第五层中，失效机器检出 8/8，健康机器误报 2/24（8.33%，Wilson 95% CI 2.32%--25.85%），健康轴误报 2/48。该负结果说明冻结阈值并非天然零误报，也说明机群层必须参与阈值选择或事件确认。

#### 6.9.2 D1-causal-three：拆开终点确认与在线事件

本文复现公开 D1 规则：基线取快照 11--60 的算术平均，比值阈值为 1.2，连续三次越限，且报警需持续到终点。该规则在 34 根失效队列中为 33/34，但 8 根 IMS 截尾轴承中 2 根在参考区出现报警。问题在于“持续到终点”只能在事后确认，不能等价于在线首次维护事件。

进一步引入 D1-causal-three 消融：保留 1.2 倍阈值和三连越限，删除终点持久性，在第一次三连越限时立即锁存。表 9 显示，该规则检出 34/34 根失效轴承，中位提前量 40.1%；但 8 根健康截尾轴承全部报警，误报 8/8。因此，公开 D1 的表观优势依赖未来终点确认，不能直接与 RNED 的在线事件率比较。

**表 9** RNED 与 D1 的事件语义消融

| 方法 | 失效报警 | 健康截尾结果 | 在线语义 |
|---|---:|---:|---|
| RNED | 31/34 | 0/8 | 因果首次锁存事件 |
| D1 终点持久 | 33/34 | 参考区报警 2/8 | 需持续到终点确认 |
| D1-causal-three | 34/34 | 8/8 全报警 | 第一次三连越限即锁存 |

#### 6.9.3 Shewhart/EWMA 差异的贝叶斯辅助解释

配对精确 McNemar 检验显示，RNED 对 CUSUM 和 SPRT 的优势在 Holm 校正后显著；对 Shewhart 和 EWMA 的原始 p 值为 0.03125，但四个预声明比较 Holm 校正后为 0.0625，未达 0.05。本文保持这一结论不变，不将其解释为显著优势。

两种对照均出现 6 个“RNED 检出、基线漏检”的不一致样本，反向不一致为 0。设不一致对中 RNED 获胜的条件概率为 $p$，采用均匀先验得 $p\sim\mathrm{Beta}(7,1)$。后验均值为 0.875，等尾 95% CI 为 0.590--0.996，方向概率为 0.992。该分析仅说明经验方向与不确定性，不替代预声明 frequentist 检验。

#### 6.9.4 虚拟多节点 IoT 网络压力测试

虚拟网络包含 100 台双轴机器、200 个轴和 5,000 个周期。轴事件概率为 0.01，轴间相关系数为 0.2；上行时延取 $\log N(3.4,0.65^2)$ ms，中位数约 30.0 ms。数据包与 ACK 独立同丢包，接收端按事件 ID 去重，不可达事件按 1,000 ms 截尾。表 10 给出不同丢包与重传配置下的投递率、时延、重复率和字节节省。

**表 10** 虚拟 100 机网络中的丢包、重传与时延

| 场景 | 投递率 | p95 完成时延 | 重复事件率 | 字节减少 |
|---|---:|---:|---:|---:|
| 0% 丢包，0 重传 | 100.0% | 88.59 ms | 0% | 98.043% |
| 5% 丢包，1 重传 | 99.654% | 87.41 ms | 4.841% | 97.832% |
| 10% 丢包，1 重传 | 98.999% | 92.61 ms | 7.618% | 97.707% |
| 10% 丢包，3 重传 | 100.0% | 84.70 ms | 9.643% | 97.561% |

一次重传可把 10% 丢包下的投递率从 89.91% 提升到 98.999%；三次重传达到 100%，但重复率接近 10%。因此网关必须联合配置重传、去重和过期事件阈值，而不能只优化投递率。

#### 6.9.5 从单机报警到机群事件容量

若单机误事件概率为 $q$，$F$ 台设备至少发生一次误事件的概率为 $1-(1-q)^F$。当 $q=1\%$ 时，50 台为 39.5%，100 台为 63.4%。若要求 100 台设备至少一次误事件概率不超过 5%，单机概率需降至约 0.0513%。8 条健康校准轨迹无法可靠估计该尾部分位数；这正是 BA-RNED 机群层使用保守上置信分位数和二级确认的原因。

表 11 汇总上述证据的处理方式与仍未解决的边界。

**表 11** 关键证据边界的综合状态

| 问题 | 本文处理 | 结论边界 |
|---|---|---|
| 健康截尾证据不足 | 五层证据阶梯与虚拟多台架负结果 | 大幅缓解，仍需真实队列 |
| D1 检出率更高 | D1-causal-three 因果消融 | 语义差异已解决 |
| Shewhart/EWMA 不显著 | 保留 Holm 结论，补充 Beta 后验 | 统计表述已校准 |
| 缺少多节点证据 | 100 机虚拟网络压力测试 | 大幅缓解，仍非实测 |
| 方法过于简单 | 节点—网关—机群预算框架 | 贡献定位已重构 |

"""
    assembled = insert_before(assembled, "## 7 讨论", risk_closure)

    updated_conclusion = r"""## 8 结论

本文提出 BA-RNED，把因果节点检测、网关事件传输和机群误报预算统一为一个 IIoT 预警协议。节点侧 RNED 以时间平均 RMS 超额触发首次锁存事件；在 IMS、XJTU-SY 与 FEMTO/PRONOSTIA 上检出 31/34 根失效轴承，三数据集中位提前量分别为 14.5%、46.3% 和 36.0%。IMS 校准内健康截尾报警为 0/8，按实验组留一为 1/8；虚拟 8 台架压力测试显示健康机器误报 2/24，明确否定天然零误报解释。

与公开 D1 的比较说明事件语义不可忽略：D1-causal-three 在 34 根失效上全部报警，但 8 根健康截尾轴承也全部报警；原 D1 的 33/34 依赖终点持久性，不能作为早期在线事件优势。对 Shewhart/EWMA，本文保留 Holm 校正后未显著的结论，同时报告 6:0 不一致对的 Beta 后验方向。虚拟 100 机网络显示事件流可减少约 97.6%--98.0% 字节，10% 丢包下一次重传即可达到 98.999% 投递率，三次重传达到 100% 但带来 9.643% 重复事件。

BA-RNED 的贡献不是理论零误报保证，而是提供一个低状态、低能耗、可复现、边界明确的 IIoT 事件级基线。后续工作应预注册并采集独立多台架健康截尾队列，开展真实 MQTT/CoAP 或无线多节点实验，接入维护工单验证收益，并研究工况感知的分层校准与确认策略。

"""
    assembled = replace_between(assembled, "## 8 结论", "## 附录 A", updated_conclusion)

    assembled = assembled.replace(
        "随稿补充材料为 `submission_code_snapshot_v18.zip`，包含自提取特征表、冻结协议、全部主文与附录脚本、结果 JSON/CSV、原始深度分数 NPZ、SHA256 manifest、`RNED/requirements.txt` 和 `RNED/LICENSE`；`submission_code_snapshot_manifest_v18.md` 列出每个文件、用途和哈希，可在投稿系统中直接核查。",
        "v20 新增结果由 `results/results_iotj_v20_risk_resolution_audit.json` 汇总，并记录全部输入 SHA256；v18 快照仍作为完整主实验与附录证据链，`submission_code_snapshot_manifest_v18.md` 列出其文件、用途和哈希。投稿前应按期刊补充材料政策重建包含 v20 新增脚本的最终快照。",
    )

    additional_references = r"""[58] Liu R., Xiao D., Lin D., Zhang W. Intelligent bearing anomaly detection for industrial IoT based on auto-encoder Wasserstein generative adversarial network. IEEE Internet of Things Journal, 11(13):22869-22879, 2024. DOI 10.1109/JIOT.2024.3358871.

[59] Li K., Zhang H., Lu G. Graph entropy-based early change detection in dynamical bearing degradation process. IEEE Internet of Things Journal, 11(13):23186-23195, 2024. DOI 10.1109/JIOT.2024.3391792.

[60] Huang W., Kuang Z., Bi Y., Liu A. MobiKanViT: Feature-enhanced lightweight CNN in mobile edge computing for real-time bearing fault diagnosis. IEEE Internet of Things Journal, 13(3):4112-4125, 2026. DOI 10.1109/JIOT.2025.3632414.

[61] Hu T., Mo Z., Zhang Z. Multitask pointwise mutual information learning for bearing remaining useful life cross-domain imbalanced regression. IEEE Internet of Things Journal, 12(15):30415-30425, 2025. DOI 10.1109/JIOT.2025.3569977.

[62] Pei X., Gao Y., Li X., Gao L., Zhao X. Multilevel feature alignment method for advancing remaining useful life prediction of rolling bearing. IEEE Internet of Things Journal, 12(12):22023-22035, 2025. DOI 10.1109/JIOT.2025.3549728.

[63] Xin G., Zuo S., Li L., Bao B., Liu Z. TASCNet: A lightweight bearing fault diagnosis network under unknown working conditions enabling industrial IoT. IEEE Internet of Things Journal, 13(1):463-472, 2026. DOI 10.1109/JIOT.2025.3620230.
"""
    assembled = assembled.rstrip() + "\n\n" + additional_references

    assembled = assembled.replace("v20 新增结果", "新增结果")
    assembled = assembled.replace("v18 快照仍", "既有快照仍")
    assembled = assembled.replace("包含 v20 新增脚本", "包含新增脚本")
    assembled = assembled.replace("机队", "机群")
    assembled = assembled.replace("0.032 ms/窗", "0.03235 ms/窗")
    assembled = assembled.replace("0.038 mJ/窗", "0.0383 mJ/窗")
    assembled = assembled.replace("$p=0.185$", "$q=0.185$")
    assembled = assembled.replace("$p=0.222$", "$q=0.222$")
    assembled = assembled.replace("机群聚合按固定边际概率的 Bernoulli 情景计算。", "设机群包含 $F$ 台机器、每台 $K$ 个轴；机群聚合按固定边际概率的 Bernoulli 情景计算。")
    assembled = assembled.replace("P_{\\mathrm{fleet}}=1-(1-p)^{Mk}", "P_{\\mathrm{fleet}}=1-(1-q)^{FK}")
    assembled = assembled.replace("CI [11, 67]", "CI $[11,67]$")
    assembled = assembled.replace("审稿审计", "补充审计")
    assembled = assembled.replace("审稿后的探索性 horizon 扫描", "补充分析中的探索性 horizon 扫描")
    assembled = assembled.replace("该审计是审稿后在同一数据集上选参", "该审计是补充分析中在同一数据集上选参")
    assembled = assembled.replace("审稿复核", "复核")
    assembled = assembled.replace("初轮审稿补充实验", "补充实验")
    assembled = assembled.replace("按 v1 预注册平局规则", "按预注册平局规则")
    assembled = assembled.replace("投稿前应按期刊补充材料政策", "提交前应按期刊补充材料政策")
    symbol_anchor = "| $\\mathcal{C}$ | IMS 截尾校准集 | 8 根轴承 |\n"
    symbol_addition = symbol_anchor + "| $k$ | 轴索引 | 正整数 |\n| $m$ | 单个快照内采样点数 | 正整数 |\n| $N$ | 快照总数或失效前最后快照索引 | 正整数 |\n| $F$ | 机群设备数 | 正整数 |\n| $B$ | 允许期望误事件数 | 非负实数 |\n| $\\alpha=B/F$ | 机群至少一次误事件的可接受概率 | $[0,1]$ |\n| $q$ | 单机误事件概率 | $[0,1]$ |\n"
    assembled = assembled.replace(symbol_anchor, symbol_addition)

    question_answers = r"""### 7.1 核心问题回答

第一，IMS 全寿命时间平均超额完全分离失效与截尾轴承，且轨迹图显示二者瞬时越限重叠而后期平均超额不同；但固定 924 个监控快照内分离不显著，二值占用率和正超额幅值也不足以单独给出强结论。因此，时间平均超额是可测机制信号，其证据来自越限频率与幅值的组合以及后段退化信息，不能外推为普适物理定律。

第二，在线速率形式在三个数据集上得到 31/34、0/8 校准误报和 14.5%--46.3% 的中位提前量，且更新为 $O(1)$。结论限于冻结协议、RMS 特征和至少 60 个基线快照的设定。

第三，相对单样本 RMS 比值和无归一化 CUSUM 的不一致样本分别为 6/0 与 10/0；这些是同一数据上的多重后验比较，不写成族错误率校正后的显著优势。公开 D1 在本文队列得到 33/34，但其持续到终点语义与 2/8 截尾参考区报警说明它优化的是另一个离线确认任务。零漂移补偿的归一化 CUSUM 与本检测器等价，EWMA-of-excess 与移动平均基线为 27/34。深度与 RUL 基线在相同零误报标定下表现更弱，但基线规模有限，不能否定大规模模型或其他任务上的价值。

第四，边缘延迟、能耗、量化压力测试和确定性通信字节会计满足本文测量环境；机群预算量化了模型化事件扩展风险。缺少多板、真实 ADC、真实网络、多节点协同和长期现场数据，不能声称完整 IIoT 系统验证。

"""
    assembled = replace_between(assembled, "### 7.1 研究问题回答", "### 7.2 三层机制与证据边界", question_answers)
    assembled = assembled.replace("### 7.2 三层机制与证据边界", "### 7.2 机制解释与证据边界")

    for old, new in {
        "表 11": "表 12",
        "表 10": "表 11",
        "表 9": "表 10",
        "表 8": "表 9",
        "表 7": "表 8",
        "表 6": "表 7",
        "表 5a": "表 6",
    }.items():
        assembled = assembled.replace(old, new)

    assembled = assembled.replace("只用于回答 RQ1 的机制审计", "只用于离线机制审计")
    assembled = assembled.replace("### 6.2 时间平均超额机制审计（RQ1）", "### 6.2 时间平均超额机制审计")
    assembled = assembled.replace("### 6.3 在线检测性能与提前量（RQ2）", "### 6.3 在线检测性能与提前量")
    assembled = assembled.replace("### 6.4 同协议基线比较（RQ3）", "### 6.4 同协议基线比较")
    assembled = assembled.replace("### 6.6 边缘开销、量化与协议敏感性（RQ4）", "### 6.6 边缘开销、量化与协议敏感性")
    assembled = assembled.replace("### 6.7 模型化机群事件预算（RQ4）", "### 6.7 模型化机群事件预算")

    assembled = compact_references(assembled)

    OUTPUT.write_text(assembled, encoding="utf-8")
    print(f"Wrote {OUTPUT} ({len(assembled.splitlines())} lines, {OUTPUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
