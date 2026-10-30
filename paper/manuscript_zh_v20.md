# 预算感知的事件级速率归一化早期预警：面向工业物联网轴承的虚拟环境验证

**Budget-Aware Event-Level Rate-Normalized Early Warning for Bearings in Industrial IoT: Validation in a Virtual IoT Environment**

**作者**：Zhiguo Xiao¹,²,\*，Yujing Guo²，Yanqi Shao²，Guangteng Chen²，Dingbo Hou²，Yuanyuan Liu²，Xinyao Cao¹,²，Junli Liu¹,²

**单位**：¹北京理工大学计算机科学与技术学院，北京 100081；²长春大学计算机科学与技术学院，长春 130022

**通讯作者**：Zhiguo Xiao，3220215169@bit.edu.cn

## 摘要

工业物联网中的轴承预测性维护需要同时约束边缘检测、事件上报和机群误报预算。本文提出预算感知的事件级速率归一化检测框架 BA-RNED：节点层将超过早寿命 RMS 基线 1.2 倍的非负超额转化为时间平均超额速率，并用冻结阈值触发首次锁存报警；网关层传输、去重和重传小事件包；机群层把误事件预算转化为校准分位数需求。在 IMS、XJTU-SY 与 FEMTO/PRONOSTIA 的 34 根失效和 8 根健康截尾轴承上，RNED 检出 31/34（精确 95% CI：76.3%–98.1%），三数据集中位提前量为 14.5%、46.3% 和 36.0% 寿命；五层健康证据显示其并非天然零误报。D1-causal-three 消融表明公开 D1 的 33/34 优势依赖终点确认：去掉该条件后虽得到 34/34 检出，但健康截尾报警为 8/8。Jetson Orin NX 上更新耗时 0.03235 ms/窗、能耗 0.0383 mJ/窗；100 台设备仿真中事件流较周期状态流减少 98.043% 字节，10% 丢包与一次重传下投递率为 98.999%。BA-RNED 的贡献是把因果事件、网络代价和机群预算纳入同一可复现协议，而不是提出更复杂的诊断模型。

## 1 引言

IIoT 轴承维护系统的输出不应只是“第 *t* 个窗口是否异常”，而应是可锁存、可传输、可预算的首次维护事件。传感节点长期采集振动波形，但持续上传原始数据会消耗能量和带宽；云侧若只接收连续异常分数，仍需要另行决定报警阈值、确认次数和机群容错配额。若把所有偏离都视为故障，良性漂移会造成虚假停机；若等待状态持续到数据终点，报警又失去在线维护价值。

现有 IoTJ 研究在轻量诊断、未知工况泛化、跨域 RUL 和边缘模型加速方面进展显著[14--18]。这些工作关心模型容量、特征对齐或寿命回归精度，通常以窗口分类、异常分数或 RUL 误差为核心。相比之下，可部署维护系统还必须回答三个问题：第一，节点何时第一次锁存事件；第二，事件在网络中丢包、重复和时延下的代价；第三，给定格点规模和误事件预算时需要多少健康校准样本。

本文把这三个问题合并为 **BA-RNED** 框架。节点层沿用速率归一化检测器 RNED；网关层把报警变成 64 B 事件包并处理重传与去重；机群层用显式预算公式将运维容错转化为阈值分位数和样本容量需求。该方法刻意保持低复杂度，使新颖性位于事件协议和预算机制，而不是模型堆叠。

### 1.1 本文贡献

1. **预算感知事件框架。** 提出节点检测、网关事件传输和机群预算三层联合设计，并给出从事件预算到校准分位数需求的转换关系。
2. **因果 D1 消融。** 证明公开 D1 的表观优势来自终点确认：去掉该条件后 D1-causal-three 虽 34/34 检出，但健康截尾 8/8 报警。
3. **五层健康证据阶梯。** 结合 IMS 校准、留一、稳态正常、跨台架早寿命代理和虚拟多台架截尾，系统刻画误报不确定性。
4. **统计解释重构。** 对 Shewhart/EWMA 保持 Holm 未显著结论，同时报告 6:0 不一致对的 Beta 后验，避免过度声称或信息丢失。
5. **虚拟 IoT 网络压力测试。** 在 100 机、200 轴、5,000 周期下量化丢包、重传、重复包、时延和字节数。

## 2 相关工作

振动诊断方法利用 RMS、包络谱、谱峭度和学习特征刻画损伤[1--3]。2024–2026 年 IoTJ 工作进一步推进轻量 CNN、可解释二次神经网络、图熵早期变化检测、跨域 RUL 和多级特征对齐[14--18]。这些方法追求更强的诊断或寿命预测能力，通常不把首次在线锁存事件、事件包网络行为和机群误报预算作为同一优化对象。

Shewhart、CUSUM、EWMA 和 SPRT 提供经典顺序检测构件[4--7]。RNED 与无漂移 CUSUM 的差别在于将累计超额除以监控时长，使平稳均值不随监测长度线性增长。深度重构模型可以学习复杂分布，但在每轴仅有早寿命样本、没有跨轴承训练时，仍需要一个可校准的事件决策层。

因此，本文不声称超过所有高容量诊断模型。BA-RNED 的目标是成为低功耗节点的第一层事件触发器，并在事件后触发更高复杂度诊断或人工确认。

## 3 BA-RNED 框架

### 3.1 三层架构

**节点层**提取每轴 RMS，估计早寿命基线，计算速率并锁存首次报警。**网关层**缓存事件 ID，执行去重、重传和安全上传；原始波形不出节点。**机群层**根据设备数 `F`、允许期望误事件数 `B` 和机群可接受概率 `α=B/F` 设定阈值；若单机误事件概率为 `q`，则要求 `q≤1-(1-α)^(1/F)`，对应健康校准分布的保守上置信分位数。样本不足时触发二级确认而非直接部署。

![图 1](figures/fig1_architecture.png)

**图 1** BA-RNED 的节点—网关—机群架构。图中网络层为虚拟 IoT 压力测试，不是真实 broker/无线实测。

### 3.2 节点检测规则

对第 `i` 个窗口的 RMS `f_i`，丢弃前 10 个 run-in 快照，用第 11–60 个快照的中位数估计基线 `b`。相对幅值和非负超额为

$$r_i=f_i/b,\qquad e_i=\max(r_i-1.2,0).$$

自第 61 个快照起，

$$S_t=\sum_{i=61}^{t}e_i,\qquad R_t=S_t/(t-60).$$

首次事件为

$$\tau=\min\{t\ge90:R_t>T\}.$$

阈值 `T=0.042407782315` 在 8 根 IMS 健康截尾轴承上按最大值的 1.01 倍冻结；最大健康分数为 0.041987903282。该 1.0% 裕量是复现边界，不是分布无关保证。任意轴触发即轴承事件，事件一旦锁存不再复位。

### 3.3 设计性质与边界

在平稳正漂移下，无界累积统计量几乎必然穿越固定阈值；时间平均速率的期望保持为单步超额均值，并对一阶尺度缩放不变。这些性质只说明为什么选择速率归一化，不推导数值误报上界。跨台架使用仍要求健康样本可交换；若工况分布变化，必须重新校准或加入工况感知确认层。

## 4 实验设计

### 4.1 数据与协议

主队列来自 IMS、XJTU-SY 和 FEMTO/PRONOSTIA，共 34 根失效轴承（IMS 4、XJTU-SY 13、FEMTO 17）和 8 根 IMS 健康截尾轴承。外部公共复制、CWRU 稳态正常数据和 27 根 XJTU/FEMTO 早寿命代理仅用于压力测试。虚拟环境包含 8 个台架、24 台健康截尾设备、8 台失效设备、每机 2 轴；网络测试包含 100 机、200 轴和 5,000 周期。

### 4.2 指标

检测指标为失效检出、首次事件时间和寿命提前量。健康指标为报警轴承比例及其区间。网络指标为投递率、p50/p95/p99 完成时延、重复率、过期事件率和字节数。统计检验为配对精确 McNemar 加 Holm 校正；Beta 后验仅用于解释 6:0 不一致对，不替代预声明检验。

## 5 结果

### 5.1 主结果

**表 1** RNED 主结果

| 数据集 | 失效数 | 检出 | 中位提前量 |
|---|---:|---:|---:|
| IMS | 4 | 4 | 14.5% |
| XJTU-SY | 13 | 13 | 46.3% |
| FEMTO/PRONOSTIA | 17 | 14 | 36.0% |
| 合计 | 34 | 31 | 28.8% |

合计精确 95% CI 为 76.3%–98.1%。三根漏检均在 FEMTO，说明短寿命或晚期可分退化仍是单 RMS 特征边界。

### 5.2 五层健康证据阶梯

**表 2** 健康证据阶梯

| 层级 | 证据 | 报警/样本 | 解释 |
|---|---|---:|---|
| 1 | IMS 校准内 | 0/8 | 仅校准内，不可外推 |
| 2 | IMS 实验组留一 | 1/8 | 小样本敏感性 |
| 3 | CWRU 稳态正常 | 0/3 台 | 非截尾、非多台架验证 |
| 4 | XJTU/FEMTO 早寿命代理 | 6/27 | 跨台架压力，非完整健康寿命 |
| 5 | 虚拟 8 台架健康截尾 | 2/24 | 合成机制，非确认性验证 |

虚拟多台架中，RNED 检出 8/8 失效，但健康误报为 8.33%（Wilson 95% CI：2.32%–25.85%）。该负结果十分重要：它证明冻结阈值不是天然零误报，也说明机群部署必须使用预算层或确认层。

### 5.3 D1 语义消融

**表 3** D1 与 RNED 的事件语义对比

| 方法 | 失效报警 | 健康截尾 | 在线语义 |
|---|---:|---:|---|
| RNED | 31/34 | 0/8 | 因果首次锁存 |
| D1 终点持久 | 33/34 | 参考区 2/8 | 需持续到终点确认 |
| D1-causal-three | 34/34 | 8/8 | 三连越限即锁存 |

D1-causal-three 的失效中位提前量为 40.1%，但健康截尾中位“提前量”为 7.1%，即所有健康样本均被当作早期事件。这说明 D1 的终点持久条件不是无关细节，而是抑制健康报警的核心机制，同时使其不能作为早期在线事件使用。

### 5.4 与经典基线的统计解释

RNED 对 Shewhart 和 EWMA 均有 6 个“RNED 检出、基线漏检”的不一致样本，反向为 0；原始 McNemar `p=0.03125`，四个预声明比较 Holm 校正后为 `p=0.0625`。Beta(7,1) 后验均值为 0.875，95% CI 为 0.590–0.996，方向概率 0.992。本文结论是：经验方向一致且效应集中在少数漏检样本上，但 frequentist 多重校正后不能声称显著优于 Shewhart/EWMA。

### 5.5 边缘开销

Jetson Orin NX 上，更新耗时 0.03235 ms/窗，吞吐 18,265.9 窗/s，能耗 0.0383 mJ/窗。8 bit 因果 RMS 特征量化保持 31/34、0/8 和 28.46% 中位提前量。该结果限于特征域，不代表完整 ADC 链路。

### 5.6 虚拟 IoT 网络

**表 4** 虚拟 100 机网络压力测试

| 场景 | 投递率 | p95 完成时延 | 重复率 | 字节减少 |
|---|---:|---:|---:|---:|
| 0% 丢包，0 重传 | 100.0% | 88.59 ms | 0% | 98.043% |
| 5% 丢包，1 重传 | 99.654% | 87.41 ms | 4.841% | 97.832% |
| 10% 丢包，1 重传 | 98.999% | 92.61 ms | 7.618% | 97.707% |
| 10% 丢包，3 重传 | 100.0% | 84.70 ms | 9.643% | 97.561% |

事件流在严重丢包下仍可将字节降低约 97.7%。一次重传已将 10% 丢包下的投递率提升到 99.0%；三次重传可达到 100%，但重复率接近 10%。网关因此必须在可靠性和重复包处理之间显式权衡。

### 5.7 机群预算

若单机误事件概率为 `q`，`F` 台设备至少发生一次误事件的概率为 `1-(1-q)^F`。当 `q=1%` 时，50 台为 39.5%，100 台为 63.4%。若要求 100 台设备中至少一次误事件概率不超过 5%，则单机概率需降至约 0.0513%。这解释了为什么单节点 0/8 不能直接部署到大规模机群。BA-RNED 的预算层将该要求转化为健康校准分位数和二级确认策略。

## 6 讨论

BA-RNED 回答了上一版投稿的五个主要风险。第一，健康证据不再依赖单一 0/8，而形成从真实校准到虚拟截尾的五层阶梯，并暴露非零误报风险。第二，D1-causal-three 消融表明 D1 的检出优势依赖终点确认，不能直接比较在线事件率。第三，统计结论保留 Holm 未显著事实，同时用后验解释方向性。第四，虚拟网络测试量化了丢包、重传、重复和字节代价。第五，贡献从简单检测器升级为预算感知事件框架，与 2025–2026 年轻量诊断和 RUL 论文形成清晰分工。

### 局限

1. 独立真实多台架健康截尾队列仍缺失。
2. 虚拟网络使用设定分布，不是真实 MQTT/CoAP、无线信道或安全栈测试。
3. 8 根健康样本不足以支持 1% 以下机群误事件分位数。
4. RMS 对载荷、转速和安装状态敏感。
5. 维护收益仍是场景分析，没有现场工单验证。

## 7 结论

本文提出 BA-RNED，把因果节点检测、事件网络传输和机群误报预算统一为一个 IIoT 预警协议。RNED 在公开失效队列上达到 31/34，校准内 0/8、留一 1/8；虚拟多台架显示健康误报 2/24，说明跨台架风险不可忽略。D1 因果消融显示其 34/34 在线检出伴随 8/8 健康报警，验证了事件语义的关键作用。虚拟网络显示事件流可减少约 97.6%–98.0% 字节，并在 10% 丢包下通过重传达到高可靠交付。BA-RNED 是一个边界明确的低成本事件基线，不是理论零误报或真实机群验证结论。

## 数据与代码

主实验、D1 复现、D1-causal-three 消融、健康证据阶梯、贝叶斯审计、虚拟网络结果和输入哈希由 `results/results_iotj_v20_risk_resolution_audit.json` 汇总。公共原始数据不重新分发。

## 参考文献

[1] D. Dyer and R. M. Stewart, “Detection of rolling element bearing damage by statistical vibration analysis,” *ASME J. Mech. Des.*, 1978.

[2] R. B. Randall and J. Antoni, “Rolling element bearing diagnostics—A tutorial,” *Mech. Syst. Signal Process.*, 2011.

[3] J. Antoni, “The spectral kurtosis,” *Mech. Syst. Signal Process.*, 2006.

[4] W. A. Shewhart, *Economic Control of Quality of Manufactured Product*, 1931.

[5] E. S. Page, “Continuous inspection schemes,” *Biometrika*, 1954.

[6] S. W. Roberts, “Control chart tests based on geometric moving averages,” *Technometrics*, 1959.

[7] A. Wald, “Sequential tests of statistical hypotheses,” *Ann. Math. Stat.*, 1945.

[8] L. Ruff et al., “Deep one-class classification,” *ICML*, 2018.

[9] R. Liu, D. Xiao, D. Lin, and W. Zhang, “Intelligent bearing anomaly detection for industrial IoT based on auto-encoder Wasserstein GAN,” *IEEE Internet Things J.*, 2024.

[10] K. Li, H. Zhang, and G. Lu, “Graph entropy-based early change detection in dynamical bearing degradation process,” *IEEE Internet Things J.*, 2024.

[11] K. You, P. Wang, and Y. Gu, “Toward efficient and interpretative rolling bearing fault diagnosis via quadratic neural network with Bi-LSTM,” *IEEE Internet Things J.*, 2024.

[12] X. Chen, S. Lu, H. Wang, and R. Yan, “Bearing RUL prediction using client selection and personalized aggregation enhancement in federated learning,” *IEEE Internet Things J.*, 2024.

[13] W. Huang, Z. Kuang, Y. Bi, and A. Liu, “MobiKanViT: Feature-enhanced lightweight CNN in mobile edge computing for real-time bearing fault diagnosis,” *IEEE Internet Things J.*, 2026.

[14] T. Hu, Z. Mo, and Z. Zhang, “Multitask pointwise mutual information learning for bearing RUL cross-domain imbalanced regression,” *IEEE Internet Things J.*, 2025.

[15] X. Pei, Y. Gao, X. Li, L. Gao, and X. Zhao, “Multilevel feature alignment method for advancing RUL prediction of rolling bearing,” *IEEE Internet Things J.*, 2025.

[16] G. Xin et al., “TASCNet: A lightweight bearing fault diagnosis network under unknown working conditions enabling industrial IoT,” *IEEE Internet Things J.*, 2026.

[17] H. Qiu, J. Lee, J. Lin, and G. Yu, “Robust performance degradation assessment methods for enhanced rolling element bearing prognostics,” *Adv. Eng. Inform.*, 2004.

[18] P. Nectoux et al., “PRONOSTIA: An experimental platform for bearings accelerated degradation tests,” *IEEE PHM*, 2012.
