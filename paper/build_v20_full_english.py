#!/usr/bin/env python3
"""Build the complete English v20 manuscript from the audited English base."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "paper" / "manuscript_en.tex"
OUTPUT = ROOT / "paper" / "manuscript_en_v20_full.tex"
BIB_SOURCE = ROOT / "paper" / "references.bib"
BIB_OUTPUT = ROOT / "paper" / "references_v20.bib"

SELECTED_KEYS = [
    "lei2018", "fink2020", "lu2023edge", "abadade2023tinyml",
    "dyer1978", "randall2011", "shewhart1931", "page1954",
    "roberts1959", "wald1945", "ruff2018", "hendrycks2017",
    "wang2020", "hora_submitted", "hora_unpublished", "smith2015",
    "wangd2018", "xiao2024dg", "su2019", "tuli2022tranad",
    "gebraeel2004", "chen2021", "xiao2024bcefl", "liu2023iot",
    "lee2007", "nectoux2012", "mcc5dataset", "javanmardi2024",
    "aimiyekagbon2024", "seferli2026", "siemens2026",
]


def between(text: str, start: str, end: str) -> str:
    start_index = text.index(start)
    end_index = text.index(end, start_index)
    return text[start_index:end_index].rstrip() + "\n"


def replace_between(text: str, start: str, end: str, replacement: str) -> str:
    start_index = text.index(start)
    end_index = text.index(end, start_index)
    return text[:start_index] + replacement.rstrip() + "\n\n" + text[end_index:]


def extract_bib_entries(text: str) -> dict[str, str]:
    entries: dict[str, str] = {}
    matches = list(re.finditer(r"@(\w+)\s*\{\s*([^,\s]+)\s*,", text))
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        entries[match.group(2)] = text[match.start() : end].strip() + "\n"
    return entries


def filter_citations(text: str) -> str:
    replacement_groups = {
        ("fink2020", "hoang2019"): ("fink2020",),
        ("dyer1978", "antoni2006", "randall2011"): ("dyer1978", "randall2011"),
        ("antoni2006", "randall2011"): ("randall2011",),
        ("su2019", "yang2024"): ("su2019",),
        ("su2019", "audibert2020usad", "tuli2022tranad", "xu2022anomaly"): ("su2019", "tuli2022tranad"),
        ("xiao2024bcefl", "jiang2024wtfl", "liu2023iot"): ("xiao2024bcefl", "liu2023iot"),
    }
    for old, new in replacement_groups.items():
        text = text.replace("\\cite{" + ",".join(old) + "}", "\\cite{" + ",".join(new) + "}")

    allowed = set(SELECTED_KEYS)

    def replace_command(match: re.Match[str]) -> str:
        keys = [key.strip() for key in match.group(1).split(",")]
        kept = tuple(key for key in keys if key in allowed)
        return "\\cite{" + ",".join(kept) + "}" if kept else ""

    text = re.sub(r"\\cite\{([^}]+)\}", replace_command, text)
    text = re.sub(r"~\\cite\{\}", "", text)
    text = re.sub(r"\s+\\cite\{\}", "", text)
    return text


def build_references() -> None:
    source_entries = extract_bib_entries(BIB_SOURCE.read_text(encoding="utf-8"))
    added_entries = r"""
@article{javanmardi2024,
  author  = {Javanmardi, Alireza and Aimiyekagbon, Osarenren Isoken and Bender, Andreas and Kimotho, James Kuria and Sextro, Walter and H{\"u}llermeier, Eyke},
  title   = {Remaining useful lifetime estimation of bearings operating under time-varying conditions},
  journal = {Proceedings of the PHM Society European Conference},
  volume  = {8},
  number  = {1},
  pages   = {9},
  year    = {2024},
  doi     = {10.36001/phme.2024.v8i1.4101}
}

@misc{aimiyekagbon2024,
  author       = {Aimiyekagbon, Osarenren Isoken},
  title        = {Run-to-failure data set of ball bearings subjected to time-varying load and speed conditions},
  howpublished = {Zenodo},
  year         = {2024},
  doi          = {10.5281/zenodo.10868257}
}

@misc{seferli2026,
  author       = {Seferli, Theodoros and others},
  title        = {Needle bearing endurance test data for early fault detection},
  howpublished = {Zenodo},
  year         = {2026},
  doi          = {10.5281/zenodo.20742913},
  note         = {Accessed Oct. 26, 2026}
}

@misc{siemens2026,
  title        = {{SCADA} dataset of a 2 {MW} {SIEMENS} wind turbine},
  howpublished = {Mendeley Data},
  year         = {2026},
  doi          = {10.17632/3sys4562ny},
  note         = {Accessed Oct. 26, 2026}
}
"""
    entries = {key: source_entries[key] for key in SELECTED_KEYS if key in source_entries}
    for match in re.finditer(r"@\w+\s*\{\s*([^,\s]+)\s*,", added_entries):
        if match.group(1) not in entries:
            end = added_entries.find("\n}\n", match.start())
            entries[match.group(1)] = added_entries[match.start() : end + 3]
    missing = [key for key in SELECTED_KEYS if key not in entries]
    if missing:
        raise ValueError(f"Missing BibTeX entries: {missing}")
    BIB_OUTPUT.write_text("\n".join(entries[key] for key in SELECTED_KEYS), encoding="utf-8")


def main() -> None:
    text = SOURCE.read_text(encoding="utf-8")

    author = r"""\author{\IEEEauthorblockN{Zhiguo Xiao\textsuperscript{1,2,*},
Yujing Guo\textsuperscript{2}, Yanqi Shao\textsuperscript{2},
Guangteng Chen\textsuperscript{2}, Dingbo Hou\textsuperscript{2},
Yuanyuan Liu\textsuperscript{2}, Xinyao Cao\textsuperscript{1,2}, Junli Liu\textsuperscript{1,2}}
\IEEEauthorblockA{\textsuperscript{1}School of Computer Science and Technology,
Beijing Institute of Technology, Beijing 100081, China\\
\textsuperscript{2}School of Computer Science and Technology,
Changchun University, Changchun 130022, China}
\IEEEauthorblockA{\textsuperscript{*}Corresponding author: Zhiguo Xiao
(email: 3220215169@bit.edu.cn)}}"""
    text = replace_between(text, "\\author{", "\\title{", author)
    text = replace_between(
        text,
        "\\title{",
        "\\markboth{",
        r"\title{Budget-Aware Event-Level Rate-Normalized Early Warning for Bearings in Industrial IoT: Validation in a Virtual IoT Environment}",
    )
    text = replace_between(
        text,
        "\\markboth{",
        "\\begin{document}",
        "\\markboth{IEEE Internet of Things Journal}%\n{Xiao \\MakeLowercase{\\textit{et al.}}: Budget-Aware Event-Level Rate-Normalized Early Warning}",
    )

    abstract = r"""\begin{abstract}
Predictive maintenance in the Industrial Internet of Things requires joint control of edge detection, event delivery, and fleet-scale false-event budgets. This paper proposes budget-aware event-level rate-normalized detection (BA-RNED). A node converts nonnegative RMS excess above an early-life baseline into a time-averaged excess rate and latches the first alarm at a frozen threshold; a gateway deduplicates and retries compact events; a fleet layer converts a false-event budget into a calibration-quantile requirement. On IMS, XJTU-SY, and FEMTO/PRONOSTIA, RNED detects 31/34 failures (exact 95\% CI: 76.3--98.1\%) with median leads of 14.5\%, 46.3\%, and 36.0\% of life; five healthy-evidence tiers show that it is not inherently alarm-free. A D1-causal-three ablation shows that public D1's 33/34 advantage depends on endpoint confirmation: without it, detection is 34/34 but healthy-censored alarms are 8/8. On Jetson Orin NX, an update costs 0.03235~ms and 0.0383~mJ. In a 100-machine simulation, event traffic saves 98.043\% bytes without loss, and one retry under 10\% loss achieves 98.999\% delivery. BA-RNED contributes a reproducible causal event protocol and budget boundary rather than a more complex diagnostic model.
\end{abstract}

\begin{IEEEkeywords}
Industrial IoT, bearing prognostics, edge computing, early warning, event budgeting, virtual IoT, false alarms.
\end{IEEEkeywords}"""
    text = replace_between(text, "\\begin{abstract}", "\\section{Introduction}", abstract)

    introduction = r"""\section{Introduction}\label{sec:intro}
Predictive maintenance in industrial IoT must output a latchable, transmittable, and budgetable first maintenance event rather than a sequence of sample-level anomaly scores; machinery-health and deep-learning surveys frame the broader data-to-decision pipeline~\cite{lei2018,fink2020}. Sensor nodes continuously acquire vibration, but raw-waveform uploading consumes bandwidth and energy. If the cloud receives continuous scores, it must still choose an alarm threshold, confirmation rule, and fleet-level tolerance. Treating every deviation as a fault causes unnecessary shutdowns, whereas confirming a state only at the endpoint removes the online value of a warning.

The research gap is the absence of a failure-versus-drift discriminator that, under a zero-false-alarm constraint, simultaneously supports causal updating, a fixed threshold, interpretability, edge deployment, lead-time reporting, and fleet-event accounting. Accordingly, this paper examines four questions: (1) which trajectory statistic separates failure-bound degradation from benign drift under healthy-censored controls; (2) whether that statistic yields a causal, fixed-threshold detector with $O(1)$ state updates; (3) whether rate normalization is necessary and effective relative to classical, deep-reconstruction, and supervised-RUL baselines; and (4) whether edge cost, quantization robustness, and fleet-event capacity satisfy IIoT deployment constraints.

The central idea is to convert ``how far above baseline is the amplitude'' into ``what is the average nonnegative excess during monitoring.'' For each axis, RMS snapshots define an early-life median baseline $b$; the relative amplitude is $r_t=f_t/b$ and the nonnegative excess is $e_t=\max(r_t-1.2,0)$. Failure-bound excess tends to occur more often and with larger magnitude, whereas benign transients are diluted by the monitoring duration. A single causal threshold therefore tests accumulated excess load rather than instantaneous amplitude.

The main contributions are as follows. First, we propose the rate-normalized RMS-excess detector and factor the time-averaged excess into exceedance frequency and positive-excess magnitude; full-life and fixed-window audits prevent persistence alone from being overinterpreted. Second, under one causal protocol and zero-false-alarm calibration, we compare classical detectors, the closest RMS-ratio rule, persistence rules, deep reconstruction models, and an RUL baseline, including leave-one-IMS-experiment-out sensitivity. Third, we measure edge latency and energy, add causal 8-bit feature quantization, deterministic byte accounting, and a modeled fleet-event budget, while making explicit that the study validates a node-side detector rather than a complete multi-node IIoT system.

The remainder is organized as follows. Section~\ref{sec:related} reviews related work. Section~\ref{sec:framework} presents BA-RNED. Section~\ref{sec:method} formalizes the detector. Section~\ref{sec:experiments} describes the experimental protocol. Section~\ref{sec:results} reports results. Section~\ref{sec:discussion} discusses mechanism, statistical boundaries, and engineering implications. Section~\ref{sec:conclusion} concludes."""
    text = replace_between(text, "\\section{Introduction}", "\\section{Related Work}", introduction)

    related = r"""\section{Related Work}\label{sec:related}
\subsection{Health indicators and early-warning rules}
RMS, kurtosis, skewness, and peak-to-peak amplitude have long been used for bearing monitoring~\cite{dyer1978}, while envelope analysis characterizes impulsive faults~\cite{randall2011}. Smith and Randall benchmarked common indicators on CWRU data~\cite{smith2015}, and reviews emphasize monotonicity, robustness, and early sensitivity~\cite{wangd2018}. These lightweight, causal indicators are suitable for edge nodes, but one snapshot cannot distinguish persistent degradation from transient drift. This work does not introduce a new index; it converts RMS excess into a time-averaged event rate.

Cross-condition fault diagnosis is related but different: transfer, domain-adaptation, and domain-generalization methods reduce training/deployment distribution mismatch~\cite{xiao2024dg}. They mainly identify fault categories, whereas this work uses each machine's early-life baseline for single-machine degradation warning.

\subsection{Classical online change detection}
Shewhart monitors the current sample~\cite{shewhart1931}, CUSUM accumulates positive deviations~\cite{page1954}, EWMA smooths exponentially~\cite{roberts1959}, and SPRT applies sequential likelihood ratios~\cite{wald1945}. These are the closest online baselines, but unbounded accumulators can be pushed upward by positive healthy excess under a fixed zero-false-alarm threshold. We quantify this failure under the same protocol and identify time normalization as the key difference.

\subsection{Unsupervised OOD and anomaly detection}
Deep one-class and confidence-based OOD methods learn a healthy or known-class distribution~\cite{ruff2018,hendrycks2017}; stochastic recurrent networks and TranAD represent time-series anomaly detection~\cite{su2019,tuli2022tranad}. These methods usually score sample- or window-level deviation and do not by themselves distinguish transient drift from persistent accumulated excess.

\subsection{Supervised RUL regression}
From early neural networks to attention-based models, RUL research has developed a rich literature~\cite{wang2020,gebraeel2004,chen2021}. These methods require degradation labels and predict remaining life. This work requires only a healthy baseline and outputs a maintenance event, so the RUL model is an advantage-mismatched comparison rather than an equivalent task.

\subsection{IoT edge intelligence}
Edge computing and TinyML support on-device processing under latency, bandwidth, and energy constraints~\cite{lu2023edge,abadade2023tinyml}; federated diagnosis and unsupervised IoT time-series learning further characterize large-scale systems~\cite{xiao2024bcefl,liu2023iot}. BA-RNED is a low-state first-event trigger for such systems, not a replacement for high-capacity diagnosis.

\subsection{Closest label-free warning protocol}
An unpublished protocol evaluated RMS-ratio rules on IMS and tested transfer across three datasets~\cite{hora_submitted,hora_unpublished}. We retain its early-life baseline and 1.2-fold excess idea, but distinguish a single-sample rule, endpoint-persistent D1, and the causal event rate under identical calibration."""
    text = replace_between(text, "\\section{Related Work}", "\\section{Method}", related)

    framework = r"""\section{BA-RNED: Budget-Aware Event-Level Framework}\label{sec:framework}
BA-RNED extends the rate-normalized detector to a node--gateway--fleet protocol without changing the node's online state. The node computes RMS, calibrates an early-life baseline, updates a time-averaged excess rate, and latches one alarm. It stores only cumulative excess $S_t$ and monitoring count $t$.

The gateway receives an event ID, timestamp, triggering axis, score, and a compact diagnostic summary. The ID gives idempotent deduplication; data and ACK packets may be independently lost and retried. Raw waveforms are not forwarded.

Given fleet size $F$, tolerable expected false events $B$, and acceptable fleet probability $\alpha=B/F$, a per-machine false-event probability $q$ must satisfy
\begin{equation}
1-(1-q)^F\le\alpha,\qquad q\le1-(1-\alpha)^{1/F}.
\end{equation}
The point estimate is not used directly to choose a threshold. Instead, the fleet layer uses a conservative upper confidence quantile of the healthy calibration distribution; when the tail quantile cannot be estimated, it triggers secondary confirmation or human review rather than extrapolating zero false alarms from a small sample."""
    text = text.replace("\\section{Method}", framework + "\n\n\\section{System Model and Rate-Normalized Detector}", 1)

    overview = between(text, "\\subsection{Overview}", "\\subsection{Problem formalization and notation}")
    problem = between(text, "\\subsection{Problem formalization and notation}", "\\subsection{Data and feature representation}")
    offline = between(text, "\\subsection{Offline discriminative feature: cumulative excess (RQ2)}", "\\subsection{Online rate-normalized detector (RQ3)}")
    online = between(text, "\\subsection{Online rate-normalized detector (RQ3)}", "\\subsection{Pseudocode and cost}")
    pseudocode = between(text, "\\subsection{Pseudocode and cost}", "\\subsection{Online baselines (RQ4)}")
    data = between(text, "\\subsection{Data and feature representation}", "\\subsection{Evaluation protocol}")
    evaluation = between(text, "\\subsection{Evaluation protocol}", "\\subsection{Offline discriminative feature: cumulative excess (RQ2)}")
    baselines = between(text, "\\subsection{Online baselines (RQ4)}", "\\subsection{Theoretical basis of rate normalization}")

    theory = r"""\subsection{Theoretical properties}\label{sec:theory}
\textbf{Proposition 1 (failure of unbounded accumulators).} If the healthy or benign-drift excess sequence has stationary mean $\mu>0$, then an unbounded cumulative statistic crosses every finite fixed threshold almost surely. Thus, under a zero-false-alarm constraint, its threshold must grow with monitoring duration, which can miss late real failures.

\textbf{Proposition 2 (rate normalization).} Under the same mean assumption, $\mathbb{E}[R_t]=\mu$ for all monitored durations, and $R_t$ is first-order scale equivariant. These properties explain why time normalization is preferred, but they do not provide a numeric finite-population false-alarm or transfer guarantee."""

    method = (
        "\\section{System Model and Rate-Normalized Detector}\\label{sec:method}\n\n"
        + overview.replace("(RQ2)", "").replace("(RQ3)", "").replace("(RQ4)", "")
        + "\n\n" + problem + "\n\n" + offline + "\n\n" + online + "\n\n" + pseudocode + "\n\n" + theory + "\n\n"
        + "\\section{Experimental Design}\\label{sec:experiments}\n\n"
        + "\\subsection{Datasets}\n" + between(data, "\\subsection{Data and feature representation}", "").split("\n", 1)[1].rstrip()
        + "\n\n" + evaluation
        + "\n\n" + baselines.replace("(RQ4)", "")
    )
    text = replace_between(
        text,
        "\\section{System Model and Rate-Normalized Detector}",
        "\\section{Experiments and Results}",
        method,
    )
    text = text.replace("\\section{Experiments and Results}", "\\section{Results}", 1)
    text = re.sub(r"\s+\(RQ[1-4]\)", "", text)
    text = text.replace("Results are organized by RQ.", "Results are organized by the four questions in the introduction.")

    external_and_risk = r"""\subsection{External alarm coverage and operating-condition boundary}
On LDM\_UPB, the frozen protocol triggers all 11 protocol-qualified experiments without changing threshold, window, or cross-axis rule; historical-rule coverage is 11/11. The median monitoring-segment lead is 95.9\%, but LDM\_UPB has no preregistered fault onset, so this is not a remaining-life fraction or proof of early fault detection. Eight of 11 experiments trigger at the earliest permitted monitoring point. Figure~\ref{fig:condition} shows that B01 triggers when dynamic load changes by 35.5\% and speed by 27.5\%, whereas B11 triggers later under a more stable condition. Thus, a fixed RMS baseline can mistake an operating-condition response for persistent excess. A ridge-residual probe delays five of eight early alarms and reduces median lead to 63.2\%, but it is post hoc and does not change the main method.

\begin{figure}[!t]
\centering
\includegraphics[width=\linewidth]{../figures/fig10_ldm_upb_condition_sensitivity.pdf}
\caption{RMS and operating-condition changes in the external LDM\_UPB replication.}
\label{fig:condition}
\end{figure}

\subsection{Healthy evidence and system-level stress tests}\label{sec:risk}
\subsubsection{Five-tier healthy-evidence ladder}
Table~\ref{tab:healthy-evidence} expands a single 0/8 calibration result into real calibration, real held-out data, steady-state normal records, cross-rig early-life proxies, and a virtual multi-rig mechanism stress test. The tiers do not establish a universal false-alarm rate; they expose its uncertainty.

\begin{table}[!t]
\centering
\caption{Five-tier healthy-evidence ladder.}
\label{tab:healthy-evidence}
\begin{tabular}{clrl}
\toprule
Tier & Evidence source & Alarms/sample & Boundary \\
\midrule
1 & IMS in-sample censored & 0/8 & Calibration only \\
2 & IMS leave-one-experiment & 1/8 & Small-sample sensitivity \\
3 & CWRU steady normal & 0/3 & Not censored or multi-rig \\
4 & XJTU/FEMTO pseudo-health & 6/27 & Not full healthy life \\
5 & Virtual eight-rig cohort & 2/24 & Mechanism stress only \\
\bottomrule
\end{tabular}
\end{table}

\subsubsection{D1-causal-three}
Public D1 uses a 1.2 threshold, three consecutive crossings, and persistence to the endpoint. It detects 33/34 in this cohort but produces reference-zone episodes in 2/8 censored bearings. D1-causal-three removes endpoint persistence and latches at the first third consecutive crossing: it detects 34/34 failures but alarms on all 8/8 censored bearings. Therefore D1's apparent advantage depends on future endpoint confirmation rather than a stronger online event rule.

\subsubsection{Shewhart/EWMA interpretation}
Against Shewhart and EWMA, RNED has six favorable and zero unfavorable discordant pairs. The raw exact McNemar $p$ is 0.03125, but Holm correction over four comparisons gives 0.0625. A Beta$(7,1)$ posterior has mean 0.875 and equal-tailed 95\% interval 0.590--0.996; it is directional auxiliary evidence, not a replacement for the frequentist test.

\subsubsection{Virtual multi-node IoT network}
The deterministic network has 100 dual-axis machines, 200 axes, and 5{,}000 cycles. Event probability is 0.01 per axis, inter-axis correlation is 0.2, and uplink latency is lognormal with median 30.0~ms. Data and ACK packets are independently lost; event IDs are deduplicated and unreachable events are censored at 1{,}000~ms. Without loss, event traffic saves 98.043\% bytes. Under 10\% loss, one retry gives 98.999\% delivery and 92.61~ms p95 completion; three retries give 100\% and 84.70~ms but 9.643\% duplicates.

\subsubsection{From one machine to fleet capacity}
If each machine has false-event probability $q$, $F$ machines produce at least one false event with probability $1-(1-q)^F$. For $q=1\%$, 100 machines reach 63.4\%; keeping the probability below 5\% requires $q\approx0.0513\%$. Eight calibration trajectories cannot identify such a tail quantile reliably, motivating conservative upper confidence quantiles and secondary confirmation."""
    text = text.replace("\\section{Discussion and Limitations}", external_and_risk + "\n\n\\section{Discussion and Limitations}", 1)

    discussion = r"""\section{Discussion}\label{sec:discussion}
\subsection{Answers to the core questions}
First, full-life time-averaged excess completely separates IMS failure from censored bearings, while instantaneous exceedances overlap; fixed-window separation weakens. The signal combines exceedance frequency, positive magnitude, and late degradation information, so it is not a universal physical law. Second, the online rate detects 31/34 with 0/8 in-sample alarms, 1/8 held-out alarms, and 14.5--46.3\% median leads under the frozen protocol. Third, compared with the single-sample RMS ratio and unnormalized CUSUM, discordant pairs are 6/0 and 10/0, but these are post hoc multiple comparisons. D1's 33/34 depends on endpoint semantics; deep and RUL baselines are weaker under the same calibration but do not rule out other models. Fourth, measured edge cost and deterministic network accounting satisfy the tested environment; the absence of multiple boards, real ADCs, real networks, and field data prevents a complete IIoT-system claim.

\subsection{Mechanism interpretation and evidence boundary}
Instantaneous exceedance alone cannot separate failure from drift. Full-life excess separates the small IMS cohort, but fixed common-window separation is not significant; occupation and positive magnitude alone do not explain the result. RMS is robust relative to other scalar features in this protocol, yet it remains condition-sensitive. The physical interpretation is an energy proxy for cumulative degradation, not a fitted damage model.

\subsection{Relation to prior work}
BA-RNED does not replace sequential testing, deep anomaly detection, diagnosis, or RUL regression. It supplies a causal, low-state first-event rule and an explicit communication/fleet budget layer that can trigger a higher-capacity diagnostic model or human confirmation.

\subsection{Failure conditions and negative results}
The main misses are low-amplitude, low-persistence FEMTO bearings. Cross-rig pseudo-health and the virtual cohort reveal false-alarm risk. The hierarchical rule can reach 34/34 and 0/8 in-sample but raises held-out alarms to 2/8 and lacks independent validation. These results bound the claim rather than invalidate the method.

\subsection{Engineering significance}
One accumulator, one counter, sub-millisecond updates, compact events, and explicit retry/deduplication make the rule suitable as a first-layer trigger. Fleet thresholds require calibration capacity planning; merely raising the node threshold first sacrifices low-energy failures.

\subsection{Limitations and future work}
The evidence lacks an independent real multi-rig healthy-censored cohort, real wireless broker experiments, multi-board energy measurements, and maintenance-value validation. Future work should collect preregistered external cohorts, test MQTT/CoAP and security stacks, incorporate condition-aware calibration, and connect events to work orders."""
    text = replace_between(text, "\\section{Discussion and Limitations}", "\\section{Conclusion}", discussion)

    conclusion = r"""\section{Conclusion}\label{sec:conclusion}
BA-RNED unifies causal node detection, gateway event transport, and fleet false-event budgeting. RNED detects 31/34 failed bearings with 14.5\%, 46.3\%, and 36.0\% median leads on IMS, XJTU-SY, and FEMTO; in-sample and held-out censored alarms are 0/8 and 1/8, while the virtual eight-rig cohort gives 2/24 and rules out an inherent zero-alarm claim. D1-causal-three shows that D1's apparent 33/34 advantage depends on endpoint confirmation. The virtual 100-machine network saves 97.6--98.0\% bytes and reaches 100\% delivery with three retries at the cost of 9.643\% duplicates. BA-RNED is therefore a low-state, reproducible, boundary-aware IIoT event baseline, not a theoretical zero-false-alarm guarantee or a complete physical-network deployment claim."""
    text = replace_between(text, "\\section{Conclusion}", "\\appendix", conclusion)

    proofs = r"""\appendix
\section{Proofs of Propositions 1--2}\label{app:proof}
Let $n=t-60$, reindex the excess sequence as $e_1,\dots,e_n$, $S_n=\sum_{i=1}^{n}e_i$, and $R_n=S_n/n$.

\textbf{Proof of Proposition 1.} If $\{e_i\}$ is strictly stationary and ergodic with $\mathbb{E}[e_1]=\mu>0$, Birkhoff's ergodic theorem gives $S_n/n\to\mu$ almost surely. Hence $S_n\to\infty$ and every finite fixed threshold is eventually crossed. For CUSUM, drift compensation $\delta<\mu$ leaves a positive-mean increment. For SPRT under a misspecified healthy null, the expected pre-reset log-likelihood increment can be positive; the reflected process is then a positive-drift random walk and is unbounded. Thus, a fixed zero-false-alarm threshold for an unbounded accumulator must grow with duration, which can miss late real failures. Table~\ref{tab:detectors} reports the corresponding empirical failures (CUSUM 21/34 and SPRT 1/34).

\textbf{Proof of Proposition 2.} Stationarity and linearity give $\mathbb{E}[R_n]=(1/n)\sum_i\mathbb{E}[e_i]=\mu$, independent of monitoring duration. For $a>0$, $R_n(ae)=aR_n(e)$, yielding first-order scale equivariance. These are qualitative design properties only; they require the stated mean assumption and do not provide a numeric finite-population false-alarm bound."""
    text = replace_between(text, "\\appendix", "\\section{Added contrast experiments and baseline definitions}", proofs)

    declarations = r"""\section*{Declarations}
\textbf{Data availability.} The primary experiment uses public IMS~\cite{lee2007}, XJTU-SY~\cite{wang2020}, and FEMTO/PRONOSTIA~\cite{nectoux2012}; task-boundary analyses additionally use CWRU~\cite{smith2015}, MCC5~\cite{mcc5dataset}, LDM\_UPB~\cite{javanmardi2024,aimiyekagbon2024}, DLR needle-bearing data~\cite{seferli2026}, and Siemens SCADA~\cite{siemens2026}. Raw public data are not redistributed.

\textbf{Author contributions.} Zhiguo Xiao: conceptualization, methodology, software, formal analysis, writing---original draft, review, and editing. Yujing Guo, Yanqi Shao, Guangteng Chen, Dingbo Hou, Yuanyuan Liu, Xinyao Cao: data curation, investigation, visualization, review, and editing. Junli Liu: supervision and funding acquisition.

\textbf{Conflict of interest.} The authors declare no conflict of interest.

\textbf{Funding.} Jilin Provincial Scientific and Technological Development Program (20260102278JC); 2023 Postgraduate Education and Teaching Reform Research Project of Jilin Province (JJKH20230100YJG); Scientific Research Project of Jilin Provincial Department of Education (JKH20262389CY).

\textbf{AI use disclosure.} AI tools assisted language editing and translation. The authors produced and verified all technical content, derivations, experiments, and conclusions and take full responsibility for the manuscript."""
    text = replace_between(text, "\\section*{Declarations}", "\\bibliographystyle", declarations)
    text = text.replace("\\bibliography{IEEEabrv,references}", "\\bibliography{references_v20}")
    text = text.replace("\\subsection{Online baselines }", "\\subsection{Online baselines}")
    text = text.replace(
        "Inputs: the cross-rig per-axis false-alarm rate $p=0.185$ (per 150-snapshot early-life segment) and the data-driven excess bound $M=4.43$ (censored maximum RMS ratio 5.63 minus 1.2). Fleet aggregation uses an independent Bernoulli model per axis: $P(\\text{fleet $\\geq$1 false alarm per epoch})=1-(1-p)^{Mk}$. The Proposition~4 fleet union bound is $Mk\\cdot\\mathrm{e}^{-2n_0(T-\\mu)^2/M^2}$, which with $M=4.43$, $\\mu=0.0153$, $T=0.0424$ barely decays at realistic horizons, showing the theory gives a form guarantee rather than a fleet-level numeric guarantee.",
        "Let a fleet contain $F$ machines with $K$ axes each, and let $q$ be the fixed marginal false-event probability. The independent-axis model gives $P_{\\mathrm{fleet}}=1-(1-q)^{FK}$. With the cross-rig early-life proxy $q=0.185$, the model exposes the noncomposition of per-node zero alarms but is not an estimated real-fleet false-alarm rate or a distribution-free upper bound.",
    )
    text = text.replace(
        "On the data path, raw waveforms never leave the node and only alarm events travel uplink, matching the massive-node, narrow-uplink constraints of IIoT; Proposition~4 gives a duration-independent time-uniform false-alarm bound that makes every uplinked alarm statistically trustworthy. This three-layer form is annotated in Fig.~\\ref{fig:arch}, and it answers the ``removing the edge constraint leaves the evaluation intact'' concern: energy, bandwidth, and fleet scale are not evaluation appendices but the constraints that drive the design choice of rate normalization ($O(1)$, duration-independent, provably bounded false alarms).",
        "On the data path, raw waveforms never leave the node and only compact alarm events travel uplink, matching massive-node and narrow-uplink IIoT constraints. This three-layer form is annotated in Fig.~\\ref{fig:arch}; energy, bandwidth, and fleet scale are design constraints rather than evaluation appendices. The theoretical properties remain qualitative and do not make individual uplinked alarms statistically trustworthy.",
    )
    text = text.replace(
        "Aggregating the cross-rig per-axis false-alarm rate $p=0.185$ (per 150-snapshot early-life segment) as independent Bernoulli trials (Fig.~\\ref{fig:fleet}): one machine with two axes raises at least one false alarm in an epoch with probability $1-(1-p)^2\\approx33.6\\%$; this reaches 98.3\\% at 10 machines and saturates to $100\\%$ at 100 machines (37 expected false alarms per epoch, 370 at 1000). The independence assumption is bootstrap-checked (Appendix~\\ref{app:baselines}): only 2 of the 54 pseudo-health axes belong to bearings with mixed-axis alarms, so alarm events cluster within bearings and the axis-independent model is conservative; under a bearing-clustered rate $p=0.222$, a 100-machine fleet has about 22 false alarms per epoch, with a bootstrap 95\\% CI of $[11,67]$---the order of magnitude and the conclusion (fleet false alarms are unavoidable in bulk) are unchanged. To hold the fleet-wide ``at least one false alarm per epoch'' probability below 5\\%, the per-axis rate must be tightened to roughly $\\alpha/(Mk)$, i.e., about 0.0256\\% for 100 machines and 2 axes---a $\\approx720\\times$ tightening relative to 18.5\\% that cannot be achieved by raising one threshold without losing detection. This agrees with the union-bound composition of Proposition~4: bounding the per-axis anytime-valid bound over $Mk$ axes, the data-driven excess bound $M=4.43$ makes the guarantee essentially non-decaying at realistic horizons (about 0.99 per axis at $n_0=30$), so the theory provides a \\emph{form} guarantee (false-alarm probability does not diverge with monitoring duration) while fleet-level numeric control can only come from empirical calibration and fleet-level alarm management (multi-axis confirmation, event aggregation, hierarchical thresholds).",
        "Using the cross-rig early-life proxy $q=0.185$ as a modeled per-axis probability (Fig.~\\ref{fig:fleet}), a two-axis machine has at least one false event with probability $1-(1-q)^2\\approx33.6\\%$; 10 machines reach 98.3\\%, and 100 dual-axis machines yield 37 expected false events. A bearing-clustered rate $q=0.222$ gives about 22 false events per 100 machines, with bootstrap 95\\% CI $[11,67]$. Keeping the fleet-wide probability below 5\\% would require $q\\approx0.0256\\%$, roughly a $720\\times$ tightening relative to 18.5\\%. These are modeled capacity estimates, not measured fleet alarms; practical control requires empirical calibration, multi-axis confirmation, event aggregation, and hierarchical thresholds.",
    )
    text = text.replace("Data-side validation of Propositions 1--4:", "Data-side validation of the rate-normalization mechanism:")
    text = text.replace("the bearing-level rate is $p=0.222$", "the bearing-level rate is $q=0.222$")
    text = filter_citations(text)
    text = re.sub(r"RQ[1-4]", "the corresponding question", text)

    OUTPUT.write_text(text, encoding="utf-8")
    build_references()
    print(f"Wrote {OUTPUT} ({len(text.splitlines())} lines)")
    print(f"Wrote {BIB_OUTPUT} ({len(SELECTED_KEYS)} entries)")


if __name__ == "__main__":
    main()
