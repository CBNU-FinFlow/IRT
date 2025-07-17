import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import os
import pickle
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from collections import deque
import json
import copy
import seaborn as sns
from html_dashboard import generate_dashboard
from immune_visualization import create_visualizations
from typing import Dict, List, Tuple, Any, Optional

warnings.filterwarnings("ignore")

# 디렉토리 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# 디렉토리 생성
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def create_timestamped_directory(base_dir, prefix="run"):
    """타임스탬프 기반 디렉토리 생성"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped_dir = os.path.join(base_dir, f"{prefix}_{timestamp}")
    os.makedirs(timestamped_dir, exist_ok=True)
    return timestamped_dir


class DecisionAnalyzer:
    """의사결정 과정 분석 클래스"""

    def __init__(self):
        self.decision_log = []
        self.risk_thresholds = {"low": 0.3, "medium": 0.5, "high": 0.7, "critical": 0.9}
        self.crisis_detection_log = []  # 상세 위기 감지 로그

    def _process_detailed_tcell_analysis(
        self, tcell_analysis, dominant_risk, risk_features, dominant_risk_idx
    ):
        """상세 T-cell 분석 처리"""
        # 기본 T-cell 분석 정보 (기존 형식 유지)
        basic_analysis = {
            "crisis_level": float(tcell_analysis.get("crisis_level", 0.0)),
            "dominant_risk": dominant_risk,
            "risk_intensity": float(risk_features[dominant_risk_idx]),
            "overall_threat": self._assess_threat_level(
                tcell_analysis.get("crisis_level", 0.0)
            ),
        }

        # 상세 위기 감지 로그가 있는 경우 추가
        if (
            isinstance(tcell_analysis, dict)
            and "detailed_crisis_logs" in tcell_analysis
        ):
            detailed_logs = tcell_analysis["detailed_crisis_logs"]
            analysis = basic_analysis.copy()
            analysis["detailed_crisis_detection"] = {
                "active_tcells": len(detailed_logs),
                "crisis_detections": [],
            }

            for tcell_log in detailed_logs:
                if tcell_log.get("activation_level", 0.0) > 0.15:  # 위기 감지 임계값
                    crisis_detection = {
                        "tcell_id": tcell_log.get("tcell_id", "unknown"),
                        "timestamp": tcell_log.get("timestamp", ""),
                        "activation_level": tcell_log.get("activation_level", 0.0),
                        "crisis_level_classification": tcell_log.get(
                            "crisis_level", "normal"
                        ),
                        "crisis_indicators": tcell_log.get("crisis_indicators", []),
                        "decision_reasoning": tcell_log.get("decision_reasoning", []),
                        "feature_contributions": tcell_log.get(
                            "feature_contributions", {}
                        ),
                        "market_state_analysis": tcell_log.get("market_state", {}),
                    }
                    analysis["detailed_crisis_detection"][
                        "crisis_detections"
                    ].append(crisis_detection)

                    # 위기 감지 로그에 추가
                    self.crisis_detection_log.append(
                        {
                            "timestamp": tcell_log.get("timestamp", ""),
                            "tcell_id": tcell_log.get("tcell_id", "unknown"),
                            "crisis_info": crisis_detection,
                        }
                    )

            return analysis

        return basic_analysis

    def log_decision(
        self,
        date,
        market_features,
        tcell_analysis,
        bcell_decisions,
        final_weights,
        portfolio_return,
        crisis_level,
    ):
        """의사결정 과정 기록"""

        # 지배적 위험 분석
        risk_features = market_features[:5]
        dominant_risk_idx = np.argmax(np.abs(risk_features - np.mean(risk_features)))
        risk_map = {
            0: "volatility",
            1: "correlation",
            2: "momentum",
            3: "liquidity",
            4: "macro",
        }
        dominant_risk = risk_map.get(dominant_risk_idx, "volatility")

        # T-cell 분석 처리
        tcell_analysis_result = self._process_detailed_tcell_analysis(
            tcell_analysis, dominant_risk, risk_features, dominant_risk_idx
        )

        decision_record = {
            "date": (
                date.strftime("%Y-%m-%d") if hasattr(date, "strftime") else str(date)
            ),
            "market_features": (
                market_features.tolist()
                if hasattr(market_features, "tolist")
                else list(market_features)
            ),
            "tcell_analysis": tcell_analysis_result,
            "bcell_decisions": self._serialize_bcell_decisions(bcell_decisions),
            "final_weights": (
                final_weights.tolist()
                if hasattr(final_weights, "tolist")
                else list(final_weights)
            ),
            "portfolio_return": float(portfolio_return),
            "memory_activated": bool(crisis_level > 0.3),
        }

        self.decision_log.append(decision_record)

    def _serialize_bcell_decisions(self, bcell_decisions):
        """B-cell 결정 정보 직렬화 (상세 분석 포함)"""
        if not bcell_decisions:
            return []

        serialized = []
        for bcell in bcell_decisions:
            serialized_bcell = {}
            for key, value in bcell.items():
                if isinstance(value, (int, float, str, bool)):
                    serialized_bcell[key] = value
                else:
                    serialized_bcell[key] = str(value)

            # B-cell 전문성 및 의사결정 근거 분석 추가
            if "risk_type" in bcell:
                serialized_bcell["specialization_analysis"] = (
                    self._analyze_bcell_specialization(bcell)
                )

            serialized.append(serialized_bcell)

        return serialized

    def _analyze_bcell_specialization(self, bcell_decision):
        """B-cell 전문성 및 의사결정 근거 분석"""
        risk_type = bcell_decision.get("risk_type", "unknown")
        activation_level = bcell_decision.get("activation_level", 0.0)
        antibody_strength = bcell_decision.get("antibody_strength", 0.0)

        # 전문성 평가
        specialization_score = bcell_decision.get("strategy_contribution", 0.0)

        # 의사결정 근거 생성
        decision_reasoning = []

        # 활성화 근거
        if activation_level > 0.7:
            decision_reasoning.append(
                f"높은 활성화 레벨({activation_level:.3f})로 인한 강력한 대응 필요"
            )
        elif activation_level > 0.5:
            decision_reasoning.append(
                f"중간 활성화 레벨({activation_level:.3f})로 인한 적극적 대응"
            )
        elif activation_level > 0.3:
            decision_reasoning.append(
                f"낮은 활성화 레벨({activation_level:.3f})로 인한 보수적 대응"
            )

        # 위험 유형별 전문성 근거
        risk_reasoning = {
            "volatility": f"시장 변동성 위험에 특화된 안전 자산 중심 포트폴리오 구성",
            "correlation": f"상관관계 위험에 특화된 분산 투자 전략 적용",
            "momentum": f"모멘텀 위험에 특화된 추세 추종 전략 활용",
            "liquidity": f"유동성 위험에 특화된 대형주 중심 포트폴리오 구성",
            "memory_recall": f"과거 위기 경험을 바탕으로 한 검증된 대응 전략 적용",
        }

        if risk_type in risk_reasoning:
            decision_reasoning.append(risk_reasoning[risk_type])

        # 항체 강도 근거
        if antibody_strength > 0.8:
            decision_reasoning.append(
                f"높은 항체 강도({antibody_strength:.3f})로 강력한 방어 전략 수행"
            )
        elif antibody_strength > 0.5:
            decision_reasoning.append(
                f"중간 항체 강도({antibody_strength:.3f})로 균형잡힌 방어 전략 수행"
            )

        # 전문화 정도 평가
        if specialization_score > 0.8:
            specialization_level = "매우 높음"
        elif specialization_score > 0.6:
            specialization_level = "높음"
        elif specialization_score > 0.4:
            specialization_level = "중간"
        else:
            specialization_level = "낮음"

        return {
            "risk_type": risk_type,
            "specialization_level": specialization_level,
            "specialization_score": specialization_score,
            "decision_reasoning": decision_reasoning,
            "activation_analysis": {
                "level": activation_level,
                "category": (
                    "high"
                    if activation_level > 0.7
                    else "medium" if activation_level > 0.3 else "low"
                ),
            },
            "antibody_analysis": {
                "strength": antibody_strength,
                "effectiveness": (
                    "high"
                    if antibody_strength > 0.8
                    else "medium" if antibody_strength > 0.5 else "low"
                ),
            },
        }

    def _assess_threat_level(self, crisis_level):
        """위기 수준 평가"""
        if crisis_level < self.risk_thresholds["low"]:
            return "stable"
        elif crisis_level < self.risk_thresholds["medium"]:
            return "caution"
        elif crisis_level < self.risk_thresholds["high"]:
            return "alert"
        else:
            return "crisis"

    def generate_analysis_report(self, start_date: str, end_date: str) -> Dict:
        """분석 보고서 생성"""

        # 해당 기간 데이터 필터링
        period_records = []
        for record in self.decision_log:
            record_date = datetime.strptime(record["date"], "%Y-%m-%d")
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")

            if start_dt <= record_date <= end_dt:
                period_records.append(record)

        if not period_records:
            return {"error": f"No data found for period {start_date} to {end_date}"}

        # 통계 계산 (더 민감한 임계값)
        total_days = len(period_records)
        crisis_days = sum(
            1 for r in period_records if r["tcell_analysis"]["crisis_level"] > 0.15
        )
        memory_activations = sum(1 for r in period_records if r["memory_activated"])

        # 지배적 위험 분포
        risk_distribution = {}
        for record in period_records:
            risk = record["tcell_analysis"]["dominant_risk"]
            risk_distribution[risk] = risk_distribution.get(risk, 0) + 1

        # 평균 수익률
        avg_return = np.mean([r["portfolio_return"] for r in period_records])

        # T-cell 위기 감지 상세 분석
        tcell_analysis = self._analyze_tcell_crisis_detection()

        # B-cell 전문가 분석
        bcell_analysis = self._analyze_bcell_expert_responses()

        # 특성 기여도 분석
        feature_attribution = self._analyze_feature_attribution()

        # 시간별 위기 진행 분석
        temporal_analysis = self._analyze_temporal_crisis_patterns()

        report = {
            "period": {"start": start_date, "end": end_date},
            "basic_stats": {
                "total_days": total_days,
                "crisis_days": crisis_days,
                "crisis_ratio": crisis_days / total_days,
                "memory_activations": memory_activations,
                "memory_activation_ratio": memory_activations / total_days,
                "avg_daily_return": avg_return,
            },
            "risk_distribution": risk_distribution,
            "system_efficiency": {
                "crisis_response_rate": crisis_days / total_days,
                "learning_activation_rate": memory_activations / total_days,
                "system_stability": "high" if avg_return > 0 else "normal",
            },
            "tcell_crisis_analysis": tcell_analysis,
            "bcell_expert_analysis": bcell_analysis,
            "feature_attribution": feature_attribution,
            "temporal_analysis": temporal_analysis,
            "explainability_summary": {
                "total_crisis_detections": len(self.crisis_detection_log),
                "total_decisions_logged": len(self.decision_log),
                "analysis_completeness": "comprehensive",
                "xai_features_implemented": [
                    "detailed_crisis_detection_reasoning",
                    "tcell_specific_analysis",
                    "feature_attribution_analysis",
                    "temporal_pattern_tracking",
                    "bcell_expert_reasoning",
                    "decision_explanation_logging",
                ],
            },
        }

        return report

    def _analyze_tcell_crisis_detection(self):
        """T-cell 위기 감지 상세 분석"""
        if not self.crisis_detection_log:
            return {"total_detections": 0, "tcell_details": {}}

        tcell_details = {}
        total_detections = 0
        feature_contributions = {}

        for crisis_log in self.crisis_detection_log:
            tcell_id = crisis_log["tcell_id"]
            crisis_info = crisis_log["crisis_info"]

            if tcell_id not in tcell_details:
                tcell_details[tcell_id] = {
                    "detections": 0,
                    "avg_activation": 0.0,
                    "crisis_types": {},
                    "indicators": {},
                    "reasoning_patterns": [],
                    "feature_contributions": {},
                }

            tcell_details[tcell_id]["detections"] += 1
            tcell_details[tcell_id]["avg_activation"] += crisis_info["activation_level"]

            # 위기 유형 분석
            crisis_level = crisis_info["crisis_level_classification"]
            tcell_details[tcell_id]["crisis_types"][crisis_level] = (
                tcell_details[tcell_id]["crisis_types"].get(crisis_level, 0) + 1
            )

            # 지표 분석
            for indicator in crisis_info["crisis_indicators"]:
                indicator_type = indicator["type"]
                tcell_details[tcell_id]["indicators"][indicator_type] = (
                    tcell_details[tcell_id]["indicators"].get(indicator_type, 0) + 1
                )

            # 특성 기여도 분석
            for feature, contribution in crisis_info["feature_contributions"].items():
                if feature not in tcell_details[tcell_id]["feature_contributions"]:
                    tcell_details[tcell_id]["feature_contributions"][feature] = []
                tcell_details[tcell_id]["feature_contributions"][feature].append(
                    contribution
                )

                # 전체 특성 기여도 누적
                if feature not in feature_contributions:
                    feature_contributions[feature] = []
                feature_contributions[feature].append(contribution)

            # 의사결정 근거 패턴 분석 (최대 3개만 저장)
            if len(tcell_details[tcell_id]["reasoning_patterns"]) < 3:
                tcell_details[tcell_id]["reasoning_patterns"].extend(
                    crisis_info["decision_reasoning"]
                )

            total_detections += 1

        # 평균 계산
        for tcell_id in tcell_details:
            if tcell_details[tcell_id]["detections"] > 0:
                tcell_details[tcell_id]["avg_activation"] /= tcell_details[tcell_id][
                    "detections"
                ]

                # 특성 기여도 평균 계산
                for feature in tcell_details[tcell_id]["feature_contributions"]:
                    contributions = tcell_details[tcell_id]["feature_contributions"][
                        feature
                    ]
                    tcell_details[tcell_id]["feature_contributions"][feature] = {
                        "avg": sum(contributions) / len(contributions),
                        "max": max(contributions),
                        "min": min(contributions),
                    }

        # 전체 특성 기여도 분석
        global_feature_analysis = {}
        for feature, contributions in feature_contributions.items():
            global_feature_analysis[feature] = {
                "avg_contribution": sum(contributions) / len(contributions),
                "max_contribution": max(contributions),
                "detection_frequency": len(contributions),
            }

        return {
            "total_detections": total_detections,
            "active_tcells": len(tcell_details),
            "tcell_details": tcell_details,
            "global_feature_analysis": global_feature_analysis,
        }

    def _analyze_bcell_expert_responses(self):
        """B-cell 전문가 대응 분석"""
        if not self.decision_log:
            return {"total_experts": 0, "expert_details": {}}

        expert_analysis = {}
        total_activations = 0

        for record in self.decision_log:
            for bcell_decision in record.get("bcell_decisions", []):
                if "specialization_analysis" in bcell_decision:
                    analysis = bcell_decision["specialization_analysis"]
                    risk_type = analysis["risk_type"]

                    if risk_type not in expert_analysis:
                        expert_analysis[risk_type] = {
                            "activations": 0,
                            "avg_activation_level": 0.0,
                            "avg_antibody_strength": 0.0,
                            "avg_specialization_score": 0.0,
                            "decision_patterns": {},
                            "performance_metrics": {
                                "positive_outcomes": 0,
                                "negative_outcomes": 0,
                                "avg_return_when_active": 0.0,
                                "returns": [],
                            },
                        }

                    expert_analysis[risk_type]["activations"] += 1
                    expert_analysis[risk_type]["avg_activation_level"] += analysis[
                        "activation_analysis"
                    ]["level"]
                    expert_analysis[risk_type]["avg_antibody_strength"] += analysis[
                        "antibody_analysis"
                    ]["strength"]
                    expert_analysis[risk_type]["avg_specialization_score"] += analysis[
                        "specialization_score"
                    ]

                    # 의사결정 패턴 분석
                    for reasoning in analysis["decision_reasoning"]:
                        pattern_key = (
                            reasoning.split("로 인한")[0]
                            if "로 인한" in reasoning
                            else reasoning[:50]
                        )
                        expert_analysis[risk_type]["decision_patterns"][pattern_key] = (
                            expert_analysis[risk_type]["decision_patterns"].get(
                                pattern_key, 0
                            )
                            + 1
                        )

                    # 성과 분석
                    portfolio_return = record.get("portfolio_return", 0.0)
                    expert_analysis[risk_type]["performance_metrics"]["returns"].append(
                        portfolio_return
                    )

                    if portfolio_return > 0:
                        expert_analysis[risk_type]["performance_metrics"][
                            "positive_outcomes"
                        ] += 1
                    else:
                        expert_analysis[risk_type]["performance_metrics"][
                            "negative_outcomes"
                        ] += 1

                    total_activations += 1

        # 평균 계산
        for risk_type in expert_analysis:
            analysis = expert_analysis[risk_type]
            if analysis["activations"] > 0:
                analysis["avg_activation_level"] /= analysis["activations"]
                analysis["avg_antibody_strength"] /= analysis["activations"]
                analysis["avg_specialization_score"] /= analysis["activations"]

                # 성과 메트릭 계산
                if analysis["performance_metrics"]["returns"]:
                    analysis["performance_metrics"]["avg_return_when_active"] = sum(
                        analysis["performance_metrics"]["returns"]
                    ) / len(analysis["performance_metrics"]["returns"])

                    total_outcomes = (
                        analysis["performance_metrics"]["positive_outcomes"]
                        + analysis["performance_metrics"]["negative_outcomes"]
                    )
                    analysis["performance_metrics"]["success_rate"] = (
                        (
                            analysis["performance_metrics"]["positive_outcomes"]
                            / total_outcomes
                        )
                        * 100
                        if total_outcomes > 0
                        else 0
                    )

        return {
            "total_activations": total_activations,
            "active_experts": len(expert_analysis),
            "expert_details": expert_analysis,
        }

    def _analyze_feature_attribution(self):
        """특성 기여도 분석"""
        if not self.crisis_detection_log:
            return {"total_features": 0, "feature_importance": {}}

        feature_contributions = {}
        feature_combinations = {}

        for crisis_log in self.crisis_detection_log:
            crisis_info = crisis_log["crisis_info"]

            # 개별 특성 기여도 분석
            for feature, contribution in crisis_info["feature_contributions"].items():
                if feature not in feature_contributions:
                    feature_contributions[feature] = []
                feature_contributions[feature].append(contribution)

            # 특성 조합 분석
            active_features = [
                f for f, c in crisis_info["feature_contributions"].items() if c > 0.1
            ]
            if len(active_features) > 1:
                combination = tuple(sorted(active_features))
                feature_combinations[combination] = (
                    feature_combinations.get(combination, 0) + 1
                )

        # 특성 중요도 계산
        feature_importance = {}
        for feature, contributions in feature_contributions.items():
            feature_importance[feature] = {
                "avg_contribution": sum(contributions) / len(contributions),
                "max_contribution": max(contributions),
                "min_contribution": min(contributions),
                "detection_frequency": len(contributions),
                "importance_level": self._classify_importance(
                    sum(contributions) / len(contributions)
                ),
            }

        # 상위 특성 조합
        top_combinations = sorted(
            feature_combinations.items(), key=lambda x: x[1], reverse=True
        )[:5]

        return {
            "total_features": len(feature_importance),
            "feature_importance": feature_importance,
            "top_feature_combinations": [
                {
                    "features": list(combo),
                    "frequency": freq,
                    "percentage": (freq / len(self.crisis_detection_log)) * 100,
                }
                for combo, freq in top_combinations
            ],
        }

    def _analyze_temporal_crisis_patterns(self):
        """시간별 위기 진행 패턴 분석"""
        if not self.crisis_detection_log:
            return {"total_events": 0, "patterns": {}}

        time_sorted_crises = sorted(
            self.crisis_detection_log, key=lambda x: x["timestamp"]
        )

        # 위기 클러스터 분석
        crisis_clusters = []
        current_cluster = []

        for crisis_log in time_sorted_crises:
            crisis_time = datetime.fromisoformat(crisis_log["timestamp"])

            if current_cluster:
                last_crisis_time = datetime.fromisoformat(
                    current_cluster[-1]["timestamp"]
                )
                time_diff = (crisis_time - last_crisis_time).total_seconds() / 3600

                if time_diff <= 6:  # 6시간 이내
                    current_cluster.append(crisis_log)
                else:
                    if len(current_cluster) > 1:
                        crisis_clusters.append(current_cluster)
                    current_cluster = [crisis_log]
            else:
                current_cluster = [crisis_log]

        if len(current_cluster) > 1:
            crisis_clusters.append(current_cluster)

        # 에스컬레이션 패턴 분석
        escalation_patterns = []
        for i in range(len(time_sorted_crises) - 1):
            current_activation = time_sorted_crises[i]["crisis_info"][
                "activation_level"
            ]
            next_activation = time_sorted_crises[i + 1]["crisis_info"][
                "activation_level"
            ]

            time_diff = (
                datetime.fromisoformat(time_sorted_crises[i + 1]["timestamp"])
                - datetime.fromisoformat(time_sorted_crises[i]["timestamp"])
            ).total_seconds() / 3600

            if time_diff <= 6:
                change = next_activation - current_activation
                escalation_patterns.append(
                    {
                        "time_diff": time_diff,
                        "activation_change": change,
                        "pattern": (
                            "escalation"
                            if change > 0.1
                            else "de-escalation" if change < -0.1 else "stable"
                        ),
                    }
                )

        # 시간대별 분석
        hourly_patterns = {}
        for crisis_log in time_sorted_crises:
            crisis_time = datetime.fromisoformat(crisis_log["timestamp"])
            hour = crisis_time.hour
            hourly_patterns[hour] = hourly_patterns.get(hour, 0) + 1

        return {
            "total_events": len(time_sorted_crises),
            "crisis_clusters": {
                "total_clusters": len(crisis_clusters),
                "cluster_details": [
                    {
                        "duration_hours": (
                            datetime.fromisoformat(cluster[-1]["timestamp"])
                            - datetime.fromisoformat(cluster[0]["timestamp"])
                        ).total_seconds()
                        / 3600,
                        "events_count": len(cluster),
                        "max_activation": max(
                            c["crisis_info"]["activation_level"] for c in cluster
                        ),
                        "avg_activation": sum(
                            c["crisis_info"]["activation_level"] for c in cluster
                        )
                        / len(cluster),
                    }
                    for cluster in crisis_clusters
                ],
            },
            "escalation_patterns": {
                "total_patterns": len(escalation_patterns),
                "pattern_distribution": {
                    pattern: sum(
                        1 for p in escalation_patterns if p["pattern"] == pattern
                    )
                    for pattern in ["escalation", "de-escalation", "stable"]
                },
            },
            "hourly_distribution": hourly_patterns,
        }

    def _classify_importance(self, avg_contribution):
        """중요도 분류"""
        if avg_contribution > 0.3:
            return "very_high"
        elif avg_contribution > 0.2:
            return "high"
        elif avg_contribution > 0.1:
            return "medium"
        else:
            return "low"

    def save_analysis_to_file(
        self,
        start_date: str,
        end_date: str,
        filename: str = None,
        output_dir: str = None,
    ):
        """분석 결과를 파일로 저장"""

        if output_dir is None:
            output_dir = self.output_dir  # 전역 output_dir 사용

        if filename is None:
            filename = f"decision_analysis_{start_date}_{end_date}"

        # JSON 보고서 생성
        report = self.generate_analysis_report(start_date, end_date)

        # JSON 파일 저장
        json_path = os.path.join(output_dir, f"{filename}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        # Markdown 보고서 생성
        md_content = self._generate_markdown_report(report)
        md_path = os.path.join(output_dir, f"{filename}.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        return json_path, md_path

    def _generate_markdown_report(self, report: Dict) -> str:
        """마크다운 형식 보고서 생성"""

        if "error" in report:
            return f"# 분석 오류\n\n{report['error']}"

        period = report["period"]
        stats = report["basic_stats"]
        risk_dist = report["risk_distribution"]
        efficiency = report["system_efficiency"]

        md_content = f"""# BIPD 시스템 분석 보고서

## 분석 기간
- 시작일: {period['start']}
- 종료일: {period['end']}

## 기본 통계
- 총 거래일: {stats['total_days']}일
- 위기 감지일: {stats['crisis_days']}일 ({stats['crisis_ratio']:.1%})
- 기억 세포 활성화: {stats['memory_activations']}일 ({stats['memory_activation_ratio']:.1%})
- 평균 일수익률: {stats['avg_daily_return']:+.3%}

## 위험 유형별 분포
"""

        for risk, count in sorted(risk_dist.items(), key=lambda x: x[1], reverse=True):
            percentage = count / stats["total_days"] * 100
            md_content += f"- {risk}: {count}일 ({percentage:.1f}%)\n"

        md_content += f"""
## 시스템 효율성
- 위기 대응률: {efficiency['crisis_response_rate']:.1%}
- 학습 활성화율: {efficiency['learning_activation_rate']:.1%}
- 시스템 안정성: {efficiency['system_stability']}

## T-cell 상세 위기 감지 분석"""

        # T-cell 위기 감지 상세 분석
        if self.crisis_detection_log:
            crisis_by_tcell = {}
            total_crisis_detections = 0

            for crisis_log in self.crisis_detection_log:
                tcell_id = crisis_log["tcell_id"]
                crisis_info = crisis_log["crisis_info"]

                if tcell_id not in crisis_by_tcell:
                    crisis_by_tcell[tcell_id] = {
                        "detections": 0,
                        "avg_activation": 0.0,
                        "crisis_types": {},
                        "main_indicators": {},
                    }

                crisis_by_tcell[tcell_id]["detections"] += 1
                crisis_by_tcell[tcell_id]["avg_activation"] += crisis_info[
                    "activation_level"
                ]

                # 위기 유형 분석
                crisis_level = crisis_info["crisis_level_classification"]
                crisis_by_tcell[tcell_id]["crisis_types"][crisis_level] = (
                    crisis_by_tcell[tcell_id]["crisis_types"].get(crisis_level, 0) + 1
                )

                # 주요 지표 분석
                for indicator in crisis_info["crisis_indicators"]:
                    indicator_type = indicator["type"]
                    crisis_by_tcell[tcell_id]["main_indicators"][indicator_type] = (
                        crisis_by_tcell[tcell_id]["main_indicators"].get(
                            indicator_type, 0
                        )
                        + 1
                    )

                total_crisis_detections += 1

            # 평균 활성화 레벨 계산
            for tcell_id in crisis_by_tcell:
                if crisis_by_tcell[tcell_id]["detections"] > 0:
                    crisis_by_tcell[tcell_id]["avg_activation"] /= crisis_by_tcell[
                        tcell_id
                    ]["detections"]

            if total_crisis_detections > 0:
                md_content += f"""
### 위기 감지 통계
- 총 위기 감지 건수: {total_crisis_detections}건
- 참여 T-cell 수: {len(crisis_by_tcell)}개

### T-cell별 위기 감지 상세 분석
"""

                for tcell_id, tcell_data in sorted(
                    crisis_by_tcell.items(),
                    key=lambda x: x[1]["detections"],
                    reverse=True,
                ):
                    md_content += f"""
#### T-cell {tcell_id}
- 위기 감지 횟수: {tcell_data['detections']}회
- 평균 활성화 레벨: {tcell_data['avg_activation']:.3f}
- 위기 유형 분포:
"""
                    for crisis_type, count in tcell_data["crisis_types"].items():
                        percentage = (count / tcell_data["detections"]) * 100
                        md_content += (
                            f"  - {crisis_type}: {count}회 ({percentage:.1f}%)\n"
                        )

                    md_content += "- 주요 감지 지표:\n"
                    for indicator, count in sorted(
                        tcell_data["main_indicators"].items(),
                        key=lambda x: x[1],
                        reverse=True,
                    ):
                        percentage = (count / tcell_data["detections"]) * 100
                        md_content += (
                            f"  - {indicator}: {count}회 ({percentage:.1f}%)\n"
                        )
            else:
                md_content += "\n\n분석 기간 중 위기 감지 사례가 없습니다."
        else:
            md_content += "\n\n위기 감지 로그가 없습니다."

        # 특성별 기여도 분석
        if self.crisis_detection_log:
            md_content += """

## 특성별 기여도 분석 (Feature Attribution)

### 글로벌 특성 기여도 분석
"""
            # 전체 특성 기여도 분석
            global_features = {}
            for crisis_log in self.crisis_detection_log:
                crisis_info = crisis_log["crisis_info"]
                for feature, contribution in crisis_info[
                    "feature_contributions"
                ].items():
                    if feature not in global_features:
                        global_features[feature] = []
                    global_features[feature].append(contribution)

            if global_features:
                # 특성별 통계 계산
                feature_stats = {}
                for feature, contributions in global_features.items():
                    feature_stats[feature] = {
                        "avg": sum(contributions) / len(contributions),
                        "max": max(contributions),
                        "min": min(contributions),
                        "count": len(contributions),
                    }

                # 평균 기여도 순으로 정렬
                sorted_features = sorted(
                    feature_stats.items(), key=lambda x: x[1]["avg"], reverse=True
                )

                md_content += "위기 감지에서 각 특성의 기여도를 분석한 결과:\n\n"

                for feature, stats in sorted_features:
                    md_content += f"#### {feature}\n"
                    md_content += f"- 평균 기여도: {stats['avg']:.3f}\n"
                    md_content += f"- 최대 기여도: {stats['max']:.3f}\n"
                    md_content += f"- 최소 기여도: {stats['min']:.3f}\n"
                    md_content += f"- 기여 횟수: {stats['count']}회\n"

                    # 기여도 수준 분류
                    if stats["avg"] > 0.3:
                        importance = "매우 높음"
                    elif stats["avg"] > 0.2:
                        importance = "높음"
                    elif stats["avg"] > 0.1:
                        importance = "중간"
                    else:
                        importance = "낮음"

                    md_content += f"- 중요도 평가: {importance}\n\n"

                # 특성간 상관관계 분석
                md_content += """
### 특성간 상호작용 분석
"""

                # 자주 함께 나타나는 특성 조합 분석
                feature_combinations = {}
                for crisis_log in self.crisis_detection_log:
                    crisis_info = crisis_log["crisis_info"]
                    active_features = [
                        f
                        for f, c in crisis_info["feature_contributions"].items()
                        if c > 0.1
                    ]

                    if len(active_features) > 1:
                        combination = tuple(sorted(active_features))
                        feature_combinations[combination] = (
                            feature_combinations.get(combination, 0) + 1
                        )

                if feature_combinations:
                    top_combinations = sorted(
                        feature_combinations.items(), key=lambda x: x[1], reverse=True
                    )[:5]

                    md_content += "자주 함께 나타나는 특성 조합:\n\n"
                    for combination, count in top_combinations:
                        md_content += f"- {' + '.join(combination)}: {count}회\n"

                # 임계값 분석
                md_content += """

### 위기 감지 임계값 분석
"""

                # 각 특성별 위기 감지 임계값 패턴 분석
                threshold_analysis = {}
                for crisis_log in self.crisis_detection_log:
                    crisis_info = crisis_log["crisis_info"]
                    for indicator in crisis_info["crisis_indicators"]:
                        indicator_type = indicator["type"]
                        if indicator_type not in threshold_analysis:
                            threshold_analysis[indicator_type] = {
                                "values": [],
                                "thresholds": [],
                                "contributions": [],
                            }

                        threshold_analysis[indicator_type]["values"].append(
                            indicator["value"]
                        )
                        threshold_analysis[indicator_type]["thresholds"].append(
                            indicator["threshold"]
                        )
                        threshold_analysis[indicator_type]["contributions"].append(
                            indicator["contribution"]
                        )

                if threshold_analysis:
                    md_content += "각 지표별 임계값 분석:\n\n"
                    for indicator_type, data in threshold_analysis.items():
                        avg_value = sum(data["values"]) / len(data["values"])
                        avg_threshold = sum(data["thresholds"]) / len(
                            data["thresholds"]
                        )
                        avg_contribution = sum(data["contributions"]) / len(
                            data["contributions"]
                        )

                        md_content += f"#### {indicator_type}\n"
                        md_content += f"- 평균 감지값: {avg_value:.3f}\n"
                        md_content += f"- 평균 임계값: {avg_threshold:.3f}\n"
                        md_content += f"- 평균 기여도: {avg_contribution:.3f}\n"
                        md_content += f"- 감지 횟수: {len(data['values'])}회\n\n"

        # 시간별 위기 진행 과정 추적
        if self.crisis_detection_log:
            md_content += """

## 시간별 위기 진행 과정 추적

### 위기 패턴 분석
"""

            # 시간순 정렬된 위기 이벤트 분석
            time_sorted_crises = sorted(
                self.crisis_detection_log, key=lambda x: x["timestamp"]
            )

            if time_sorted_crises:
                # 위기 클러스터 분석 (연속된 위기 이벤트)
                crisis_clusters = []
                current_cluster = []

                for i, crisis_log in enumerate(time_sorted_crises):
                    crisis_time = datetime.fromisoformat(crisis_log["timestamp"])

                    if current_cluster:
                        last_crisis_time = datetime.fromisoformat(
                            current_cluster[-1]["timestamp"]
                        )
                        time_diff = (
                            crisis_time - last_crisis_time
                        ).total_seconds() / 3600  # 시간 단위

                        # 6시간 이내의 위기는 같은 클러스터로 간주
                        if time_diff <= 6:
                            current_cluster.append(crisis_log)
                        else:
                            if len(current_cluster) > 1:
                                crisis_clusters.append(current_cluster)
                            current_cluster = [crisis_log]
                    else:
                        current_cluster = [crisis_log]

                # 마지막 클러스터 추가
                if len(current_cluster) > 1:
                    crisis_clusters.append(current_cluster)

                if crisis_clusters:
                    md_content += f"식별된 위기 클러스터: {len(crisis_clusters)}개\n\n"

                    for i, cluster in enumerate(crisis_clusters, 1):
                        start_time = datetime.fromisoformat(cluster[0]["timestamp"])
                        end_time = datetime.fromisoformat(cluster[-1]["timestamp"])
                        duration = (end_time - start_time).total_seconds() / 3600

                        md_content += f"#### 위기 클러스터 {i}\n"
                        md_content += f"- 기간: {start_time.strftime('%Y-%m-%d %H:%M')} ~ {end_time.strftime('%Y-%m-%d %H:%M')}\n"
                        md_content += f"- 지속 시간: {duration:.1f}시간\n"
                        md_content += f"- 위기 이벤트 수: {len(cluster)}개\n"

                        # 클러스터 내 위기 진행 패턴 분석
                        activation_levels = [
                            c["crisis_info"]["activation_level"] for c in cluster
                        ]
                        avg_activation = sum(activation_levels) / len(activation_levels)
                        max_activation = max(activation_levels)

                        md_content += f"- 평균 활성화 레벨: {avg_activation:.3f}\n"
                        md_content += f"- 최대 활성화 레벨: {max_activation:.3f}\n"

                        # 참여 T-cell 분석
                        involved_tcells = set(c["tcell_id"] for c in cluster)
                        md_content += f"- 참여 T-cell: {len(involved_tcells)}개 ({', '.join(sorted(involved_tcells))})\n"

                        # 주요 위기 지표
                        cluster_indicators = {}
                        for crisis_log in cluster:
                            for indicator in crisis_log["crisis_info"][
                                "crisis_indicators"
                            ]:
                                indicator_type = indicator["type"]
                                cluster_indicators[indicator_type] = (
                                    cluster_indicators.get(indicator_type, 0) + 1
                                )

                        if cluster_indicators:
                            top_indicators = sorted(
                                cluster_indicators.items(),
                                key=lambda x: x[1],
                                reverse=True,
                            )[:3]
                            md_content += f"- 주요 위기 지표: {', '.join([f'{ind}({count})' for ind, count in top_indicators])}\n\n"

                # 위기 에스컬레이션 패턴 분석
                md_content += """
### 위기 에스컬레이션 패턴
"""

                # 연속된 위기 이벤트에서 활성화 레벨 변화 분석
                escalation_patterns = []
                for i in range(len(time_sorted_crises) - 1):
                    current_activation = time_sorted_crises[i]["crisis_info"][
                        "activation_level"
                    ]
                    next_activation = time_sorted_crises[i + 1]["crisis_info"][
                        "activation_level"
                    ]

                    time_diff = (
                        datetime.fromisoformat(time_sorted_crises[i + 1]["timestamp"])
                        - datetime.fromisoformat(time_sorted_crises[i]["timestamp"])
                    ).total_seconds() / 3600

                    if time_diff <= 6:  # 6시간 이내의 연속 이벤트
                        change = next_activation - current_activation
                        escalation_patterns.append(
                            {
                                "time_diff": time_diff,
                                "activation_change": change,
                                "pattern": (
                                    "escalation"
                                    if change > 0.1
                                    else "de-escalation" if change < -0.1 else "stable"
                                ),
                            }
                        )

                if escalation_patterns:
                    pattern_counts = {}
                    for pattern in escalation_patterns:
                        pattern_type = pattern["pattern"]
                        pattern_counts[pattern_type] = (
                            pattern_counts.get(pattern_type, 0) + 1
                        )

                    total_patterns = len(escalation_patterns)
                    md_content += "위기 에스컬레이션 패턴 분석:\n\n"

                    for pattern_type, count in pattern_counts.items():
                        percentage = (count / total_patterns) * 100
                        md_content += (
                            f"- {pattern_type}: {count}회 ({percentage:.1f}%)\n"
                        )

                    # 평균 에스컬레이션 속도
                    escalations = [
                        p for p in escalation_patterns if p["pattern"] == "escalation"
                    ]
                    if escalations:
                        avg_escalation_speed = sum(
                            p["activation_change"] / p["time_diff"] for p in escalations
                        ) / len(escalations)
                        md_content += f"- 평균 에스컬레이션 속도: {avg_escalation_speed:.3f}/시간\n"

                    de_escalations = [
                        p
                        for p in escalation_patterns
                        if p["pattern"] == "de-escalation"
                    ]
                    if de_escalations:
                        avg_de_escalation_speed = abs(
                            sum(
                                p["activation_change"] / p["time_diff"]
                                for p in de_escalations
                            )
                        ) / len(de_escalations)
                        md_content += (
                            f"- 평균 완화 속도: {avg_de_escalation_speed:.3f}/시간\n"
                        )

                # 위기 예측 정확도 분석
                md_content += """

### 위기 예측 및 조기 경고 분석
"""

                # 위기 감지 후 실제 시장 변화 분석 (간단한 패턴)
                prediction_accuracy = []
                for i, crisis_log in enumerate(time_sorted_crises):
                    if i < len(time_sorted_crises) - 1:
                        current_activation = crisis_log["crisis_info"][
                            "activation_level"
                        ]
                        next_activation = time_sorted_crises[i + 1]["crisis_info"][
                            "activation_level"
                        ]

                        # 현재 위기 감지가 다음 위기를 예측했는지 평가
                        prediction_accuracy.append(
                            {
                                "predicted_high": current_activation > 0.5,
                                "actual_high": next_activation > 0.5,
                            }
                        )

                if prediction_accuracy:
                    correct_predictions = sum(
                        1
                        for p in prediction_accuracy
                        if p["predicted_high"] == p["actual_high"]
                    )
                    accuracy = (correct_predictions / len(prediction_accuracy)) * 100

                    md_content += f"위기 예측 정확도: {accuracy:.1f}% ({correct_predictions}/{len(prediction_accuracy)})\n\n"

                # 시간대별 위기 발생 패턴
                md_content += """
### 시간대별 위기 발생 패턴
"""

                hourly_crisis_count = {}
                for crisis_log in time_sorted_crises:
                    crisis_time = datetime.fromisoformat(crisis_log["timestamp"])
                    hour = crisis_time.hour
                    hourly_crisis_count[hour] = hourly_crisis_count.get(hour, 0) + 1

                if hourly_crisis_count:
                    sorted_hours = sorted(
                        hourly_crisis_count.items(), key=lambda x: x[1], reverse=True
                    )
                    md_content += "시간대별 위기 발생 빈도:\n\n"

                    for hour, count in sorted_hours[:5]:  # 상위 5개 시간대
                        percentage = (count / len(time_sorted_crises)) * 100
                        md_content += f"- {hour:02d}시: {count}회 ({percentage:.1f}%)\n"

        # B-cell 전문가 대응 분석
        if self.decision_log:
            md_content += """

## B-cell 전문가 대응 분석

### 전문가별 활성화 패턴
"""

            # B-cell 활성화 통계 수집
            bcell_stats = {}
            total_activations = 0

            for record in self.decision_log:
                for bcell_decision in record.get("bcell_decisions", []):
                    if "specialization_analysis" in bcell_decision:
                        analysis = bcell_decision["specialization_analysis"]
                        risk_type = analysis["risk_type"]

                        if risk_type not in bcell_stats:
                            bcell_stats[risk_type] = {
                                "activations": 0,
                                "avg_activation_level": 0.0,
                                "avg_antibody_strength": 0.0,
                                "avg_specialization_score": 0.0,
                                "decision_patterns": {},
                                "activation_levels": [],
                                "antibody_strengths": [],
                                "specialization_scores": [],
                            }

                        bcell_stats[risk_type]["activations"] += 1
                        bcell_stats[risk_type]["activation_levels"].append(
                            analysis["activation_analysis"]["level"]
                        )
                        bcell_stats[risk_type]["antibody_strengths"].append(
                            analysis["antibody_analysis"]["strength"]
                        )
                        bcell_stats[risk_type]["specialization_scores"].append(
                            analysis["specialization_score"]
                        )

                        # 의사결정 패턴 분석
                        for reasoning in analysis["decision_reasoning"]:
                            pattern_key = (
                                reasoning.split("로 인한")[0]
                                if "로 인한" in reasoning
                                else reasoning[:50]
                            )
                            bcell_stats[risk_type]["decision_patterns"][pattern_key] = (
                                bcell_stats[risk_type]["decision_patterns"].get(
                                    pattern_key, 0
                                )
                                + 1
                            )

                        total_activations += 1

            # 평균 계산
            for risk_type in bcell_stats:
                stats = bcell_stats[risk_type]
                if stats["activations"] > 0:
                    stats["avg_activation_level"] = sum(
                        stats["activation_levels"]
                    ) / len(stats["activation_levels"])
                    stats["avg_antibody_strength"] = sum(
                        stats["antibody_strengths"]
                    ) / len(stats["antibody_strengths"])
                    stats["avg_specialization_score"] = sum(
                        stats["specialization_scores"]
                    ) / len(stats["specialization_scores"])

            if bcell_stats:
                # 전문가별 활성화 통계
                sorted_bcells = sorted(
                    bcell_stats.items(), key=lambda x: x[1]["activations"], reverse=True
                )

                for risk_type, stats in sorted_bcells:
                    activation_percentage = (
                        stats["activations"] / total_activations
                    ) * 100

                    md_content += f"""
#### {risk_type} 전문가
- 활성화 횟수: {stats['activations']}회 ({activation_percentage:.1f}%)
- 평균 활성화 레벨: {stats['avg_activation_level']:.3f}
- 평균 항체 강도: {stats['avg_antibody_strength']:.3f}
- 평균 전문화 점수: {stats['avg_specialization_score']:.3f}
"""

                    # 성과 평가
                    if stats["avg_specialization_score"] > 0.8:
                        performance = "매우 우수"
                    elif stats["avg_specialization_score"] > 0.6:
                        performance = "우수"
                    elif stats["avg_specialization_score"] > 0.4:
                        performance = "보통"
                    else:
                        performance = "개선 필요"

                    md_content += f"- 성과 평가: {performance}\n"

                    # 주요 의사결정 패턴
                    if stats["decision_patterns"]:
                        top_patterns = sorted(
                            stats["decision_patterns"].items(),
                            key=lambda x: x[1],
                            reverse=True,
                        )[:3]
                        md_content += "- 주요 의사결정 패턴:\n"
                        for pattern, count in top_patterns:
                            pattern_percentage = (count / stats["activations"]) * 100
                            md_content += f"  - {pattern}: {count}회 ({pattern_percentage:.1f}%)\n"

                    md_content += "\n"

                # 전문가간 협력 분석
                md_content += """
### 전문가간 협력 패턴
"""

                # 동시 활성화 패턴 분석
                collaboration_patterns = {}
                for record in self.decision_log:
                    active_experts = []
                    for bcell_decision in record.get("bcell_decisions", []):
                        if "specialization_analysis" in bcell_decision:
                            risk_type = bcell_decision["specialization_analysis"][
                                "risk_type"
                            ]
                            active_experts.append(risk_type)

                    if len(active_experts) > 1:
                        collaboration_key = " + ".join(sorted(active_experts))
                        collaboration_patterns[collaboration_key] = (
                            collaboration_patterns.get(collaboration_key, 0) + 1
                        )

                if collaboration_patterns:
                    top_collaborations = sorted(
                        collaboration_patterns.items(), key=lambda x: x[1], reverse=True
                    )[:5]
                    md_content += "자주 함께 활성화되는 전문가 조합:\n\n"

                    for collaboration, count in top_collaborations:
                        percentage = (count / len(self.decision_log)) * 100
                        md_content += (
                            f"- {collaboration}: {count}회 ({percentage:.1f}%)\n"
                        )

                # 전문가 효율성 분석
                md_content += """

### 전문가 효율성 분석
"""

                # 각 전문가별 효율성 지표
                efficiency_analysis = {}
                for record in self.decision_log:
                    portfolio_return = record.get("portfolio_return", 0.0)

                    for bcell_decision in record.get("bcell_decisions", []):
                        if "specialization_analysis" in bcell_decision:
                            risk_type = bcell_decision["specialization_analysis"][
                                "risk_type"
                            ]
                            contribution = bcell_decision.get(
                                "strategy_contribution", 0.0
                            )

                            if risk_type not in efficiency_analysis:
                                efficiency_analysis[risk_type] = {
                                    "total_contribution": 0.0,
                                    "positive_outcomes": 0,
                                    "negative_outcomes": 0,
                                    "avg_return_when_active": 0.0,
                                    "returns_when_active": [],
                                }

                            efficiency_analysis[risk_type][
                                "total_contribution"
                            ] += contribution
                            efficiency_analysis[risk_type][
                                "returns_when_active"
                            ].append(portfolio_return)

                            if portfolio_return > 0:
                                efficiency_analysis[risk_type]["positive_outcomes"] += 1
                            else:
                                efficiency_analysis[risk_type]["negative_outcomes"] += 1

                for risk_type, analysis in efficiency_analysis.items():
                    if analysis["returns_when_active"]:
                        analysis["avg_return_when_active"] = sum(
                            analysis["returns_when_active"]
                        ) / len(analysis["returns_when_active"])
                        total_outcomes = (
                            analysis["positive_outcomes"]
                            + analysis["negative_outcomes"]
                        )
                        success_rate = (
                            (analysis["positive_outcomes"] / total_outcomes) * 100
                            if total_outcomes > 0
                            else 0
                        )

                        md_content += f"""
#### {risk_type} 전문가 효율성
- 평균 기여도: {analysis['total_contribution'] / len(analysis['returns_when_active']):.3f}
- 활성화 시 평균 수익률: {analysis['avg_return_when_active']:+.3%}
- 성공률: {success_rate:.1f}% ({analysis['positive_outcomes']}/{total_outcomes})
"""

        return md_content


class ImmuneCell:
    """면역 세포 기본 클래스"""

    def __init__(self, cell_id, activation_threshold=0.5):
        self.cell_id = cell_id
        self.activation_threshold = activation_threshold
        self.activation_level = 0.0
        self.memory_strength = 0.0


class TCell(ImmuneCell):
    """T-세포: 위험 탐지"""

    def __init__(self, cell_id, sensitivity=0.1, random_state=None):
        super().__init__(cell_id)
        self.sensitivity = sensitivity
        self.detector = IsolationForest(
            contamination=sensitivity, random_state=random_state
        )
        self.is_trained = False
        self.training_data = deque(maxlen=200)  # 훈련 데이터 저장
        self.historical_scores = deque(maxlen=100)
        self.market_state_history = deque(maxlen=50)  # 시장 상태 히스토리

    def detect_anomaly(self, market_features):
        """시장 이상 상황 탐지"""
        # 입력 특성 크기 확인 및 조정
        if len(market_features.shape) == 1:
            market_features = market_features.reshape(1, -1)

        # 훈련 데이터 축적
        self.training_data.append(market_features[0])

        if not self.is_trained:
            if len(self.training_data) >= 200:  # 충분한 데이터가 쌓인 후 훈련
                training_matrix = np.array(list(self.training_data))
                self.detector.fit(training_matrix)
                self.is_trained = True
                self.expected_features = training_matrix.shape[1]
                print(
                    f"[정보] T-cell {self.cell_id} 훈련 완료 (데이터: {len(self.training_data)}개)"
                )
            return 0.0

        # 특성 크기 확인
        if market_features.shape[1] != self.expected_features:
            print(
                f"[경고] T-cell 특성 크기 불일치: 기대 {self.expected_features}, 실제 {market_features.shape[1]}"
            )
            min_features = min(self.expected_features, market_features.shape[1])
            market_features = market_features[:, :min_features]

            if market_features.shape[1] < self.expected_features:
                padding = np.zeros(
                    (
                        market_features.shape[0],
                        self.expected_features - market_features.shape[1],
                    )
                )
                market_features = np.hstack([market_features, padding])

        # 이상 점수 계산
        anomaly_scores = self.detector.decision_function(market_features)
        current_score = np.mean(anomaly_scores)

        # 시장 상태 분석
        market_state = self._analyze_market_state(market_features[0])
        self.market_state_history.append(market_state)

        # 히스토리 업데이트
        self.historical_scores.append(current_score)

        # 위기 감지 로직 및 상세 분석
        crisis_detection = self._detailed_crisis_analysis(current_score, market_state)
        self.activation_level = crisis_detection["activation_level"]

        # 위기 감지 로그 저장 (활성화 레벨이 0.15 이상일 때)
        if self.activation_level > 0.15:
            self.last_crisis_detection = crisis_detection

        return self.activation_level

    def _detailed_crisis_analysis(self, current_score, market_state):
        """위기 감지 상세 분석"""
        crisis_info = {
            "tcell_id": self.cell_id,
            "timestamp": datetime.now().isoformat(),
            "raw_anomaly_score": current_score,
            "market_state": market_state,
            "activation_level": 0.0,
            "crisis_indicators": [],
            "feature_contributions": {},
            "decision_reasoning": [],
        }

        if len(self.historical_scores) >= 10:
            historical_mean = np.mean(self.historical_scores)
            historical_std = np.std(self.historical_scores)

            # Z-score 기반 이상 탐지
            z_score = (current_score - historical_mean) / (historical_std + 1e-8)

            # 시장 상태 기반 조정
            market_volatility = market_state["volatility"]
            market_stress = market_state["stress"]
            market_correlation = market_state["correlation"]

            # 기본 활성화 레벨 계산 및 근거 기록
            base_activation = 0.0
            if z_score < -1.5:
                base_activation = 0.8
                crisis_info["crisis_indicators"].append(
                    {
                        "type": "extreme_anomaly",
                        "value": z_score,
                        "threshold": -1.5,
                        "contribution": 0.8,
                        "description": f"매우 이상한 이상 점수 (Z-score: {z_score:.3f})",
                    }
                )
                crisis_info["decision_reasoning"].append(
                    f"이상 점수가 과거 평균보다 {abs(z_score):.1f} 표준편차 낮음 (매우 이상)"
                )
            elif z_score < -1.0:
                base_activation = 0.6
                crisis_info["crisis_indicators"].append(
                    {
                        "type": "high_anomaly",
                        "value": z_score,
                        "threshold": -1.0,
                        "contribution": 0.6,
                        "description": f"상당히 이상한 이상 점수 (Z-score: {z_score:.3f})",
                    }
                )
                crisis_info["decision_reasoning"].append(
                    f"이상 점수가 과거 평균보다 {abs(z_score):.1f} 표준편차 낮음 (상당히 이상)"
                )
            elif z_score < -0.5:
                base_activation = 0.4
                crisis_info["crisis_indicators"].append(
                    {
                        "type": "moderate_anomaly",
                        "value": z_score,
                        "threshold": -0.5,
                        "contribution": 0.4,
                        "description": f"약간 이상한 이상 점수 (Z-score: {z_score:.3f})",
                    }
                )
                crisis_info["decision_reasoning"].append(
                    f"이상 점수가 과거 평균보다 {abs(z_score):.1f} 표준편차 낮음 (약간 이상)"
                )
            elif z_score < 0.0:
                base_activation = 0.2
                crisis_info["crisis_indicators"].append(
                    {
                        "type": "mild_anomaly",
                        "value": z_score,
                        "threshold": 0.0,
                        "contribution": 0.2,
                        "description": f"주의 수준 이상 점수 (Z-score: {z_score:.3f})",
                    }
                )
                crisis_info["decision_reasoning"].append(
                    f"이상 점수가 과거 평균보다 낮음 (주의 필요)"
                )

            # 시장 상태 기반 조정 및 근거 기록
            volatility_boost = 0.0
            if market_volatility > 0.3:
                volatility_boost = 0.2
                base_activation += volatility_boost
                crisis_info["crisis_indicators"].append(
                    {
                        "type": "high_volatility",
                        "value": market_volatility,
                        "threshold": 0.3,
                        "contribution": volatility_boost,
                        "description": f"높은 시장 변동성 ({market_volatility:.3f})",
                    }
                )
                crisis_info["decision_reasoning"].append(
                    f"시장 변동성이 임계값 0.3을 초과함 ({market_volatility:.3f})"
                )

            stress_boost = 0.0
            if market_stress > 0.5:
                stress_boost = 0.15
                base_activation += stress_boost
                crisis_info["crisis_indicators"].append(
                    {
                        "type": "high_stress",
                        "value": market_stress,
                        "threshold": 0.5,
                        "contribution": stress_boost,
                        "description": f"높은 시장 스트레스 ({market_stress:.3f})",
                    }
                )
                crisis_info["decision_reasoning"].append(
                    f"시장 스트레스 지수가 임계값 0.5를 초과함 ({market_stress:.3f})"
                )

            # 상관관계 위험 분석
            corr_boost = 0.0
            if market_correlation > 0.8:
                corr_boost = 0.1
                base_activation += corr_boost
                crisis_info["crisis_indicators"].append(
                    {
                        "type": "high_correlation",
                        "value": market_correlation,
                        "threshold": 0.8,
                        "contribution": corr_boost,
                        "description": f"높은 시장 상관관계 ({market_correlation:.3f})",
                    }
                )
                crisis_info["decision_reasoning"].append(
                    f"시장 상관관계가 과도하게 높음 ({market_correlation:.3f}) - 시스템적 위험"
                )

            # 최근 시장 상태 변화 고려
            trend_boost = 0.0
            if len(self.market_state_history) >= 5:
                recent_volatility_change = np.mean(
                    [s["volatility"] for s in list(self.market_state_history)[-5:]]
                )
                if recent_volatility_change > 0.4:
                    trend_boost = 0.1
                    base_activation += trend_boost
                    crisis_info["crisis_indicators"].append(
                        {
                            "type": "volatility_trend",
                            "value": recent_volatility_change,
                            "threshold": 0.4,
                            "contribution": trend_boost,
                            "description": f"지속적인 높은 변동성 ({recent_volatility_change:.3f})",
                        }
                    )
                    crisis_info["decision_reasoning"].append(
                        f"최근 5일 평균 변동성이 지속적으로 높음 ({recent_volatility_change:.3f})"
                    )

            # 특성별 기여도 분석
            crisis_info["feature_contributions"] = {
                "z_score_base": base_activation
                - volatility_boost
                - stress_boost
                - trend_boost
                - corr_boost,
                "volatility_boost": volatility_boost,
                "stress_boost": stress_boost,
                "correlation_boost": corr_boost,
                "trend_boost": trend_boost,
                "total_score": base_activation,
            }

            crisis_info["activation_level"] = np.clip(base_activation, 0.0, 1.0)

            # 위기 수준 분류
            if crisis_info["activation_level"] > 0.7:
                crisis_info["crisis_level"] = "severe"
            elif crisis_info["activation_level"] > 0.5:
                crisis_info["crisis_level"] = "high"
            elif crisis_info["activation_level"] > 0.3:
                crisis_info["crisis_level"] = "moderate"
            elif crisis_info["activation_level"] > 0.15:
                crisis_info["crisis_level"] = "mild"
            else:
                crisis_info["crisis_level"] = "normal"

        else:
            # 초기 학습 기간
            raw_score = max(0, min(1, (1 - current_score) / 1.5))
            crisis_info["activation_level"] = raw_score * 0.5
            crisis_info["crisis_level"] = "learning"
            crisis_info["decision_reasoning"].append("초기 학습 기간 - 보수적 설정")

        return crisis_info

    def _analyze_market_state(self, features):
        """시장 상태 분석"""
        # 특성 기반 시장 상태 분석
        volatility = features[0] if len(features) > 0 else 0.0
        correlation = features[1] if len(features) > 1 else 0.0
        returns = features[2] if len(features) > 2 else 0.0

        # 스트레스 지수 계산
        stress_indicators = []
        if len(features) > 4:
            stress_indicators.append(abs(features[4]))  # 왜도
        if len(features) > 5:
            stress_indicators.append(abs(features[5]))  # 첨도
        if len(features) > 6:
            stress_indicators.append(features[6])  # 하락일 비율

        stress_level = np.mean(stress_indicators) if stress_indicators else 0.0

        return {
            "volatility": abs(volatility),
            "correlation": abs(correlation),
            "returns": returns,
            "stress": stress_level,
        }


class StrategyNetwork(nn.Module):
    """전략 생성 신경망"""

    def __init__(self, input_size, n_assets, hidden_size=64):
        super(StrategyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, n_assets)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return F.softmax(x, dim=-1)


class BCell(ImmuneCell):
    """B-세포: 전문화된 대응 전략 생성"""

    def __init__(self, cell_id, risk_type, input_size, n_assets, learning_rate=0.001):
        super().__init__(cell_id)
        self.risk_type = risk_type
        self.n_assets = n_assets

        # 신경망 초기화
        self.strategy_network = StrategyNetwork(input_size, n_assets)
        self.optimizer = optim.Adam(
            self.strategy_network.parameters(), lr=learning_rate
        )

        # 강화학습 파라미터
        self.experience_buffer = []
        self.episode_buffer = []
        self.antibody_strength = 0.1
        self.epsilon = 0.3
        self.epsilon_decay = 0.995
        self.min_epsilon = 0.05

        # 학습 설정
        self.batch_size = 32
        self.update_frequency = 10
        self.experience_count = 0

        # 전문화 관련 속성
        self.specialization_buffer = deque(maxlen=1000)
        self.general_buffer = deque(maxlen=500)
        self.specialization_strength = 0.1

        # 전문 분야별 특화 기준
        self.specialization_criteria = self._initialize_specialization_criteria()

        # 적응형 학습률
        self.adaptive_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.8, patience=15, verbose=False
        )

        # 성과 추적
        self.specialist_performance = deque(maxlen=50)
        self.general_performance = deque(maxlen=50)

        # 전문화 가중치
        self.specialization_weights = self._initialize_specialization(
            risk_type, n_assets
        )

    def _initialize_specialization(self, risk_type, n_assets):
        """위험 유형별 초기 특화 설정"""
        weights = torch.ones(n_assets) * 0.1

        if risk_type == "volatility":
            safe_indices = [6, 7, 8] if n_assets >= 9 else [n_assets - 1]
            for idx in safe_indices:
                if idx < n_assets:
                    weights[idx] = 0.3
        elif risk_type == "correlation":
            weights = torch.ones(n_assets) * (0.8 / n_assets)
        elif risk_type == "momentum":
            weights = torch.ones(n_assets) * 0.5
        elif risk_type == "liquidity":
            large_cap_indices = [0, 1, 2, 3] if n_assets >= 4 else list(range(n_assets))
            for idx in large_cap_indices:
                if idx < n_assets:
                    weights[idx] = 0.25

        return weights

    def _initialize_specialization_criteria(self):
        """위험 유형별 전문화 기준 설정"""
        criteria = {
            "volatility": {
                "feature_indices": [0, 5],
                "thresholds": [0.4, 0.3],
                "crisis_range": (0.3, 0.9),
            },
            "correlation": {
                "feature_indices": [1],
                "thresholds": [0.6],
                "crisis_range": (0.4, 1.0),
            },
            "momentum": {
                "feature_indices": [2],
                "thresholds": [0.2],
                "crisis_range": (0.2, 0.8),
            },
            "liquidity": {
                "feature_indices": [6],
                "thresholds": [0.4],
                "crisis_range": (0.3, 0.9),
            },
            "macro": {
                "feature_indices": [3, 4, 7],
                "thresholds": [0.5, 1.0, 0.5],
                "crisis_range": (0.4, 1.0),
            },
        }
        return criteria.get(
            self.risk_type,
            {"feature_indices": [0], "thresholds": [0.5], "crisis_range": (0.3, 0.8)},
        )

    def is_my_specialty_situation(self, market_features, crisis_level):
        """현재 상황이 전문 분야인지 판단"""

        criteria = self.specialization_criteria

        # 위기 수준 확인
        min_crisis, max_crisis = criteria["crisis_range"]
        if not (min_crisis <= crisis_level <= max_crisis):
            return False

        # 시장 특성 확인
        feature_indices = criteria["feature_indices"]
        thresholds = criteria["thresholds"]

        specialty_signals = 0
        for idx, threshold in zip(feature_indices, thresholds):
            if idx < len(market_features):
                if abs(market_features[idx]) >= threshold:
                    specialty_signals += 1

        required_signals = max(1, len(feature_indices) // 2)
        is_specialty = specialty_signals >= required_signals

        confidence_boost = 1.0 + self.specialization_strength * 0.5

        return is_specialty and (
            specialty_signals * confidence_boost >= required_signals
        )

    def produce_antibody(self, market_features, crisis_level, training=True):
        """전략 생성"""

        try:
            features_tensor = torch.FloatTensor(market_features)
            crisis_tensor = torch.FloatTensor([crisis_level])
            combined_input = torch.cat(
                [features_tensor, crisis_tensor, self.specialization_weights]
            )

            with torch.no_grad():
                raw_strategy = self.strategy_network(combined_input.unsqueeze(0))
                strategy_tensor = raw_strategy.squeeze(0)

            # 전문 상황 여부에 따른 조정
            is_specialty = self.is_my_specialty_situation(market_features, crisis_level)

            if is_specialty:
                strategy_tensor = self._apply_specialist_strategy(
                    strategy_tensor, market_features, crisis_level
                )
                confidence_multiplier = 1.0 + self.specialization_strength
            else:
                strategy_tensor = self._apply_conservative_adjustment(strategy_tensor)
                confidence_multiplier = 0.7

            # 탐험/활용
            if training and np.random.random() < self.epsilon:
                exploration_strength = 0.05 if is_specialty else 0.1
                noise = torch.randn_like(strategy_tensor) * exploration_strength
                strategy_tensor = strategy_tensor + noise
                strategy_tensor = F.softmax(strategy_tensor, dim=0)

            # 마지막 행동 저장
            self.last_strategy = strategy_tensor

            # 강도 계산
            base_confidence = 1.0 - float(torch.std(strategy_tensor))
            final_strength = max(0.1, base_confidence * confidence_multiplier)
            self.antibody_strength = final_strength

            return strategy_tensor.numpy(), final_strength

        except Exception as e:
            print(f"[경고] {self.risk_type} B-세포 전략 생성 오류: {e}")
            default_strategy = np.ones(self.n_assets) / self.n_assets
            return default_strategy, 0.1

    def _apply_specialist_strategy(
        self, strategy_tensor, market_features, crisis_level
    ):
        """전문가 전략 적용"""

        specialized_strategy = strategy_tensor.clone()

        if self.risk_type == "volatility" and crisis_level > 0.5:
            safe_indices = [6, 7, 8]
            for idx in safe_indices:
                if idx < len(specialized_strategy):
                    specialized_strategy[idx] *= 1.0 + self.specialization_strength

        elif self.risk_type == "correlation" and market_features[1] > 0.7:
            uniform_weight = torch.ones_like(specialized_strategy) / len(
                specialized_strategy
            )
            blend_ratio = 0.3 + self.specialization_strength * 0.2
            specialized_strategy = (
                1 - blend_ratio
            ) * specialized_strategy + blend_ratio * uniform_weight

        elif self.risk_type == "momentum" and abs(market_features[2]) > 0.3:
            if market_features[2] > 0:
                growth_indices = [0, 1, 4]
                for idx in growth_indices:
                    if idx < len(specialized_strategy):
                        specialized_strategy[idx] *= (
                            1.0 + self.specialization_strength * 0.5
                        )
            else:
                defensive_indices = [6, 7, 8]
                for idx in defensive_indices:
                    if idx < len(specialized_strategy):
                        specialized_strategy[idx] *= (
                            1.0 + self.specialization_strength * 0.8
                        )

        elif self.risk_type == "liquidity" and market_features[6] > 0.5:
            large_cap_indices = [0, 1, 2, 3]
            boost_factor = 1.0 + self.specialization_strength * 0.6
            for idx in large_cap_indices:
                if idx < len(specialized_strategy):
                    specialized_strategy[idx] *= boost_factor

        elif self.risk_type == "macro":
            defensive_indices = [7, 8, 9]
            boost_factor = 1.0 + self.specialization_strength * 0.7
            for idx in defensive_indices:
                if idx < len(specialized_strategy):
                    specialized_strategy[idx] *= boost_factor

        specialized_strategy = F.softmax(specialized_strategy, dim=0)
        return specialized_strategy

    def _apply_conservative_adjustment(self, strategy_tensor):
        """보수적 조정"""

        uniform_weight = torch.ones_like(strategy_tensor) / len(strategy_tensor)
        conservative_blend = 0.3

        conservative_strategy = (
            1 - conservative_blend
        ) * strategy_tensor + conservative_blend * uniform_weight
        return F.softmax(conservative_strategy, dim=0)

    def add_experience(self, market_features, crisis_level, action, reward):
        """경험 저장"""

        experience = {
            "state": market_features.copy(),
            "crisis_level": crisis_level,
            "action": action.copy(),
            "reward": reward,
            "timestamp": datetime.now(),
            "is_specialty": self.is_my_specialty_situation(
                market_features, crisis_level
            ),
        }

        if experience["is_specialty"]:
            self.specialization_buffer.append(experience)
            self.specialist_performance.append(reward)
            self.specialization_strength = min(
                1.0, self.specialization_strength + 0.005
            )
        else:
            self.general_buffer.append(experience)
            self.general_performance.append(reward)

        self.episode_buffer.append(experience)
        self.experience_count += 1

    def learn_from_batch(self):
        """배치 학습"""
        if len(self.episode_buffer) < self.batch_size:
            return

        try:
            batch_size = min(self.batch_size, len(self.episode_buffer))
            batch = np.random.choice(self.episode_buffer, batch_size, replace=False)

            states = []
            actions = []
            rewards = []

            for exp in batch:
                features_tensor = torch.FloatTensor(exp["state"])
                crisis_tensor = torch.FloatTensor([exp["crisis_level"]])
                combined_state = torch.cat(
                    [features_tensor, crisis_tensor, self.specialization_weights]
                )
                states.append(combined_state)

                actions.append(torch.FloatTensor(exp["action"]))
                rewards.append(exp["reward"])

            states = torch.stack(states)
            actions = torch.stack(actions)
            rewards = torch.FloatTensor(rewards)

            if len(rewards) > 1:
                rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

            predicted_actions = self.strategy_network(states)

            log_probs = torch.log(predicted_actions + 1e-8)
            policy_loss = -torch.mean(
                log_probs * actions.unsqueeze(1) * rewards.unsqueeze(1)
            )

            entropy = -torch.mean(
                predicted_actions * torch.log(predicted_actions + 1e-8)
            )
            entropy_bonus = 0.01 * entropy

            total_loss = policy_loss - entropy_bonus

            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.strategy_network.parameters(), 0.5)
            self.optimizer.step()

            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

        except Exception as e:
            print(f"[경고] {self.risk_type} B-세포 배치 학습 오류: {e}")

    def learn_from_specialized_experience(self):
        """전문 분야 집중 학습"""

        if len(self.specialization_buffer) < self.batch_size:
            return False

        try:
            specialist_batch = list(self.specialization_buffer)[-self.batch_size :]

            states = []
            actions = []
            rewards = []

            for exp in specialist_batch:
                features_tensor = torch.FloatTensor(exp["state"])
                crisis_tensor = torch.FloatTensor([exp["crisis_level"]])
                combined_state = torch.cat(
                    [features_tensor, crisis_tensor, self.specialization_weights]
                )
                states.append(combined_state)

                actions.append(torch.FloatTensor(exp["action"]))
                rewards.append(exp["reward"])

            states = torch.stack(states)
            actions = torch.stack(actions)
            rewards = torch.FloatTensor(rewards)

            predicted_actions = self.strategy_network(states)
            log_probs = torch.log(predicted_actions + 1e-8)

            specialist_weight = 3.0
            specialist_loss = -torch.mean(
                log_probs
                * actions.unsqueeze(1)
                * rewards.unsqueeze(1)
                * specialist_weight
            )

            self.optimizer.zero_grad()
            specialist_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.strategy_network.parameters(), 0.5)
            self.optimizer.step()

            avg_specialist_reward = torch.mean(rewards).item()
            self.adaptive_scheduler.step(avg_specialist_reward)

            return True

        except Exception as e:
            print(f"[경고] {self.risk_type} B-세포 전문가 학습 오류: {e}")
            return False

    def end_episode(self):
        """에피소드 종료"""
        if len(self.episode_buffer) > 0:
            self.experience_buffer.extend(self.episode_buffer)

            if len(self.episode_buffer) >= self.batch_size:
                self.learn_from_batch()

            self.episode_buffer = []

            if len(self.experience_buffer) > 1000:
                self.experience_buffer = self.experience_buffer[-1000:]

    def get_expertise_metrics(self):
        """전문성 지표 반환"""

        specialist_avg = (
            np.mean(self.specialist_performance) if self.specialist_performance else 0
        )
        general_avg = (
            np.mean(self.general_performance) if self.general_performance else 0
        )

        expertise_advantage = specialist_avg - general_avg if general_avg != 0 else 0

        return {
            "specialization_strength": self.specialization_strength,
            "specialist_experiences": len(self.specialization_buffer),
            "general_experiences": len(self.general_buffer),
            "specialist_avg_reward": specialist_avg,
            "general_avg_reward": general_avg,
            "expertise_advantage": expertise_advantage,
            "specialization_ratio": len(self.specialization_buffer)
            / max(1, len(self.specialization_buffer) + len(self.general_buffer)),
            "risk_type": self.risk_type,
        }

    def learn_from_experience(self, market_features, crisis_level, effectiveness):
        """호환성 래퍼"""
        if len(market_features) >= 8:
            dummy_action = np.ones(self.n_assets) / self.n_assets
            self.add_experience(
                market_features, crisis_level, dummy_action, effectiveness
            )

            if self.experience_count % self.update_frequency == 0:
                self.learn_from_batch()

    def adapt_response(self, antigen_pattern, effectiveness):
        """호환성 래퍼"""
        if len(antigen_pattern) >= 8:
            crisis_level = 0.5
            self.learn_from_experience(antigen_pattern, crisis_level, effectiveness)


class LegacyBCell(ImmuneCell):
    """규칙 기반 B-세포"""

    def __init__(self, cell_id, risk_type, response_strategy):
        super().__init__(cell_id)
        self.risk_type = risk_type
        self.response_strategy = response_strategy
        self.antibody_strength = 0.1

    def produce_antibody(self, antigen_pattern):
        """전략 생성"""
        if hasattr(self, "learned_patterns"):
            similarities = [
                cosine_similarity([antigen_pattern], [pattern])[0][0]
                for pattern in self.learned_patterns
            ]
            max_similarity = max(similarities) if similarities else 0
        else:
            max_similarity = 0

        self.antibody_strength = min(1.0, max_similarity + 0.1)
        return self.antibody_strength

    def adapt_response(self, antigen_pattern, effectiveness):
        """적응적 학습"""
        if not hasattr(self, "learned_patterns"):
            self.learned_patterns = []

        if effectiveness > 0.6:
            self.learned_patterns.append(antigen_pattern.copy())
            if len(self.learned_patterns) > 10:
                self.learned_patterns.pop(0)


class MemoryCell:
    """기억 세포"""

    def __init__(self, max_memories=20):
        self.max_memories = max_memories
        self.crisis_memories = []

    def store_memory(self, crisis_pattern, response_strategy, effectiveness):
        """기억 저장"""
        memory = {
            "pattern": crisis_pattern.copy(),
            "strategy": response_strategy.copy(),
            "effectiveness": effectiveness,
            "strength": 1.0,
        }

        self.crisis_memories.append(memory)

        if len(self.crisis_memories) > self.max_memories:
            self.crisis_memories.sort(key=lambda x: x["effectiveness"])
            self.crisis_memories.pop(0)

    def recall_memory(self, current_pattern):
        """기억 회상"""
        if not self.crisis_memories:
            return None, 0.0

        similarities = []
        for memory in self.crisis_memories:
            similarity = cosine_similarity([current_pattern], [memory["pattern"]])[0][0]
            similarities.append(similarity * memory["effectiveness"])

        best_memory_idx = np.argmax(similarities)
        best_similarity = similarities[best_memory_idx]

        if best_similarity > 0.7:
            return self.crisis_memories[best_memory_idx], best_similarity

        return None, 0.0


class ImmunePortfolioSystem:
    """면역 포트폴리오 시스템"""

    def __init__(
        self,
        n_assets,
        n_tcells=3,
        n_bcells=5,
        random_state=None,
        use_learning_bcells=True,
        logging_level="full",
    ):
        self.n_assets = n_assets
        self.use_learning_bcells = use_learning_bcells
        self.logging_level = logging_level  # 'full', 'sample', 'minimal'

        # T-세포 초기화
        self.tcells = [
            TCell(
                f"T{i}",
                sensitivity=0.05 + i * 0.02,
                random_state=None if random_state is None else random_state + i,
            )
            for i in range(n_tcells)
        ]

        # B-세포 초기화
        if use_learning_bcells:
            feature_size = 12  # 특성 크기
            input_size = feature_size + 1 + n_assets

            self.bcells = [
                BCell("B1-Vol", "volatility", input_size, n_assets),
                BCell("B2-Corr", "correlation", input_size, n_assets),
                BCell("B3-Mom", "momentum", input_size, n_assets),
                BCell("B4-Liq", "liquidity", input_size, n_assets),
                BCell("B5-Macro", "macro", input_size, n_assets),
            ]
            print("시스템 유형: 적응형 신경망 기반 BIPD 모델")
        else:
            self.bcells = [
                LegacyBCell("LB1-Vol", "volatility", self._volatility_response),
                LegacyBCell("LB2-Corr", "correlation", self._correlation_response),
                LegacyBCell("LB3-Mom", "momentum", self._momentum_response),
                LegacyBCell("LB4-Liq", "liquidity", self._liquidity_response),
                LegacyBCell("LB5-Macro", "macro", self._macro_response),
            ]
            print("시스템 유형: 규칙 기반 레거시 BIPD 모델")

        # 기억 세포
        self.memory_cell = MemoryCell()

        # 포트폴리오 가중치
        self.base_weights = np.ones(n_assets) / n_assets
        self.current_weights = self.base_weights.copy()

        # 시스템 상태
        self.immune_activation = 0.0
        self.crisis_level = 0.0

        # 분석 시스템
        self.analyzer = DecisionAnalyzer()
        self.enable_logging = True

        # 로깅 레벨에 따른 설정
        if logging_level == "full":
            print("분석 시스템이 활성화되었습니다. (전체 로깅)")
        elif logging_level == "sample":
            print("분석 시스템이 활성화되었습니다. (샘플링 로깅)")
        elif logging_level == "minimal":
            print("분석 시스템이 활성화되었습니다. (최소 로깅)")
        else:
            print("분석 시스템이 활성화되었습니다.")

    def extract_market_features(self, market_data, lookback=20):
        """시장 특성 추출"""
        if len(market_data) < lookback:
            return np.zeros(12)

        # 현재 날짜 기준으로 기술적 지표 데이터 활용
        return self._extract_technical_features(market_data, lookback)

    def _extract_basic_features(self, market_data, lookback=20):
        """기본 특성 추출 (기존 방식)"""
        returns = market_data.pct_change().dropna()
        if len(returns) == 0:
            return np.zeros(8)

        recent_returns = returns.iloc[-lookback:]
        if len(recent_returns) == 0:
            return np.zeros(8)

        def safe_mean(x):
            if len(x) == 0 or x.isnull().all():
                return 0.0
            return x.mean() if not np.isnan(x.mean()) else 0.0

        def safe_std(x):
            if len(x) == 0 or x.isnull().all():
                return 0.0
            return x.std() if not np.isnan(x.std()) else 0.0

        def safe_corr(x):
            try:
                if len(x) <= 1 or x.isnull().all().all():
                    return 0.0
                corr_matrix = np.corrcoef(x.T)
                if np.isnan(corr_matrix).any():
                    return 0.0
                return np.mean(corr_matrix[~np.eye(corr_matrix.shape[0], dtype=bool)])
            except:
                return 0.0

        def safe_skew(x):
            try:
                skew_vals = x.skew()
                if skew_vals.isnull().all():
                    return 0.0
                return skew_vals.mean() if not np.isnan(skew_vals.mean()) else 0.0
            except:
                return 0.0

        def safe_kurtosis(x):
            try:
                kurt_vals = x.kurtosis()
                if kurt_vals.isnull().all():
                    return 0.0
                return kurt_vals.mean() if not np.isnan(kurt_vals.mean()) else 0.0
            except:
                return 0.0

        features = [
            safe_std(recent_returns.std()),
            safe_corr(recent_returns),
            safe_mean(recent_returns.mean()),
            safe_skew(recent_returns),
            safe_kurtosis(recent_returns),
            safe_std(recent_returns.std()),
            len(recent_returns[recent_returns.sum(axis=1) < -0.02])
            / max(len(recent_returns), 1),
            (
                max(recent_returns.max().max() - recent_returns.min().min(), 0)
                if not recent_returns.empty
                else 0
            ),
        ]

        features = np.array(features)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        return features

    def _extract_technical_features(self, market_data, lookback=20):
        """기술적 지표 기반 특성 추출"""
        if not hasattr(self, "train_features") or not hasattr(self, "test_features"):
            # 기술적 지표 데이터가 없는 경우 기본 방식 사용
            basic_features = self._extract_basic_features(market_data, lookback)
            return self._expand_to_12_features(basic_features)

        # 현재 날짜 기준으로 특성 데이터 선택
        current_date = market_data.index[-1]

        # 훈련 또는 테스트 데이터에서 특성 추출
        if current_date in self.train_features.index:
            feature_data = self.train_features.loc[current_date]
        elif current_date in self.test_features.index:
            feature_data = self.test_features.loc[current_date]
        else:
            basic_features = self._extract_basic_features(market_data, lookback)
            return self._expand_to_12_features(basic_features)

        # 핵심 시장 지표 선택 (위기 감지에 중요한 지표들)
        selected_features = []

        # 1. 시장 전체 변동성 (가장 중요한 위기 지표)
        market_volatility = feature_data.get("market_volatility", 0.0)
        selected_features.append(
            np.clip(market_volatility * 5, 0, 1)
        )  # 증폭하여 민감도 증가

        # 2. 시장 상관관계 (시스템적 위험 지표)
        market_correlation = feature_data.get("market_correlation", 0.5)
        selected_features.append(np.clip(abs(market_correlation), 0, 1))

        # 3. 시장 수익률 (방향성 지표)
        market_return = feature_data.get("market_return", 0.0)
        selected_features.append(np.clip(market_return * 10, -1, 1))  # 증폭

        # 4. VIX 대용 지표 (변동성의 변동성)
        vix_proxy = feature_data.get("vix_proxy", 0.1)
        selected_features.append(np.clip(vix_proxy * 3, 0, 1))  # 증폭

        # 5. 시장 스트레스 지수
        market_stress = feature_data.get("market_stress", 0.0)
        selected_features.append(np.clip(market_stress / 10, 0, 1))  # 정규화

        # 6. 평균 RSI (과매수/과매도 지표)
        rsi_cols = [col for col in feature_data.index if "_rsi" in col]
        if rsi_cols:
            avg_rsi = np.mean(
                [
                    feature_data[col]
                    for col in rsi_cols
                    if not pd.isna(feature_data[col])
                ]
            )
            # RSI 50에서 벗어날수록 위험 증가
            rsi_risk = abs(avg_rsi - 50) / 50
            selected_features.append(np.clip(rsi_risk, 0, 1))
        else:
            selected_features.append(0.0)

        # 7. 모멘텀 지표
        momentum_cols = [col for col in feature_data.index if "_momentum" in col]
        if momentum_cols:
            avg_momentum = np.mean(
                [
                    feature_data[col]
                    for col in momentum_cols
                    if not pd.isna(feature_data[col])
                ]
            )
            selected_features.append(np.clip(abs(avg_momentum), 0, 1))
        else:
            selected_features.append(0.0)

        # 8. 볼린저 밴드 위치 (극단적 위치일수록 위험)
        bb_cols = [col for col in feature_data.index if "_bb_position" in col]
        if bb_cols:
            avg_bb_position = np.mean(
                [feature_data[col] for col in bb_cols if not pd.isna(feature_data[col])]
            )
            # 0.5에서 벗어날수록 위험
            bb_risk = abs(avg_bb_position - 0.5) * 2
            selected_features.append(np.clip(bb_risk, 0, 1))
        else:
            selected_features.append(0.0)

        # 9. 거래량 이상 지표
        volume_cols = [col for col in feature_data.index if "_volume_ratio" in col]
        if volume_cols:
            avg_volume_ratio = np.mean(
                [
                    feature_data[col]
                    for col in volume_cols
                    if not pd.isna(feature_data[col])
                ]
            )
            # 정상 거래량(1.0)에서 벗어날수록 위험
            volume_risk = abs(avg_volume_ratio - 1.0) / 2
            selected_features.append(np.clip(volume_risk, 0, 1))
        else:
            selected_features.append(0.0)

        # 10. 가격 범위 확장 지표
        range_cols = [col for col in feature_data.index if "_price_range" in col]
        if range_cols:
            avg_range = np.mean(
                [
                    feature_data[col]
                    for col in range_cols
                    if not pd.isna(feature_data[col])
                ]
            )
            selected_features.append(np.clip(avg_range * 2, 0, 1))
        else:
            selected_features.append(0.1)

        # 11. 이동평균 이탈도
        sma_cols = [col for col in feature_data.index if "_price_sma20_ratio" in col]
        if sma_cols:
            avg_sma_ratio = np.mean(
                [
                    feature_data[col]
                    for col in sma_cols
                    if not pd.isna(feature_data[col])
                ]
            )
            # 1.0에서 벗어날수록 위험
            sma_risk = abs(avg_sma_ratio - 1.0)
            selected_features.append(np.clip(sma_risk, 0, 1))
        else:
            selected_features.append(0.0)

        # 12. 종합 변동성 지표
        vol_cols = [col for col in feature_data.index if "_volatility" in col]
        if vol_cols:
            avg_volatility = np.mean(
                [
                    feature_data[col]
                    for col in vol_cols
                    if not pd.isna(feature_data[col])
                ]
            )
            selected_features.append(np.clip(avg_volatility * 5, 0, 1))  # 증폭
        else:
            selected_features.append(0.1)

        # 12개 특성 보장
        while len(selected_features) < 12:
            selected_features.append(0.0)

        # 최종 특성 배열 생성 (정확히 12개)
        features = np.array(selected_features[:12])
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        return features

    def _expand_to_12_features(self, basic_features):
        """8개 기본 특성을 12개로 확장"""
        if len(basic_features) >= 12:
            return basic_features[:12]

        # 기본 8개 특성에 추가 특성 4개 추가
        expanded_features = list(basic_features)

        # 추가 특성들 (기본값으로 설정)
        additional_features = [
            0.5,  # RSI 중립값
            0.0,  # 모멘텀 중립값
            0.5,  # 볼린저 밴드 중립값
            1.0,  # 거래량 비율 중립값
        ]

        # 필요한 만큼 추가
        for i in range(12 - len(expanded_features)):
            if i < len(additional_features):
                expanded_features.append(additional_features[i])
            else:
                expanded_features.append(0.0)

        return np.array(expanded_features[:12])

    def _get_dominant_risk(self, market_features):
        """지배적 위험 유형 계산"""
        risk_features = market_features[:5]
        dominant_risk_idx = np.argmax(np.abs(risk_features - np.mean(risk_features)))
        risk_map = {
            0: "volatility",
            1: "correlation",
            2: "momentum",
            3: "liquidity",
            4: "macro",
        }
        return risk_map.get(dominant_risk_idx, "volatility")

    def immune_response(self, market_features, training=False):
        """면역 반응 실행"""

        # T-세포 활성화 및 상세 위기 감지 로그 수집
        tcell_activations = []
        detailed_crisis_logs = []

        for tcell in self.tcells:
            activation = tcell.detect_anomaly(market_features)
            tcell_activations.append(activation)

            # 상세 위기 감지 로그 수집 (활성화 레벨이 임계값 이상인 경우)
            if hasattr(tcell, "last_crisis_detection") and tcell.last_crisis_detection:
                detailed_crisis_logs.append(tcell.last_crisis_detection)

        self.crisis_level = np.mean(tcell_activations)

        # 상세 T-cell 분석 정보 저장
        self.detailed_tcell_analysis = {
            "crisis_level": self.crisis_level,
            "detailed_crisis_logs": detailed_crisis_logs,
        }

        # 기억 세포 확인
        recalled_memory, memory_strength = self.memory_cell.recall_memory(
            market_features
        )

        if recalled_memory and memory_strength > 0.8:
            bcell_decisions = [
                {
                    "id": "Memory",
                    "risk_type": "memory_recall",
                    "activation_level": memory_strength,
                    "antibody_strength": memory_strength,
                    "strategy_contribution": 1.0,
                    "specialized_for_today": True,
                }
            ]
            return recalled_memory["strategy"], "memory_response", bcell_decisions

        # B-세포 활성화 (더 민감한 임계값)
        if self.crisis_level > 0.15:
            if self.use_learning_bcells:
                response_weights = []
                antibody_strengths = []

                for bcell in self.bcells:
                    strategy, antibody_strength = bcell.produce_antibody(
                        market_features, self.crisis_level, training=training
                    )
                    response_weights.append(strategy)
                    antibody_strengths.append(antibody_strength)

                if len(antibody_strengths) > 0 and sum(antibody_strengths) > 0:
                    normalized_strengths = np.array(antibody_strengths) / sum(
                        antibody_strengths
                    )

                    ensemble_strategy = np.zeros(self.n_assets)
                    for i, (strategy, weight) in enumerate(
                        zip(response_weights, normalized_strengths)
                    ):
                        ensemble_strategy += strategy * weight

                    ensemble_strategy = ensemble_strategy / np.sum(ensemble_strategy)
                    self.immune_activation = np.mean(antibody_strengths)

                    # B-세포 정보 수집
                    bcell_decisions = []
                    dominant_risk = self._get_dominant_risk(market_features)

                    for i, bcell in enumerate(self.bcells):
                        bcell_decisions.append(
                            {
                                "id": bcell.cell_id,
                                "risk_type": bcell.risk_type,
                                "activation_level": float(normalized_strengths[i]),
                                "antibody_strength": float(antibody_strengths[i]),
                                "strategy_contribution": float(normalized_strengths[i]),
                                "specialized_for_today": bcell.risk_type
                                == dominant_risk,
                            }
                        )

                    dominant_bcell_idx = np.argmax(antibody_strengths)
                    response_type = (
                        f"ensemble_{self.bcells[dominant_bcell_idx].risk_type}"
                    )

                    return ensemble_strategy, response_type, bcell_decisions
                else:
                    return self.base_weights, "fallback", []
            else:
                # 규칙 기반 시스템
                response_weights = []
                antibody_strengths = []

                for bcell in self.bcells:
                    antibody_strength = bcell.produce_antibody(market_features)
                    response_weight = bcell.response_strategy(
                        self.crisis_level * antibody_strength
                    )
                    response_weights.append(response_weight)
                    antibody_strengths.append(antibody_strength)

                best_response_idx = np.argmax(antibody_strengths)
                self.immune_activation = antibody_strengths[best_response_idx]

                bcell_decisions = [
                    {
                        "id": self.bcells[best_response_idx].cell_id,
                        "risk_type": self.bcells[best_response_idx].risk_type,
                        "activation_level": float(self.immune_activation),
                        "antibody_strength": float(self.immune_activation),
                        "strategy_contribution": 1.0,
                        "specialized_for_today": True,
                    }
                ]

                return (
                    response_weights[best_response_idx],
                    f"legacy_{self.bcells[best_response_idx].risk_type}",
                    bcell_decisions,
                )

        return self.base_weights, "normal", []

    def _volatility_response(self, activation_level):
        """변동성 위험 대응"""
        risk_reduction = activation_level * 0.3
        weights = self.base_weights * (1 - risk_reduction)
        safe_indices = [6, 7, 8]
        for idx in safe_indices:
            if idx < len(weights):
                weights[idx] += risk_reduction / len(safe_indices)
        return weights / np.sum(weights)

    def _correlation_response(self, activation_level):
        """상관관계 위험 대응"""
        diversification_boost = activation_level * 0.2
        weights = self.base_weights.copy()
        weights = weights * (1 - diversification_boost) + diversification_boost / len(
            weights
        )
        return weights / np.sum(weights)

    def _momentum_response(self, activation_level):
        """모멘텀 위험 대응"""
        neutral_adjustment = activation_level * 0.25
        weights = self.base_weights * (1 - neutral_adjustment) + (
            self.base_weights * neutral_adjustment
        )
        return weights / np.sum(weights)

    def _liquidity_response(self, activation_level):
        """유동성 위험 대응"""
        large_cap_boost = activation_level * 0.2
        weights = self.base_weights.copy()
        large_cap_indices = [0, 1, 2, 3]
        for idx in large_cap_indices:
            if idx < len(weights):
                weights[idx] += large_cap_boost / len(large_cap_indices)
        return weights / np.sum(weights)

    def _macro_response(self, activation_level):
        """거시경제 위험 대응"""
        defensive_boost = activation_level * 0.3
        weights = self.base_weights * (1 - defensive_boost)
        defensive_indices = [7, 8, 9]
        for idx in defensive_indices:
            if idx < len(weights):
                weights[idx] += defensive_boost / len(defensive_indices)
        return weights / np.sum(weights)

    def pretrain_bcells(self, market_data, episodes=500):
        """B-세포 사전 훈련"""
        if not self.use_learning_bcells:
            return

        print(f"B-세포 네트워크 사전 훈련 시작 (에피소드: {episodes})")

        expert_policy_functions = {
            "volatility": self._volatility_response,
            "correlation": self._correlation_response,
            "momentum": self._momentum_response,
            "liquidity": self._liquidity_response,
            "macro": self._macro_response,
        }

        loss_function = nn.MSELoss()

        for episode in tqdm(range(episodes), desc="사전 훈련 진행률"):
            start_idx = np.random.randint(
                20, len(market_data.pct_change().dropna()) - 50
            )
            current_data = market_data.iloc[:start_idx]
            market_features = self.extract_market_features(current_data)
            crisis_level = np.random.uniform(0.2, 0.8)

            for bcell in self.bcells:
                if bcell.risk_type in expert_policy_functions:
                    expert_action = expert_policy_functions[bcell.risk_type](
                        crisis_level
                    )
                    target_policy = torch.FloatTensor(expert_action)

                    features_tensor = torch.FloatTensor(market_features)
                    crisis_tensor = torch.FloatTensor([crisis_level])
                    specialization_tensor = bcell.specialization_weights
                    combined_input = torch.cat(
                        [features_tensor, crisis_tensor, specialization_tensor]
                    )
                    current_policy = bcell.strategy_network(
                        combined_input.unsqueeze(0)
                    ).squeeze(0)

                    bcell.optimizer.zero_grad()
                    loss = loss_function(current_policy, target_policy)
                    loss.backward()
                    bcell.optimizer.step()

        print("B-세포 네트워크 사전 훈련이 완료되었습니다.")

    def update_memory(self, crisis_pattern, response_strategy, effectiveness):
        """기억 업데이트"""
        self.memory_cell.store_memory(crisis_pattern, response_strategy, effectiveness)

        if self.use_learning_bcells:
            for bcell in self.bcells:
                bcell.learn_from_experience(
                    crisis_pattern, self.crisis_level, effectiveness
                )
        else:
            for bcell in self.bcells:
                bcell.adapt_response(crisis_pattern, effectiveness)


class ImmunePortfolioBacktester:
    def __init__(self, symbols, train_start, train_end, test_start, test_end):
        self.symbols = symbols
        self.train_start = train_start
        self.train_end = train_end
        self.test_start = test_start
        self.test_end = test_end
        
        # 타임스탬프 기반 통합 출력 디렉토리 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(RESULTS_DIR, f"analysis_{timestamp}")
        os.makedirs(self.output_dir, exist_ok=True)

        # 데이터 로드
        data_filename = f"market_data_{'_'.join(symbols)}_{train_start}_{test_end}.pkl"
        self.data_path = os.path.join(DATA_DIR, data_filename)

        if os.path.exists(self.data_path):
            print(f"기존 데이터 로드 중: {data_filename}")
            with open(self.data_path, "rb") as f:
                self.data = pickle.load(f)
        else:
            print("포괄적 시장 데이터 다운로드 중...")
            raw_data = yf.download(
                symbols, start="2007-12-01", end="2025-01-01", progress=True
            )

            # 다중 지표 데이터 처리
            self.data = self._process_comprehensive_data(raw_data, symbols)

            # 데이터 전처리는 _process_comprehensive_data에서 이미 완료됨
            print("데이터 전처리 완료")

            # 데이터 저장
            with open(self.data_path, "wb") as f:
                pickle.dump(self.data, f)
            print(f"포괄적 시장 데이터 저장 완료: {data_filename}")
            print(f"데이터 구조: {list(self.data.keys())}")

        # 데이터 분할
        self.train_data = self.data["prices"][train_start:train_end]
        self.test_data = self.data["prices"][test_start:test_end]
        self.train_features = self.data["features"][train_start:train_end]
        self.test_features = self.data["features"][test_start:test_end]

        # 기존 코드 호환성을 위한 추가 정리
        self.train_data = self._clean_data(self.train_data)
        self.test_data = self._clean_data(self.test_data)

    def _process_comprehensive_data(self, raw_data, symbols):
        """포괄적인 시장 데이터 처리"""
        print("다중 지표 데이터 처리 중...")

        # 기본 가격 데이터 추출
        if len(symbols) == 1:
            if "Adj Close" in raw_data.columns:
                prices = raw_data["Adj Close"].to_frame(symbols[0])
            elif "Close" in raw_data.columns:
                prices = raw_data["Close"].to_frame(symbols[0])
            else:
                raise ValueError("가격 데이터를 찾을 수 없습니다.")
        else:
            try:
                prices = raw_data["Adj Close"]
            except KeyError:
                try:
                    prices = raw_data["Close"]
                    print("주의: 'Adj Close' 없음, 'Close' 사용")
                except KeyError:
                    price_data = {}
                    for symbol in symbols:
                        if ("Adj Close", symbol) in raw_data.columns:
                            price_data[symbol] = raw_data[("Adj Close", symbol)]
                        elif ("Close", symbol) in raw_data.columns:
                            price_data[symbol] = raw_data[("Close", symbol)]
                        else:
                            print(f"경고: {symbol} 가격 데이터를 찾을 수 없습니다.")
                            continue
                    if not price_data:
                        raise ValueError("사용 가능한 가격 데이터가 없습니다.")
                    prices = pd.DataFrame(price_data)

        # 추가 지표 계산
        features = self._calculate_technical_indicators(raw_data, symbols)

        # 데이터 정리
        prices = self._clean_data(prices)
        features = self._clean_data(features)

        return {"prices": prices, "features": features, "raw_data": raw_data}

    def _calculate_technical_indicators(self, raw_data, symbols):
        """기술적 지표 계산"""
        print("기술적 지표 계산 중...")

        features = {}

        for symbol in symbols:
            try:
                # 가격 데이터 추출
                if len(symbols) == 1:
                    high = (
                        raw_data["High"]
                        if "High" in raw_data.columns
                        else raw_data["Close"]
                    )
                    low = (
                        raw_data["Low"]
                        if "Low" in raw_data.columns
                        else raw_data["Close"]
                    )
                    close = (
                        raw_data["Adj Close"]
                        if "Adj Close" in raw_data.columns
                        else raw_data["Close"]
                    )
                    volume = (
                        raw_data["Volume"]
                        if "Volume" in raw_data.columns
                        else pd.Series(1, index=raw_data.index)
                    )
                else:
                    high = (
                        raw_data["High"][symbol]
                        if "High" in raw_data.columns
                        else raw_data["Close"][symbol]
                    )
                    low = (
                        raw_data["Low"][symbol]
                        if "Low" in raw_data.columns
                        else raw_data["Close"][symbol]
                    )
                    close = (
                        raw_data["Adj Close"][symbol]
                        if "Adj Close" in raw_data.columns
                        else raw_data["Close"][symbol]
                    )
                    volume = (
                        raw_data["Volume"][symbol]
                        if "Volume" in raw_data.columns
                        else pd.Series(1, index=raw_data.index)
                    )

                # 기술적 지표 계산
                symbol_features = pd.DataFrame(index=close.index)

                # 1. 가격 기반 지표
                symbol_features[f"{symbol}_returns"] = close.pct_change()
                symbol_features[f"{symbol}_volatility"] = (
                    symbol_features[f"{symbol}_returns"].rolling(20).std()
                )
                symbol_features[f"{symbol}_sma_20"] = close.rolling(20).mean()
                symbol_features[f"{symbol}_sma_50"] = close.rolling(50).mean()
                symbol_features[f"{symbol}_price_sma20_ratio"] = (
                    close / symbol_features[f"{symbol}_sma_20"]
                )
                symbol_features[f"{symbol}_price_sma50_ratio"] = (
                    close / symbol_features[f"{symbol}_sma_50"]
                )

                # 2. 모멘텀 지표
                symbol_features[f"{symbol}_rsi"] = self._calculate_rsi(close, 14)
                symbol_features[f"{symbol}_momentum"] = close / close.shift(10) - 1

                # 3. 볼린저 밴드
                bb_upper, bb_lower = self._calculate_bollinger_bands(close, 20, 2)
                symbol_features[f"{symbol}_bb_position"] = (close - bb_lower) / (
                    bb_upper - bb_lower
                )

                # 4. 거래량 지표
                symbol_features[f"{symbol}_volume_sma"] = volume.rolling(20).mean()
                symbol_features[f"{symbol}_volume_ratio"] = (
                    volume / symbol_features[f"{symbol}_volume_sma"]
                )

                # 5. 변동성 지표
                symbol_features[f"{symbol}_high_low_ratio"] = (high - low) / close
                symbol_features[f"{symbol}_price_range"] = (high - low) / close.rolling(
                    20
                ).mean()

                features[symbol] = symbol_features

            except Exception as e:
                print(f"[경고] {symbol} 기술적 지표 계산 중 오류 발생: {e}")
                continue

        # 전체 특성 데이터프레임 생성
        all_features = pd.concat(features.values(), axis=1)

        # 시장 전체 지표 추가
        all_features = self._add_market_indicators(all_features, symbols)

        return all_features

    def _calculate_rsi(self, prices, period=14):
        """RSI 계산"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_bollinger_bands(self, prices, period=20, std_dev=2):
        """볼린저 밴드 계산"""
        sma = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, lower_band

    def _add_market_indicators(self, features, symbols):
        """시장 전체 지표 추가"""
        print("시장 전체 지표 계산 중...")

        try:
            # 시장 전체 수익률 (동일 가중)
            return_cols = [col for col in features.columns if "_returns" in col]
            if return_cols:
                features["market_return"] = features[return_cols].mean(axis=1)
                features["market_volatility"] = features[return_cols].std(axis=1)
                # 상관계수 계산 개선
                corr_values = []
                for i in range(len(features)):
                    try:
                        window_data = features[return_cols].iloc[max(0, i - 19) : i + 1]
                        if len(window_data) >= 2:
                            corr_matrix = window_data.corr()
                            upper_tri = corr_matrix.where(
                                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
                            )
                            corr_values.append(upper_tri.stack().mean())
                        else:
                            corr_values.append(0.0)
                    except:
                        corr_values.append(0.0)
                features["market_correlation"] = pd.Series(
                    corr_values, index=features.index
                )

            # VIX 대용 지표 (변동성의 변동성)
            vol_cols = [col for col in features.columns if "_volatility" in col]
            if vol_cols:
                features["vix_proxy"] = (
                    features[vol_cols].mean(axis=1).rolling(10).std()
                )

            # 시장 스트레스 지수
            rsi_cols = [col for col in features.columns if "_rsi" in col]
            if rsi_cols:
                features["market_stress"] = features[rsi_cols].apply(
                    lambda x: (x < 30).sum() + (x > 70).sum(), axis=1
                )
            else:
                features["market_stress"] = 0

            # 결측값 처리
            market_cols = [
                "market_return",
                "market_volatility",
                "market_correlation",
                "vix_proxy",
                "market_stress",
            ]
            for col in market_cols:
                if col in features.columns:
                    features[col] = features[col].fillna(0)

        except Exception as e:
            print(f"[경고] 시장 전체 지표 계산 중 오류 발생: {e}")
            # 기본값 설정
            features["market_return"] = 0.0
            features["market_volatility"] = 0.1
            features["market_correlation"] = 0.5
            features["vix_proxy"] = 0.1
            features["market_stress"] = 0.0

        return features

    def _clean_data(self, data):
        """데이터 정리"""
        print("데이터 전처리 중...")

        if data.isnull().values.any():
            print("결측값 발견, 전방향/후방향 채우기 적용")
            data = data.fillna(method="ffill").fillna(method="bfill")

        if data.isnull().values.any():
            print("잔여 결측값을 0으로 채움")
            data = data.fillna(0)

        if np.isinf(data.values).any():
            print("무한대 값 발견, 유한값으로 변환")
            data = data.replace([np.inf, -np.inf], 0)

        if data.isnull().values.any() or np.isinf(data.values).any():
            print("최종 데이터 정리 중...")
            data = pd.DataFrame(
                np.nan_to_num(data.values, nan=0.0, posinf=0.0, neginf=0.0),
                index=data.index,
                columns=data.columns,
            )

        return data

    def calculate_metrics(self, returns, initial_capital=1e6):
        """성과 지표 계산"""
        cum_returns = (1 + returns).cumprod()
        final_value = initial_capital * cum_returns.iloc[-1]
        total_return = (final_value - initial_capital) / initial_capital

        volatility = returns.std() * np.sqrt(252)
        max_drawdown = self.calculate_max_drawdown(returns)

        sharpe_ratio = (
            returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        )

        calmar_ratio = (
            returns.mean() * 252 / abs(max_drawdown) if max_drawdown != 0 else 0
        )

        return {
            "Total Return": total_return,
            "Volatility": volatility,
            "Max Drawdown": max_drawdown,
            "Sharpe Ratio": sharpe_ratio,
            "Calmar Ratio": calmar_ratio,
            "Final Value": final_value,
            "Initial Capital": initial_capital,
        }

    def calculate_max_drawdown(self, returns):
        """최대 낙폭 계산"""
        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.expanding().max()
        drawdown = (cum_returns - running_max) / running_max
        return drawdown.min()

    def backtest_single_run(
        self,
        seed=None,
        return_model=False,
        use_learning_bcells=True,
        logging_level="full",
    ):
        """단일 백테스트 실행"""

        if seed is not None:
            np.random.seed(seed)
            if use_learning_bcells:
                torch.manual_seed(seed)

        immune_system = ImmunePortfolioSystem(
            n_assets=len(self.symbols),
            random_state=seed,
            use_learning_bcells=use_learning_bcells,
            logging_level=logging_level,
        )

        # 사전 훈련
        if use_learning_bcells:
            immune_system.pretrain_bcells(self.train_data, episodes=500)

        # 훈련 단계
        print("적응형 학습 진행 중...")
        train_returns = self.train_data.pct_change().dropna()
        portfolio_values = [1.0]

        for i in tqdm(range(len(train_returns)), desc="적응형 학습"):
            current_data = self.train_data.iloc[: i + 1]
            market_features = immune_system.extract_market_features(current_data)

            # 면역 반응 실행
            weights, response_type, bcell_decisions = immune_system.immune_response(
                market_features, training=True
            )

            portfolio_return = np.sum(weights * train_returns.iloc[i])
            portfolio_values.append(portfolio_values[-1] * (1 + portfolio_return))

            # 로깅
            if hasattr(immune_system, "analyzer") and immune_system.enable_logging:
                current_date = train_returns.index[i]
                immune_system.analyzer.log_decision(
                    date=current_date,
                    market_features=market_features,
                    tcell_analysis=getattr(
                        immune_system,
                        "detailed_tcell_analysis",
                        {"crisis_level": immune_system.crisis_level},
                    ),
                    bcell_decisions=bcell_decisions,
                    final_weights=weights,
                    portfolio_return=portfolio_return,
                    crisis_level=immune_system.crisis_level,
                )

            # 학습 로직
            if len(portfolio_values) > 20:
                base_reward = portfolio_return * 100
                if portfolio_return < 0:
                    base_reward -= (portfolio_return * 150) ** 2

                running_max = np.maximum.accumulate(portfolio_values)
                drawdown = (portfolio_values[-1] - running_max[-1]) / (
                    running_max[-1] + 1e-8
                )
                if drawdown < 0:
                    base_reward += drawdown * 50

                # 지배적 위험 판단
                risk_features = market_features[:5]
                dominant_risk_idx = np.argmax(
                    np.abs(risk_features - np.mean(risk_features))
                )
                risk_map = {
                    0: "volatility",
                    1: "correlation",
                    2: "momentum",
                    3: "liquidity",
                    4: "macro",
                }
                dominant_risk = risk_map.get(dominant_risk_idx, "volatility")

                # B-세포 학습
                if use_learning_bcells:
                    for bcell in immune_system.bcells:
                        if hasattr(bcell, "last_strategy"):
                            # 전문 분야 보상 조정
                            is_specialist_today = bcell.is_my_specialty_situation(
                                market_features, immune_system.crisis_level
                            )

                            if is_specialist_today:
                                specialist_reward = base_reward * 2.0
                            else:
                                specialist_reward = base_reward * 0.8

                            final_reward = np.clip(specialist_reward, -2, 2)

                            # 경험 추가
                            bcell.add_experience(
                                market_features,
                                immune_system.crisis_level,
                                bcell.last_strategy.numpy(),
                                final_reward,
                            )

                            # 주기적 전문가 학습
                            if i % 20 == 0:
                                bcell.learn_from_specialized_experience()

                # 기억 세포 업데이트 (더 민감한 임계값)
                if immune_system.crisis_level > 0.15:
                    immune_system.update_memory(
                        market_features, weights, np.clip(base_reward, -1, 1)
                    )

        # 에피소드 종료
        if use_learning_bcells:
            for bcell in immune_system.bcells:
                bcell.end_episode()

        # 테스트 단계
        print("테스트 데이터 기반 성능 평가 진행 중...")
        test_returns = self.test_data.pct_change().dropna()
        test_portfolio_returns = []

        for i in tqdm(range(len(test_returns)), desc="성능 평가"):
            current_data = self.test_data.iloc[: i + 1]
            market_features = immune_system.extract_market_features(current_data)

            weights, response_type, bcell_decisions = immune_system.immune_response(
                market_features, training=False
            )

            portfolio_return = np.sum(weights * test_returns.iloc[i])
            test_portfolio_returns.append(portfolio_return)

            # 테스트 로깅 (로깅 레벨에 따라 조정)
            should_log = False
            if hasattr(immune_system, "analyzer") and immune_system.enable_logging:
                if immune_system.logging_level == "full":
                    should_log = True
                elif immune_system.logging_level == "sample":
                    should_log = i % 10 == 0
                elif immune_system.logging_level == "minimal":
                    should_log = i % 50 == 0

                if should_log:
                    current_date = test_returns.index[i]
                    immune_system.analyzer.log_decision(
                        date=current_date,
                        market_features=market_features,
                        tcell_analysis=getattr(
                            immune_system,
                            "detailed_tcell_analysis",
                            {"crisis_level": immune_system.crisis_level},
                        ),
                        bcell_decisions=bcell_decisions,
                        final_weights=weights,
                        portfolio_return=portfolio_return,
                        crisis_level=immune_system.crisis_level,
                    )

        if return_model:
            return (
                pd.Series(test_portfolio_returns, index=test_returns.index),
                immune_system,
            )
        else:
            return pd.Series(test_portfolio_returns, index=test_returns.index)

    def analyze_bcell_expertise(self):
        """B-세포 전문성 분석"""

        if (
            not hasattr(self, "immune_system")
            or not self.immune_system.use_learning_bcells
        ):
            return {"error": "학습 기반 시스템을 사용할 수 없습니다."}

        print("B-세포 전문화 시스템 분석 중...")

        total_specialist_exp = 0
        total_general_exp = 0
        bcell_analysis = []

        for bcell in self.immune_system.bcells:
            if hasattr(bcell, "get_expertise_metrics"):
                metrics = bcell.get_expertise_metrics()
                bcell_analysis.append(metrics)

                total_specialist_exp += metrics["specialist_experiences"]
                total_general_exp += metrics["general_experiences"]

        overall_specialization = total_specialist_exp / max(
            1, total_specialist_exp + total_general_exp
        )

        analysis_result = {
            "bcell_metrics": bcell_analysis,
            "overall_specialization_ratio": overall_specialization,
            "total_specialist_experiences": total_specialist_exp,
            "total_general_experiences": total_general_exp,
        }

        return analysis_result

    def save_comprehensive_analysis(
        self,
        start_date: str,
        end_date: str,
        filename: str = None,
        output_dir: str = None,
    ):
        """통합 분석 결과 저장 (의사결정 분석 + 전문성 분석)"""

        if output_dir is None:
            output_dir = self.output_dir  # 전역 output_dir 사용

        if filename is None:
            filename = f"bipd_comprehensive_{start_date}_{end_date}"

        # 의사결정 분석
        decision_analysis = {}
        if hasattr(self, "immune_system") and hasattr(self.immune_system, "analyzer"):
            try:
                decision_analysis = (
                    self.immune_system.analyzer.generate_analysis_report(
                        start_date, end_date
                    )
                )
            except Exception as e:
                print(f"의사결정 분석 오류: {e}")
                decision_analysis = {"error": f"의사결정 분석 실패: {e}"}
        else:
            decision_analysis = {"error": "분석 시스템을 사용할 수 없습니다."}

        # 전문성 분석
        expertise_analysis = self.analyze_bcell_expertise()

        # 통합 데이터 구조
        comprehensive_data = {
            "metadata": {
                "analysis_timestamp": datetime.now().isoformat(),
                "analysis_period": {"start": start_date, "end": end_date},
                "system_type": (
                    "학습 기반"
                    if (
                        hasattr(self, "immune_system")
                        and self.immune_system.use_learning_bcells
                    )
                    else "규칙 기반"
                ),
            },
            "decision_analysis": decision_analysis,
            "expertise_analysis": expertise_analysis,
        }

        # JSON 저장
        json_path = os.path.join(output_dir, f"{filename}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(comprehensive_data, f, ensure_ascii=False, indent=2)

        # Markdown 저장
        md_content = self._generate_comprehensive_markdown(comprehensive_data)
        md_path = os.path.join(output_dir, f"{filename}.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        print(f"통합 분석 결과 저장 완료:")
        print(f"  디렉토리: {output_dir}")
        print(f"  JSON: {os.path.basename(json_path)}")
        print(f"  Markdown: {os.path.basename(md_path)}")

        return json_path, md_path

    def _generate_comprehensive_markdown(self, comprehensive_data: Dict) -> str:
        """통합 분석 마크다운 생성"""

        metadata = comprehensive_data["metadata"]
        decision_data = comprehensive_data["decision_analysis"]
        expertise_data = comprehensive_data["expertise_analysis"]

        md_content = f"""# BIPD 시스템 통합 분석 보고서

## 분석 메타데이터
- 분석 시간: {metadata['analysis_timestamp']}
- 시스템 유형: {metadata['system_type']}
- 분석 기간: {metadata['analysis_period']['start']} ~ {metadata['analysis_period']['end']}

---

"""

        # 의사결정 분석 섹션
        if "error" in decision_data:
            md_content += (
                f"## 의사결정 분석\n\n**오류:** {decision_data['error']}\n\n---\n\n"
            )
        else:
            period = decision_data["period"]
            stats = decision_data["basic_stats"]
            risk_dist = decision_data["risk_distribution"]
            efficiency = decision_data["system_efficiency"]

            md_content += f"""## 의사결정 분석

### 분석 기간
- 시작일: {period['start']}
- 종료일: {period['end']}

### 기본 통계
- 총 거래일: {stats['total_days']}일
- 위기 감지일: {stats['crisis_days']}일 ({stats['crisis_ratio']:.1%})
- 기억 세포 활성화: {stats['memory_activations']}일 ({stats['memory_activation_ratio']:.1%})
- 평균 일수익률: {stats['avg_daily_return']:+.3%}

### 위험 유형별 분포
"""

            for risk, count in sorted(
                risk_dist.items(), key=lambda x: x[1], reverse=True
            ):
                percentage = count / stats["total_days"] * 100
                md_content += f"- {risk}: {count}일 ({percentage:.1f}%)\n"

            md_content += f"""
### 시스템 효율성
- 위기 대응률: {efficiency['crisis_response_rate']:.1%}
- 학습 활성화율: {efficiency['learning_activation_rate']:.1%}
- 시스템 안정성: {efficiency['system_stability']}

---

"""

        # 전문성 분석 섹션
        if "error" in expertise_data:
            md_content += f"## 전문성 분석\n\n**오류:** {expertise_data['error']}\n\n"
        else:
            md_content += "## 전문성 분석\n\n"

            for bcell_metrics in expertise_data["bcell_metrics"]:
                md_content += f"### {bcell_metrics['risk_type'].upper()} 전문가\n"
                md_content += (
                    f"- 전문성 강도: {bcell_metrics['specialization_strength']:.3f}\n"
                )
                md_content += (
                    f"- 전문 경험: {bcell_metrics['specialist_experiences']}건\n"
                )
                md_content += f"- 일반 경험: {bcell_metrics['general_experiences']}건\n"
                md_content += (
                    f"- 전문화 비율: {bcell_metrics['specialization_ratio']:.1%}\n"
                )
                md_content += f"- 전문 분야 평균 보상: {bcell_metrics['specialist_avg_reward']:+.3f}\n"
                md_content += f"- 일반 분야 평균 보상: {bcell_metrics['general_avg_reward']:+.3f}\n"
                md_content += (
                    f"- 전문성 우위: {bcell_metrics['expertise_advantage']:+.3f}\n\n"
                )

            md_content += "### 전체 시스템 현황\n"
            md_content += f"- 전체 전문화 비율: {expertise_data['overall_specialization_ratio']:.1%}\n"
            md_content += (
                f"- 총 전문 경험: {expertise_data['total_specialist_experiences']}건\n"
            )
            md_content += (
                f"- 총 일반 경험: {expertise_data['total_general_experiences']}건\n"
            )

        return md_content

    def save_analysis_results(
        self, start_date: str, end_date: str, filename: str = None
    ):
        """분석 결과 저장 (HTML 대시보드 포함)"""

        if not hasattr(self, "immune_system") or not hasattr(
            self.immune_system, "analyzer"
        ):
            print("분석 시스템을 사용할 수 없습니다.")
            return None, None, None

        try:
            # 기존 JSON/MD 파일 생성
            json_path, md_path = self.immune_system.analyzer.save_analysis_to_file(
                start_date, end_date, filename
            )

            # 분석 보고서 데이터 가져오기
            analysis_report = self.immune_system.analyzer.generate_analysis_report(
                start_date, end_date
            )

            # HTML 대시보드 생성
            dashboard_paths = generate_dashboard(
                analysis_report,
                output_dir=os.path.dirname(json_path) if json_path else ".",
            )

            # 면역 시스템 시각화 생성
            immune_viz = create_visualizations(
                self,
                start_date,
                end_date,
                output_dir=os.path.dirname(json_path) if json_path else ".",
            )

            print(f"분석 결과 저장 완료:")
            print(f"  JSON: {json_path}")
            print(f"  Markdown: {md_path}")
            print(f"  HTML Dashboard: {dashboard_paths['html_dashboard']}")
            print(
                f"\nHTML 대시보드에서 T-Cell/B-Cell 판단 근거를 직관적으로 확인할 수 있습니다!"
            )
            print(
                f"면역 시스템 반응 패턴 시각화로 기존 연구와의 차별점을 강조할 수 있습니다!"
            )

            return json_path, md_path, dashboard_paths["html_dashboard"]

        except Exception as e:
            print(f"분석 결과 저장 오류: {e}")
            return None, None, None

    def save_expertise_analysis(self, filename: str = None):
        """전문성 분석 결과 저장"""

        expertise_data = self.analyze_bcell_expertise()

        if "error" in expertise_data:
            print(expertise_data["error"])
            return None

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"expertise_analysis_{timestamp}"

        # JSON 저장
        json_path = os.path.join(self.output_dir, f"{filename}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(expertise_data, f, ensure_ascii=False, indent=2)

        # Markdown 저장
        md_content = self._generate_expertise_markdown(expertise_data)
        md_path = os.path.join(self.output_dir, f"{filename}.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        print(f"전문성 분석 저장 완료:")
        print(f"  JSON: {json_path}")
        print(f"  Markdown: {md_path}")

        return json_path, md_path

    def _generate_expertise_markdown(self, expertise_data: Dict) -> str:
        """전문성 분석 마크다운 생성"""

        md_content = "# B-세포 전문성 분석 보고서\n\n"

        for bcell_metrics in expertise_data["bcell_metrics"]:
            md_content += f"## {bcell_metrics['risk_type'].upper()} 전문가\n"
            md_content += (
                f"- 전문성 강도: {bcell_metrics['specialization_strength']:.3f}\n"
            )
            md_content += f"- 전문 경험: {bcell_metrics['specialist_experiences']}건\n"
            md_content += f"- 일반 경험: {bcell_metrics['general_experiences']}건\n"
            md_content += (
                f"- 전문화 비율: {bcell_metrics['specialization_ratio']:.1%}\n"
            )
            md_content += f"- 전문 분야 평균 보상: {bcell_metrics['specialist_avg_reward']:+.3f}\n"
            md_content += (
                f"- 일반 분야 평균 보상: {bcell_metrics['general_avg_reward']:+.3f}\n"
            )
            md_content += (
                f"- 전문성 우위: {bcell_metrics['expertise_advantage']:+.3f}\n\n"
            )

        md_content += "## 전체 시스템 현황\n"
        md_content += f"- 전체 전문화 비율: {expertise_data['overall_specialization_ratio']:.1%}\n"
        md_content += (
            f"- 총 전문 경험: {expertise_data['total_specialist_experiences']}건\n"
        )
        md_content += (
            f"- 총 일반 경험: {expertise_data['total_general_experiences']}건\n"
        )

        return md_content

    def save_model(self, immune_system, filename=None, output_dir=None):
        """모델 저장"""
        if output_dir is None:
            output_dir = os.path.join(self.output_dir, "models")
            os.makedirs(output_dir, exist_ok=True)

        if filename is None:
            if immune_system.use_learning_bcells:
                filename = "immune_system"
            else:
                filename = "legacy_immune_system"

        if immune_system.use_learning_bcells:
            model_dir = os.path.join(output_dir, filename)
            os.makedirs(model_dir, exist_ok=True)

            # B-세포 신경망 저장
            for i, bcell in enumerate(immune_system.bcells):
                if hasattr(bcell, "strategy_network"):
                    network_path = os.path.join(
                        model_dir, f"bcell_{i}_{bcell.risk_type}.pth"
                    )
                    torch.save(bcell.strategy_network.state_dict(), network_path)

            # 시스템 상태 저장
            system_state = {
                "n_assets": immune_system.n_assets,
                "base_weights": immune_system.base_weights,
                "memory_cell": immune_system.memory_cell,
                "tcells": immune_system.tcells,
                "use_learning_bcells": True,
            }
            state_path = os.path.join(model_dir, "system_state.pkl")
            with open(state_path, "wb") as f:
                pickle.dump(system_state, f)

            print(f"학습 기반 모델 저장 완료: {model_dir}")
            return model_dir
        else:
            model_path = os.path.join(output_dir, f"{filename}.pkl")
            with open(model_path, "wb") as f:
                pickle.dump(immune_system, f)
            print(f"규칙 기반 모델 저장 완료: {model_path}")
            return model_path

    def save_results(self, metrics_df, filename=None, output_dir=None):
        """결과 저장"""
        if output_dir is None:
            output_dir = self.output_dir  # 전역 output_dir 사용

        if filename is None:
            filename = "bipd_performance_metrics"

        # CSV 저장
        csv_path = os.path.join(output_dir, f"{filename}.csv")
        metrics_df.to_csv(csv_path, index=False)

        # 시각화
        plt.figure(figsize=(15, 10))

        plt.subplot(2, 3, 1)
        metrics_df.boxplot(column=["Total Return"], ax=plt.gca())
        plt.title("Total Return Distribution")

        plt.subplot(2, 3, 2)
        metrics_df.boxplot(column=["Sharpe Ratio"], ax=plt.gca())
        plt.title("Sharpe Ratio Distribution")

        plt.subplot(2, 3, 3)
        metrics_df.boxplot(column=["Max Drawdown"], ax=plt.gca())
        plt.title("Max Drawdown Distribution")

        plt.subplot(2, 2, 3)
        correlation = metrics_df.corr()
        plt.imshow(correlation, cmap="coolwarm", aspect="auto")
        plt.colorbar()
        plt.title("Metrics Correlation")
        plt.xticks(range(len(correlation.columns)), correlation.columns, rotation=45)
        plt.yticks(range(len(correlation.columns)), correlation.columns)

        plt.subplot(2, 2, 4)
        plt.axis("off")
        summary_text = f"""
BIPD 백테스트 결과 요약

총 수익률: {metrics_df['Total Return'].mean():.2%}
표준편차: {metrics_df['Volatility'].mean():.3f}
최대 낙폭: {metrics_df['Max Drawdown'].mean():.2%}
Sharpe Ratio: {metrics_df['Sharpe Ratio'].mean():.2f}
Calmar Ratio: {metrics_df['Calmar Ratio'].mean():.2f}
초기 자본: {metrics_df['Initial Capital'].iloc[0]:,.0f}원
최종 자본: {metrics_df['Final Value'].mean():,.0f}원
        """
        plt.text(0.1, 0.5, summary_text, fontsize=10, verticalalignment="center")

        plt.tight_layout()
        plot_path = os.path.join(output_dir, f"{filename}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"백테스트 결과 저장 완료:")
        print(f"  디렉토리: {output_dir}")
        print(f"  CSV: {os.path.basename(csv_path)}")
        print(f"  그래프: {os.path.basename(plot_path)}")
        return csv_path, plot_path

    def run_multiple_backtests(
        self,
        n_runs=10,
        save_results=True,
        use_learning_bcells=True,
        logging_level="sample",
        base_seed=None,
    ):
        """다중 백테스트 실행"""
        all_metrics = []
        best_immune_system = None
        best_sharpe = -np.inf

        print(f"\n=== BIPD 시스템 다중 백테스트 ({n_runs}회) 실행 ===")
        if use_learning_bcells:
            print("시스템 유형: 적응형 신경망 기반 BIPD 모델")
        else:
            print("시스템 유형: 규칙 기반 레거시 BIPD 모델")

        # 시드 설정
        if base_seed is None:
            import time

            base_seed = int(time.time()) % 10000

        print(f"[설정] 기본 시드: {base_seed}")

        for run in range(n_runs):
            run_seed = base_seed + run * 1000  # 각 실행마다 다른 시드
            print(f"\n{run + 1}/{n_runs}번째 실행 (시드: {run_seed})")

            portfolio_returns, immune_system = self.backtest_single_run(
                seed=run_seed,
                return_model=True,
                use_learning_bcells=use_learning_bcells,
                logging_level=logging_level,
            )
            metrics = self.calculate_metrics(portfolio_returns)
            all_metrics.append(metrics)

            if metrics["Sharpe Ratio"] > best_sharpe:
                best_sharpe = metrics["Sharpe Ratio"]
                best_immune_system = immune_system

        metrics_df = pd.DataFrame(all_metrics)

        system_type = "학습 기반" if use_learning_bcells else "규칙 기반"
        print(f"\n=== {system_type} 모델 성능 요약 ({n_runs}회 실행 평균) ===")
        print(f"총 수익률: {metrics_df['Total Return'].mean():.2%}")
        print(f"연평균 변동성: {metrics_df['Volatility'].mean():.3f}")
        print(f"최대 낙폭: {metrics_df['Max Drawdown'].mean():.2%}")
        print(f"샤프 지수: {metrics_df['Sharpe Ratio'].mean():.2f}")
        print(f"칼마 지수: {metrics_df['Calmar Ratio'].mean():.2f}")
        print(f"최종 자산: {metrics_df['Final Value'].mean():,.0f}원")

        if save_results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_filename = f"bipd_{system_type}_{timestamp}"
            self.save_results(metrics_df, result_filename)

            if best_immune_system is not None:
                if use_learning_bcells:
                    model_filename = f"best_immune_system_{timestamp}"
                else:
                    model_filename = "best_legacy_immune_system.pkl"
                self.save_model(best_immune_system, model_filename)

        return metrics_df


# 실행
if __name__ == "__main__":
    # 설정
    symbols = ["AAPL", "MSFT", "AMZN", "GOOGL", "AMD", "TSLA", "JPM", "JNJ", "PG", "V"]
    train_start = "2008-01-02"
    train_end = "2020-12-31"
    test_start = "2021-01-01"
    test_end = "2024-12-31"

    # 시드 설정 옵션
    USE_FIXED_SEED = False  # True: 재현 가능한 결과, False: 매번 다른 결과

    if USE_FIXED_SEED:
        global_seed = 42
        print(f"[설정] 고정 시드 사용: {global_seed} (재현 가능한 결과)")
    else:
        import time

        global_seed = int(time.time()) % 10000
        print(f"[설정] 랜덤 시드 사용: {global_seed} (매번 다른 결과)")

    np.random.seed(global_seed)
    torch.manual_seed(global_seed)

    # 백테스터 초기화
    backtester = ImmunePortfolioBacktester(
        symbols, train_start, train_end, test_start, test_end
    )

    print("\n" + "=" * 60)
    print(" BIPD (Behavioral Immune Portfolio Defense) 시스템 성능 평가")
    print("=" * 60)

    try:
        # 백테스트 실행 (전역 시드 사용)
        portfolio_returns, immune_system = backtester.backtest_single_run(
            seed=global_seed,
            return_model=True,
            use_learning_bcells=True,
            logging_level="full",
        )

        # 시스템 저장
        backtester.immune_system = immune_system

        # 성과 계산
        metrics = backtester.calculate_metrics(portfolio_returns)
        print(f"\n=== 포트폴리오 성과 요약 ===")
        print(f"총 수익률: {metrics['Total Return']:.2%}")
        print(f"샤프 지수: {metrics['Sharpe Ratio']:.2f}")
        print(f"최대 낙폭: {metrics['Max Drawdown']:.2%}")
        print(f"변동성: {metrics['Volatility']:.3f}")

        # 분석 결과 저장
        print(f"\n=== 분석 결과 저장 중 ===")

        # 통합 분석 (의사결정 분석 + 전문성 분석)
        json_path, md_path = backtester.save_comprehensive_analysis(
            "2021-01-01", "2021-06-30"
        )

        # 분석 결과 저장 (HTML 대시보드 + 면역 시스템 시각화)
        analysis_json, analysis_md, dashboard_html = backtester.save_analysis_results(
            "2021-01-01", "2021-06-30"
        )

        print(f"\n=== BIPD 시스템 성능 평가 완료 ===")

        # 다중 실행 성능 검증 (다양한 시드로 안정성 확인)
        print(f"\n=== 다중 실행 안정성 검증 ===")
        results = backtester.run_multiple_backtests(
            n_runs=3,
            save_results=True,
            use_learning_bcells=True,
            logging_level="sample",
            base_seed=global_seed,
        )

    except Exception as e:
        print(f"\n[오류] 주요 실행 실패: {e}")
        import traceback

        traceback.print_exc()

        # 폴백 모드: 기본 백테스트 (최소 로깅으로 성능 확보)
        basic_results = backtester.run_multiple_backtests(
            n_runs=1,
            save_results=True,
            use_learning_bcells=True,
            logging_level="minimal",
            base_seed=global_seed,
        )
