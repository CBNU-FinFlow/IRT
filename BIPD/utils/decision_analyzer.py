"""
의사결정 분석 시스템
T-Cell과 B-Cell의 의사결정 과정을 상세히 분석하고 로깅
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Any
import json
import os


class DecisionAnalyzer:
    """
    의사결정 과정 분석 및 로깅 시스템
    """
    
    def __init__(self, output_dir: str = "analysis_logs"):
        self.decision_log = []
        self.crisis_detection_log = []
        self.output_dir = output_dir
        
        # 리스크 임계값 설정
        self.risk_thresholds = {
            "low": 0.3,
            "medium": 0.5, 
            "high": 0.7,
            "critical": 0.9
        }
        
        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
    
    def log_decision(self,
                    date: datetime,
                    market_features: np.ndarray,
                    tcell_analysis: Dict[str, Any],
                    bcell_decisions: Dict[str, Any],
                    final_weights: np.ndarray,
                    portfolio_return: float,
                    crisis_level: float):
        """
        의사결정 과정 기록
        
        Args:
            date: 의사결정 시점
            market_features: 시장 특성 벡터
            tcell_analysis: T-Cell 분석 결과
            bcell_decisions: B-Cell 의사결정 결과
            final_weights: 최종 포트폴리오 가중치
            portfolio_return: 포트폴리오 수익률
            crisis_level: 위기 수준
        """
        # 지배적 위험 분석
        risk_features = market_features[:5] if len(market_features) >= 5 else market_features
        dominant_risk_idx = np.argmax(np.abs(risk_features - np.mean(risk_features)))
        
        risk_map = {
            0: "volatility",
            1: "correlation", 
            2: "momentum",
            3: "liquidity",
            4: "macro"
        }
        dominant_risk = risk_map.get(dominant_risk_idx, "volatility")
        
        # T-Cell 분석 처리
        tcell_analysis_result = self._process_detailed_tcell_analysis(
            tcell_analysis, dominant_risk, risk_features, dominant_risk_idx
        )
        
        # 의사결정 기록 생성
        decision_record = {
            "date": date.strftime("%Y-%m-%d") if hasattr(date, "strftime") else str(date),
            "market_features": market_features.tolist() if hasattr(market_features, "tolist") else list(market_features),
            "tcell_analysis": tcell_analysis_result,
            "bcell_decisions": self._serialize_bcell_decisions(bcell_decisions),
            "final_weights": final_weights.tolist() if hasattr(final_weights, "tolist") else list(final_weights),
            "portfolio_return": float(portfolio_return),
            "crisis_level": float(crisis_level),
            "memory_activated": bool(crisis_level > 0.3),
            "risk_assessment": {
                "dominant_risk": dominant_risk,
                "risk_intensity": float(risk_features[dominant_risk_idx]),
                "threat_level": self._assess_threat_level(crisis_level)
            }
        }
        
        self.decision_log.append(decision_record)
    
    def _process_detailed_tcell_analysis(self,
                                       tcell_analysis: Dict[str, Any],
                                       dominant_risk: str,
                                       risk_features: np.ndarray,
                                       dominant_risk_idx: int) -> Dict[str, Any]:
        """
        상세한 T-Cell 분석 처리
        """
        # 기본 T-Cell 분석 정보
        basic_analysis = {
            "crisis_level": float(tcell_analysis.get("crisis_level", 0.0)),
            "dominant_risk": dominant_risk,
            "risk_intensity": float(risk_features[dominant_risk_idx]),
            "overall_threat": self._assess_threat_level(tcell_analysis.get("crisis_level", 0.0)),
            "activation_level": float(tcell_analysis.get("activation_level", 0.0)),
            "decision_reasoning": tcell_analysis.get("decision_reasoning", ""),
            "crisis_classification": tcell_analysis.get("crisis_classification", "normal")
        }
        
        # 상세 위기 감지 로그가 있는 경우 추가
        if isinstance(tcell_analysis, dict) and "detailed_crisis_logs" in tcell_analysis:
            detailed_logs = tcell_analysis["detailed_crisis_logs"]
            analysis = basic_analysis.copy()
            analysis["detailed_crisis_detection"] = {
                "active_tcells": len(detailed_logs),
                "crisis_detections": []
            }
            
            for tcell_log in detailed_logs:
                if tcell_log.get("activation_level", 0.0) > 0.15:  # 위기 감지 임계값
                    crisis_detection = {
                        "tcell_id": tcell_log.get("tcell_id", "unknown"),
                        "timestamp": tcell_log.get("timestamp", ""),
                        "activation_level": tcell_log.get("activation_level", 0.0),
                        "crisis_level_classification": tcell_log.get("crisis_level", "normal"),
                        "crisis_indicators": tcell_log.get("crisis_indicators", []),
                        "decision_reasoning": tcell_log.get("decision_reasoning", []),
                        "feature_contributions": tcell_log.get("feature_contributions", {}),
                        "market_state_analysis": tcell_log.get("market_state", {})
                    }
                    analysis["detailed_crisis_detection"]["crisis_detections"].append(crisis_detection)
                    
                    # 위기 감지 로그에 추가
                    self.crisis_detection_log.append({
                        "timestamp": tcell_log.get("timestamp", ""),
                        "tcell_id": tcell_log.get("tcell_id", "unknown"),
                        "crisis_info": crisis_detection
                    })
            
            return analysis
        
        return basic_analysis
    
    def _serialize_bcell_decisions(self, bcell_decisions: Dict[str, Any]) -> Dict[str, Any]:
        """
        B-Cell 의사결정 결과 직렬화
        """
        serialized = {}
        
        for cell_type, decision in bcell_decisions.items():
            if isinstance(decision, dict):
                serialized[cell_type] = {
                    "is_specialty_situation": decision.get("is_specialty_situation", False),
                    "activation_strength": float(decision.get("activation_strength", 0.0)),
                    "strategy_description": decision.get("strategy_description", ""),
                    "reasoning": decision.get("reasoning", ""),
                    "portfolio_weights": decision.get("portfolio_weights", []),
                    "cell_id": decision.get("cell_id", f"{cell_type}-cell")
                }
            else:
                serialized[cell_type] = {
                    "activation_strength": 0.0,
                    "is_specialty_situation": False,
                    "strategy_description": f"{cell_type} 전략",
                    "reasoning": "정보 불충분"
                }
        
        return serialized
    
    def _assess_threat_level(self, crisis_level: float) -> str:
        """
        위기 수준을 위협 등급으로 변환
        """
        if crisis_level >= self.risk_thresholds["critical"]:
            return "critical"
        elif crisis_level >= self.risk_thresholds["high"]:
            return "high"
        elif crisis_level >= self.risk_thresholds["medium"]:
            return "medium"
        elif crisis_level >= self.risk_thresholds["low"]:
            return "low"
        else:
            return "minimal"
    
    def generate_decision_summary(self, 
                                start_date: str = None,
                                end_date: str = None) -> Dict[str, Any]:
        """
        의사결정 요약 생성
        
        Args:
            start_date: 분석 시작 날짜
            end_date: 분석 종료 날짜
            
        Returns:
            의사결정 요약 딕셔너리
        """
        if not self.decision_log:
            return {"error": "의사결정 로그가 없습니다"}
        
        # 날짜 필터링
        filtered_logs = self.decision_log
        if start_date or end_date:
            filtered_logs = []
            for log in self.decision_log:
                log_date = log["date"]
                if start_date and log_date < start_date:
                    continue
                if end_date and log_date > end_date:
                    continue
                filtered_logs.append(log)
        
        if not filtered_logs:
            return {"error": "해당 기간에 의사결정 로그가 없습니다"}
        
        # 통계 계산
        crisis_levels = [log["crisis_level"] for log in filtered_logs]
        portfolio_returns = [log["portfolio_return"] for log in filtered_logs]
        
        # T-Cell 활성화 통계
        tcell_activations = [
            log["tcell_analysis"]["activation_level"] 
            for log in filtered_logs 
            if "activation_level" in log["tcell_analysis"]
        ]
        
        # B-Cell 활성화 통계
        bcell_activations = {}
        for log in filtered_logs:
            for cell_type, decision in log["bcell_decisions"].items():
                if cell_type not in bcell_activations:
                    bcell_activations[cell_type] = []
                bcell_activations[cell_type].append(decision["activation_strength"])
        
        # 위기 분류 통계
        threat_levels = [log["tcell_analysis"]["overall_threat"] for log in filtered_logs]
        threat_distribution = {}
        for level in threat_levels:
            threat_distribution[level] = threat_distribution.get(level, 0) + 1
        
        # 지배적 위험 분석
        dominant_risks = [log["risk_assessment"]["dominant_risk"] for log in filtered_logs]
        risk_distribution = {}
        for risk in dominant_risks:
            risk_distribution[risk] = risk_distribution.get(risk, 0) + 1
        
        return {
            "analysis_period": {
                "start_date": start_date,
                "end_date": end_date,
                "total_decisions": len(filtered_logs)
            },
            "crisis_statistics": {
                "avg_crisis_level": np.mean(crisis_levels),
                "max_crisis_level": np.max(crisis_levels),
                "crisis_episodes": len([c for c in crisis_levels if c > 0.5])
            },
            "performance_statistics": {
                "avg_return": np.mean(portfolio_returns),
                "volatility": np.std(portfolio_returns),
                "sharpe_ratio": np.mean(portfolio_returns) / np.std(portfolio_returns) if np.std(portfolio_returns) > 0 else 0,
                "positive_returns": len([r for r in portfolio_returns if r > 0]) / len(portfolio_returns)
            },
            "tcell_statistics": {
                "avg_activation": np.mean(tcell_activations) if tcell_activations else 0,
                "max_activation": np.max(tcell_activations) if tcell_activations else 0,
                "activation_frequency": len([a for a in tcell_activations if a > 0.15]) / len(tcell_activations) if tcell_activations else 0
            },
            "bcell_statistics": {
                cell_type: {
                    "avg_activation": np.mean(activations),
                    "max_activation": np.max(activations),
                    "activation_frequency": len([a for a in activations if a > 0.1]) / len(activations)
                }
                for cell_type, activations in bcell_activations.items()
            },
            "threat_distribution": threat_distribution,
            "risk_distribution": risk_distribution,
            "memory_activation_rate": len([log for log in filtered_logs if log["memory_activated"]]) / len(filtered_logs)
        }
    
    def save_analysis_report(self, 
                           filename: str = None,
                           start_date: str = None,
                           end_date: str = None) -> str:
        """
        분석 보고서 저장
        
        Args:
            filename: 저장할 파일명
            start_date: 분석 시작 날짜
            end_date: 분석 종료 날짜
            
        Returns:
            저장된 파일 경로
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"decision_analysis_{timestamp}.json"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # 분석 보고서 생성
        report = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "analysis_period": {
                    "start_date": start_date,
                    "end_date": end_date
                },
                "total_decisions": len(self.decision_log),
                "crisis_detections": len(self.crisis_detection_log)
            },
            "summary": self.generate_decision_summary(start_date, end_date),
            "detailed_decisions": self.decision_log,
            "crisis_detection_log": self.crisis_detection_log
        }
        
        # JSON 파일로 저장
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"의사결정 분석 보고서 저장 완료: {filepath}")
        return filepath
    
    def get_crisis_episodes(self, threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        위기 에피소드 추출
        
        Args:
            threshold: 위기 임계값
            
        Returns:
            위기 에피소드 리스트
        """
        crisis_episodes = []
        current_episode = None
        
        for log in self.decision_log:
            if log["crisis_level"] > threshold:
                if current_episode is None:
                    # 새로운 위기 에피소드 시작
                    current_episode = {
                        "start_date": log["date"],
                        "end_date": log["date"],
                        "max_crisis_level": log["crisis_level"],
                        "duration": 1,
                        "decisions": [log],
                        "avg_return": log["portfolio_return"]
                    }
                else:
                    # 기존 에피소드 확장
                    current_episode["end_date"] = log["date"]
                    current_episode["max_crisis_level"] = max(current_episode["max_crisis_level"], log["crisis_level"])
                    current_episode["duration"] += 1
                    current_episode["decisions"].append(log)
                    current_episode["avg_return"] = np.mean([d["portfolio_return"] for d in current_episode["decisions"]])
            else:
                if current_episode is not None:
                    # 위기 에피소드 종료
                    crisis_episodes.append(current_episode)
                    current_episode = None
        
        # 마지막 에피소드 처리
        if current_episode is not None:
            crisis_episodes.append(current_episode)
        
        return crisis_episodes
    
    def get_bcell_specialization_analysis(self) -> Dict[str, Any]:
        """
        B-Cell 전문화 분석
        
        Returns:
            B-Cell 전문화 분석 결과
        """
        if not self.decision_log:
            return {"error": "의사결정 로그가 없습니다"}
        
        specialization_stats = {}
        
        # 각 B-Cell 타입별 분석
        for log in self.decision_log:
            for cell_type, decision in log["bcell_decisions"].items():
                if cell_type not in specialization_stats:
                    specialization_stats[cell_type] = {
                        "total_activations": 0,
                        "specialty_activations": 0,
                        "avg_activation_strength": 0,
                        "max_activation_strength": 0,
                        "activation_returns": []
                    }
                
                stats = specialization_stats[cell_type]
                activation_strength = decision["activation_strength"]
                
                if activation_strength > 0:
                    stats["total_activations"] += 1
                    stats["activation_returns"].append(log["portfolio_return"])
                
                if decision["is_specialty_situation"]:
                    stats["specialty_activations"] += 1
                
                stats["avg_activation_strength"] += activation_strength
                stats["max_activation_strength"] = max(stats["max_activation_strength"], activation_strength)
        
        # 통계 계산 완료
        total_decisions = len(self.decision_log)
        for cell_type, stats in specialization_stats.items():
            stats["avg_activation_strength"] /= total_decisions
            stats["activation_rate"] = stats["total_activations"] / total_decisions
            stats["specialization_rate"] = stats["specialty_activations"] / total_decisions
            
            if stats["activation_returns"]:
                stats["avg_activation_return"] = np.mean(stats["activation_returns"])
                stats["activation_sharpe"] = np.mean(stats["activation_returns"]) / np.std(stats["activation_returns"]) if np.std(stats["activation_returns"]) > 0 else 0
            else:
                stats["avg_activation_return"] = 0
                stats["activation_sharpe"] = 0
        
        return specialization_stats
    
    def clear_logs(self):
        """로그 초기화"""
        self.decision_log.clear()
        self.crisis_detection_log.clear()
        print("의사결정 로그가 초기화되었습니다.")
    
    def export_logs(self, filepath: str):
        """로그 내보내기"""
        export_data = {
            "decision_log": self.decision_log,
            "crisis_detection_log": self.crisis_detection_log,
            "export_timestamp": datetime.now().isoformat()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"로그 내보내기 완료: {filepath}")
    
    def import_logs(self, filepath: str):
        """로그 가져오기"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                import_data = json.load(f)
            
            self.decision_log = import_data.get("decision_log", [])
            self.crisis_detection_log = import_data.get("crisis_detection_log", [])
            
            print(f"로그 가져오기 완료: {len(self.decision_log)}개 의사결정, {len(self.crisis_detection_log)}개 위기 감지")
            
        except Exception as e:
            print(f"로그 가져오기 실패: {str(e)}")