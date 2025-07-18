"""
HTML 대시보드 생성기 - T-Cell과 B-Cell 판단 근거 시각화
"""

import json
import os
from datetime import datetime
from typing import Dict, List
import numpy as np


class HTMLDashboardGenerator:
    """HTML 대시보드 생성기"""
    
    def __init__(self):
        self.template = self._load_template()
    
    def generate_dashboard(self, analysis_report: Dict, output_path: str = None):
        """분석 보고서를 기반으로 HTML 대시보드 생성"""
        
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"dashboard_{timestamp}.html"
        
        # 데이터 처리
        dashboard_data = self._process_analysis_data(analysis_report)
        
        # HTML 생성
        html_content = self._generate_html(dashboard_data)
        
        # 파일 저장
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"HTML dashboard generated: {output_path}")
        return output_path
    
    def _process_analysis_data(self, report: Dict) -> Dict:
        """분석 데이터 처리"""
        
        # 기본 통계
        basic_stats = report.get("basic_stats", {})
        
        # T-Cell 분석
        tcell_data = report.get("tcell_analysis", {})
        tcell_crisis_events = tcell_data.get("crisis_events", [])
        
        # B-Cell 분석
        bcell_data = report.get("bcell_analysis", {})
        expert_responses = bcell_data.get("expert_responses", [])
        
        # 특성 기여도
        feature_attribution = report.get("feature_attribution", {})
        
        # 시간별 분석
        temporal_analysis = report.get("temporal_analysis", {})
        
        return {
            "period": report.get("period", {}),
            "basic_stats": basic_stats,
            "tcell_insights": self._process_tcell_insights(tcell_crisis_events),
            "bcell_insights": self._process_bcell_insights(expert_responses),
            "feature_importance": self._process_feature_importance(feature_attribution),
            "temporal_patterns": self._process_temporal_patterns(temporal_analysis),
            "risk_distribution": basic_stats.get("risk_distribution", {}),
            "xai_explanations": self._generate_xai_explanations(report)
        }
    
    def _process_tcell_insights(self, crisis_events: List) -> List[Dict]:
        """T-Cell 위기 감지 인사이트 처리"""
        insights = []
        
        for event in crisis_events[:10]:  # 최대 10개 이벤트
            insight = {
                "date": event.get("date", ""),
                "crisis_level": event.get("crisis_level", 0),
                "detected_risks": event.get("detected_risks", []),
                "feature_scores": event.get("feature_scores", {}),
                "explanation": self._generate_tcell_explanation(event),
                "severity": self._get_severity_level(event.get("crisis_level", 0))
            }
            insights.append(insight)
        
        return insights
    
    def _process_bcell_insights(self, expert_responses: List) -> List[Dict]:
        """B-Cell 전문가 응답 인사이트 처리"""
        insights = []
        
        for response in expert_responses[:10]:  # 최대 10개 응답
            insight = {
                "date": response.get("date", ""),
                "expert_type": response.get("expert_type", ""),
                "confidence": response.get("confidence", 0),
                "recommendation": response.get("recommendation", ""),
                "reasoning": response.get("reasoning", []),
                "market_context": response.get("market_context", {}),
                "explanation": self._generate_bcell_explanation(response)
            }
            insights.append(insight)
        
        return insights
    
    def _process_feature_importance(self, feature_attribution: Dict) -> List[Dict]:
        """특성 중요도 처리"""
        top_features = feature_attribution.get("top_features", [])
        
        return [
            {
                "feature": feature.get("name", ""),
                "importance": feature.get("importance", 0),
                "impact": feature.get("impact", ""),
                "explanation": self._generate_feature_explanation(feature)
            }
            for feature in top_features[:15]  # 상위 15개 특성
        ]
    
    def _process_temporal_patterns(self, temporal_analysis: Dict) -> Dict:
        """시간별 패턴 처리"""
        return {
            "crisis_progression": temporal_analysis.get("crisis_progression", []),
            "market_cycles": temporal_analysis.get("market_cycles", {}),
            "prediction_accuracy": temporal_analysis.get("prediction_accuracy", 0)
        }
    
    def _generate_xai_explanations(self, report: Dict) -> List[Dict]:
        """XAI 설명 생성"""
        explanations = []
        
        # T-Cell 설명
        tcell_explanation = {
            "component": "T-Cell (위기 감지)",
            "decision_process": "시장 이상 탐지 → 위기 수준 평가 → 위험 요소 식별",
            "key_insights": [
                f"총 {report.get('basic_stats', {}).get('crisis_days', 0)}일의 위기 상황 감지",
                f"주요 위험 요소: {list(report.get('basic_stats', {}).get('risk_distribution', {}).keys())[:3]}",
                "적응적 임계값 조정으로 거짓 양성 최소화"
            ],
            "methodology": "Isolation Forest 기반 이상 탐지 + 동적 임계값 조정"
        }
        explanations.append(tcell_explanation)
        
        # B-Cell 설명
        bcell_explanation = {
            "component": "B-Cell (전문가 시스템)",
            "decision_process": "시장 상황 분석 → 전문가 모델 선택 → 투자 전략 추천",
            "key_insights": [
                f"평균 신뢰도: {report.get('bcell_analysis', {}).get('avg_confidence', 0):.2f}",
                f"메모리 활성화: {report.get('basic_stats', {}).get('memory_activations', 0)}회",
                "상황별 전문가 모델 적응적 선택"
            ],
            "methodology": "앙상블 전문가 모델 + 메모리 기반 학습"
        }
        explanations.append(bcell_explanation)
        
        return explanations
    
    def _generate_tcell_explanation(self, event: Dict) -> str:
        """T-Cell 이벤트 설명 생성"""
        crisis_level = event.get("crisis_level", 0)
        risks = event.get("detected_risks", [])
        
        if crisis_level > 0.8:
            severity = "심각한 위기"
        elif crisis_level > 0.5:
            severity = "중간 위기"
        else:
            severity = "경미한 위기"
        
        risk_text = ", ".join(risks[:3]) if risks else "일반적 시장 불안정"
        
        return f"{severity} 감지: {risk_text}로 인한 시장 이상 신호 포착"
    
    def _generate_bcell_explanation(self, response: Dict) -> str:
        """B-Cell 응답 설명 생성"""
        expert_type = response.get("expert_type", "")
        confidence = response.get("confidence", 0)
        recommendation = response.get("recommendation", "")
        
        confidence_text = "높은" if confidence > 0.7 else "중간" if confidence > 0.4 else "낮은"
        
        return f"{expert_type} 전문가가 {confidence_text} 신뢰도로 '{recommendation}' 전략 추천"
    
    def _generate_feature_explanation(self, feature: Dict) -> str:
        """특성 설명 생성"""
        name = feature.get("name", "")
        importance = feature.get("importance", 0)
        impact = feature.get("impact", "")
        
        importance_text = "매우 중요" if importance > 0.7 else "중요" if importance > 0.4 else "보통"
        
        return f"{name}: {importance_text}한 특성으로 {impact} 영향"
    
    def _get_severity_level(self, crisis_level: float) -> str:
        """위기 심각도 레벨"""
        if crisis_level > 0.8:
            return "critical"
        elif crisis_level > 0.5:
            return "high"
        elif crisis_level > 0.3:
            return "medium"
        else:
            return "low"
    
    def _generate_html(self, data: Dict) -> str:
        """HTML 생성"""
        return f"""
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BIPD 면역 시스템 분석 대시보드</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        {self._get_css_styles()}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>BIPD 면역 시스템 분석 대시보드</h1>
            <p>기간: {data['period'].get('start', '')} ~ {data['period'].get('end', '')}</p>
        </header>
        
        <div class="summary-cards">
            {self._generate_summary_cards(data)}
        </div>
        
        <div class="main-content">
            <div class="left-panel">
                <section class="tcell-section">
                    <h2>T-Cell 위기 감지 분석</h2>
                    {self._generate_tcell_section(data)}
                </section>
                
                <section class="bcell-section">
                    <h2>B-Cell 전문가 판단</h2>
                    {self._generate_bcell_section(data)}
                </section>
            </div>
            
            <div class="right-panel">
                <section class="feature-section">
                    <h2>특성 중요도 분석</h2>
                    {self._generate_feature_section(data)}
                </section>
                
                <section class="xai-section">
                    <h2>XAI 설명</h2>
                    {self._generate_xai_section(data)}
                </section>
            </div>
        </div>
        
        <section class="temporal-section">
            <h2>시간별 패턴 분석</h2>
            {self._generate_temporal_section(data)}
        </section>
    </div>
    
    <script>
        {self._get_javascript()}
    </script>
</body>
</html>
        """
    
    def _generate_summary_cards(self, data: Dict) -> str:
        """요약 카드 생성"""
        stats = data.get('basic_stats', {})
        
        return f"""
        <div class="card">
            <h3>총 분석 기간</h3>
            <div class="metric">{stats.get('total_days', 0)}일</div>
        </div>
        <div class="card">
            <h3>위기 감지 일수</h3>
            <div class="metric crisis">{stats.get('crisis_days', 0)}일</div>
        </div>
        <div class="card">
            <h3>위기 비율</h3>
            <div class="metric">{stats.get('crisis_ratio', 0):.1%}</div>
        </div>
        <div class="card">
            <h3>메모리 활성화</h3>
            <div class="metric">{stats.get('memory_activations', 0)}회</div>
        </div>
        """
    
    def _generate_tcell_section(self, data: Dict) -> str:
        """T-Cell 섹션 생성"""
        insights = data.get('tcell_insights', [])
        
        html = "<div class='insights-container'>"
        
        for insight in insights[:5]:  # 최대 5개 표시
            severity_class = insight.get('severity', 'low')
            html += f"""
            <div class="insight-card tcell-card {severity_class}">
                <div class="insight-header">
                    <span class="date">{insight.get('date', '')}</span>
                    <span class="severity {severity_class}">{insight.get('crisis_level', 0):.3f}</span>
                </div>
                <div class="insight-content">
                    <p><strong>감지된 위험:</strong> {', '.join(insight.get('detected_risks', [])[:3])}</p>
                    <p class="explanation">{insight.get('explanation', '')}</p>
                </div>
            </div>
            """
        
        html += "</div>"
        return html
    
    def _generate_bcell_section(self, data: Dict) -> str:
        """B-Cell 섹션 생성"""
        insights = data.get('bcell_insights', [])
        
        html = "<div class='insights-container'>"
        
        for insight in insights[:5]:  # 최대 5개 표시
            confidence = insight.get('confidence', 0)
            confidence_class = "high" if confidence > 0.7 else "medium" if confidence > 0.4 else "low"
            
            html += f"""
            <div class="insight-card bcell-card">
                <div class="insight-header">
                    <span class="date">{insight.get('date', '')}</span>
                    <span class="confidence {confidence_class}">{confidence:.2f}</span>
                </div>
                <div class="insight-content">
                    <p><strong>전문가:</strong> {insight.get('expert_type', '')}</p>
                    <p><strong>추천:</strong> {insight.get('recommendation', '')}</p>
                    <p class="explanation">{insight.get('explanation', '')}</p>
                </div>
            </div>
            """
        
        html += "</div>"
        return html
    
    def _generate_feature_section(self, data: Dict) -> str:
        """특성 섹션 생성"""
        features = data.get('feature_importance', [])
        
        html = "<div class='feature-list'>"
        
        for feature in features[:10]:  # 최대 10개 표시
            importance = feature.get('importance', 0)
            width = int(importance * 100)
            
            html += f"""
            <div class="feature-item">
                <div class="feature-name">{feature.get('feature', '')}</div>
                <div class="feature-bar">
                    <div class="feature-fill" style="width: {width}%"></div>
                    <span class="feature-value">{importance:.3f}</span>
                </div>
                <div class="feature-explanation">{feature.get('explanation', '')}</div>
            </div>
            """
        
        html += "</div>"
        return html
    
    def _generate_xai_section(self, data: Dict) -> str:
        """XAI 섹션 생성"""
        explanations = data.get('xai_explanations', [])
        
        html = "<div class='xai-container'>"
        
        for exp in explanations:
            html += f"""
            <div class="xai-card">
                <h4>{exp.get('component', '')}</h4>
                <div class="xai-content">
                    <p><strong>의사결정 과정:</strong> {exp.get('decision_process', '')}</p>
                    <p><strong>방법론:</strong> {exp.get('methodology', '')}</p>
                    <div class="key-insights">
                        <strong>주요 인사이트:</strong>
                        <ul>
                            {''.join(f'<li>{insight}</li>' for insight in exp.get('key_insights', []))}
                        </ul>
                    </div>
                </div>
            </div>
            """
        
        html += "</div>"
        return html
    
    def _generate_temporal_section(self, data: Dict) -> str:
        """시간별 섹션 생성"""
        temporal = data.get('temporal_patterns', {})
        
        return f"""
        <div class="temporal-content">
            <div class="chart-container">
                <canvas id="temporalChart"></canvas>
            </div>
            <div class="temporal-stats">
                <p><strong>예측 정확도:</strong> {temporal.get('prediction_accuracy', 0):.1%}</p>
                <p><strong>시장 사이클:</strong> {len(temporal.get('market_cycles', {}))}개 패턴 감지</p>
            </div>
        </div>
        """
    
    def _get_css_styles(self) -> str:
        """CSS 스타일"""
        return """
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #f5f7fa;
            color: #333;
            line-height: 1.6;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }
        
        header {
            text-align: center;
            margin-bottom: 30px;
            padding: 20px;
            background: white;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        h1 {
            color: #2c3e50;
            margin-bottom: 10px;
        }
        
        .summary-cards {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .card {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            text-align: center;
        }
        
        .metric {
            font-size: 2em;
            font-weight: bold;
            color: #3498db;
            margin-top: 10px;
        }
        
        .metric.crisis {
            color: #e74c3c;
        }
        
        .main-content {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin-bottom: 30px;
        }
        
        section {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        
        h2 {
            color: #2c3e50;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #ecf0f1;
        }
        
        .insights-container {
            display: flex;
            flex-direction: column;
            gap: 15px;
        }
        
        .insight-card {
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 15px;
            background: #fafafa;
        }
        
        .insight-card.critical {
            border-left: 4px solid #e74c3c;
        }
        
        .insight-card.high {
            border-left: 4px solid #f39c12;
        }
        
        .insight-card.medium {
            border-left: 4px solid #f1c40f;
        }
        
        .insight-card.low {
            border-left: 4px solid #2ecc71;
        }
        
        .insight-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }
        
        .date {
            font-weight: bold;
            color: #7f8c8d;
        }
        
        .severity, .confidence {
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.9em;
            font-weight: bold;
        }
        
        .severity.critical { background: #e74c3c; color: white; }
        .severity.high { background: #f39c12; color: white; }
        .severity.medium { background: #f1c40f; color: white; }
        .severity.low { background: #2ecc71; color: white; }
        
        .confidence.high { background: #2ecc71; color: white; }
        .confidence.medium { background: #f39c12; color: white; }
        .confidence.low { background: #e74c3c; color: white; }
        
        .explanation {
            font-style: italic;
            color: #666;
            margin-top: 10px;
        }
        
        .feature-list {
            display: flex;
            flex-direction: column;
            gap: 15px;
        }
        
        .feature-item {
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 6px;
            background: #fafafa;
        }
        
        .feature-name {
            font-weight: bold;
            margin-bottom: 5px;
        }
        
        .feature-bar {
            position: relative;
            height: 20px;
            background: #ecf0f1;
            border-radius: 10px;
            overflow: hidden;
        }
        
        .feature-fill {
            height: 100%;
            background: linear-gradient(90deg, #3498db, #2ecc71);
            transition: width 0.3s ease;
        }
        
        .feature-value {
            position: absolute;
            right: 10px;
            top: 2px;
            font-size: 0.8em;
            font-weight: bold;
        }
        
        .feature-explanation {
            font-size: 0.9em;
            color: #666;
            margin-top: 5px;
        }
        
        .xai-container {
            display: flex;
            flex-direction: column;
            gap: 15px;
        }
        
        .xai-card {
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 15px;
            background: #f8f9fa;
        }
        
        .xai-card h4 {
            color: #2c3e50;
            margin-bottom: 10px;
        }
        
        .key-insights ul {
            margin-left: 20px;
            margin-top: 5px;
        }
        
        .temporal-section {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        .temporal-content {
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 20px;
            align-items: center;
        }
        
        .chart-container {
            height: 300px;
        }
        
        .temporal-stats {
            padding: 20px;
            background: #f8f9fa;
            border-radius: 8px;
        }
        
        .temporal-stats p {
            margin-bottom: 10px;
        }
        
        @media (max-width: 768px) {
            .main-content {
                grid-template-columns: 1fr;
            }
            
            .temporal-content {
                grid-template-columns: 1fr;
            }
        }
        """
    
    def _get_javascript(self) -> str:
        """JavaScript 코드"""
        return """
        // 간단한 시간별 차트 생성
        document.addEventListener('DOMContentLoaded', function() {
            const ctx = document.getElementById('temporalChart');
            if (ctx) {
                new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: ['1월', '2월', '3월', '4월', '5월', '6월'],
                        datasets: [{
                            label: '위기 감지 수준',
                            data: [0.2, 0.4, 0.6, 0.3, 0.8, 0.5],
                            borderColor: '#e74c3c',
                            backgroundColor: 'rgba(231, 76, 60, 0.1)',
                            tension: 0.4
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {
                            y: {
                                beginAtZero: true,
                                max: 1
                            }
                        }
                    }
                });
            }
        });
        """
    
    def _load_template(self) -> str:
        """템플릿 로드 (향후 확장용)"""
        return ""


# 기존 분석 시스템과 통합하는 함수
def generate_dashboard(analysis_report: Dict, output_dir: str = None):
    """대시보드 생성"""
    
    if output_dir is None:
        output_dir = "."
    
    # HTML 대시보드 생성
    dashboard_generator = HTMLDashboardGenerator()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    html_path = os.path.join(output_dir, f"bipd_dashboard_{timestamp}.html")
    
    dashboard_generator.generate_dashboard(analysis_report, html_path)
    
    # 기존 JSON 파일도 함께 생성 (호환성)
    json_path = os.path.join(output_dir, f"bipd_analysis_{timestamp}.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(analysis_report, f, ensure_ascii=False, indent=2)
    
    return {
        "html_dashboard": html_path,
        "json_report": json_path
    }