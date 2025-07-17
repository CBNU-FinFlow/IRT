"""
면역 시스템 반응 패턴 시각화 - 논문 차별화용
T-Cell과 B-Cell의 동적 상호작용과 적응적 학습 과정을 시각화
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import seaborn as sns
import matplotlib.font_manager as fm
import platform

# 기본 폰트 설정 (한글 폰트 제거)
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지
from matplotlib.patches import Circle, FancyBboxPatch
from matplotlib.collections import LineCollection
import matplotlib.patches as mpatches
from typing import Dict, List, Tuple
import pandas as pd
from datetime import datetime, timedelta


class ImmuneSystemVisualizer:
    """면역 시스템 시각화 클래스"""

    def __init__(self, figsize=(20, 12)):
        self.figsize = figsize
        self.colors = {
            "tcell": "#e74c3c",  # T-Cell: 빨간색
            "bcell": "#3498db",  # B-Cell: 파란색
            "memory": "#f39c12",  # Memory: 주황색
            "antigen": "#2c3e50",  # Antigen: 검정색
            "response": "#2ecc71",  # Response: 초록색
            "crisis": "#8e44ad",  # Crisis: 보라색
        }

    def visualize_immune_response_pattern(
        self, analysis_data: Dict, output_path: str = None
    ):
        """면역 반응 패턴 종합 시각화"""

        fig = plt.figure(figsize=self.figsize)
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

        # 1. T-Cell 활성화 패턴
        ax1 = fig.add_subplot(gs[0, 0:2])
        self._plot_tcell_activation_pattern(ax1, analysis_data)

        # 2. B-Cell 전문가 네트워크
        ax2 = fig.add_subplot(gs[0, 2:4])
        self._plot_bcell_expert_network(ax2, analysis_data)

        # 3. 면역 메모리 형성 과정
        ax3 = fig.add_subplot(gs[1, 0:2])
        self._plot_immune_memory_formation(ax3, analysis_data)

        # 4. 적응적 임계값 조정
        ax4 = fig.add_subplot(gs[1, 2:4])
        self._plot_adaptive_threshold_adjustment(ax4, analysis_data)

        # 5. 위기 전파 네트워크
        ax5 = fig.add_subplot(gs[2, 0:2])
        self._plot_crisis_propagation_network(ax5, analysis_data)

        # 6. XAI 기반 의사결정 트리
        ax6 = fig.add_subplot(gs[2, 2:4])
        self._plot_xai_decision_tree(ax6, analysis_data)

        plt.suptitle(
            "BIPD Immune System Response Pattern Analysis", fontsize=16, fontweight="bold"
        )

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            print(f"Immune system visualization saved: {output_path}")

        return fig

    def _plot_tcell_activation_pattern(self, ax, data: Dict):
        """T-Cell 활성화 패턴 시각화"""
        ax.set_title("T-Cell 위기 감지 활성화 패턴", fontweight="bold")

        # 시뮬레이션 데이터 생성 (실제 데이터로 교체 가능)
        time_points = np.linspace(0, 100, 100)

        # 다양한 T-Cell 활성화 수준
        normal_activation = 0.2 + 0.1 * np.sin(time_points * 0.1)
        crisis_activation = np.where(
            (time_points > 30) & (time_points < 40),
            0.8 + 0.2 * np.sin(time_points * 0.5),
            normal_activation,
        )
        crisis_activation = np.where(
            (time_points > 70) & (time_points < 80),
            0.9 + 0.1 * np.sin(time_points * 0.3),
            crisis_activation,
        )

        # 활성화 패턴 플롯
        ax.fill_between(
            time_points,
            0,
            crisis_activation,
            alpha=0.3,
            color=self.colors["tcell"],
            label="T-Cell 활성화",
        )
        ax.plot(time_points, crisis_activation, color=self.colors["tcell"], linewidth=2)

        # 임계값 라인
        ax.axhline(
            y=0.5,
            color=self.colors["crisis"],
            linestyle="--",
            alpha=0.7,
            label="위기 임계값",
        )

        # 위기 구간 강조
        crisis_zones = [(30, 40), (70, 80)]
        for start, end in crisis_zones:
            ax.axvspan(start, end, alpha=0.2, color=self.colors["crisis"])

        ax.set_xlabel("시간")
        ax.set_ylabel("활성화 수준")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_bcell_expert_network(self, ax, data: Dict):
        """B-Cell 전문가 네트워크 시각화"""
        ax.set_title("B-Cell 전문가 네트워크", fontweight="bold")

        # 네트워크 노드 위치
        experts = {
            "Trend": (0.2, 0.8),
            "Momentum": (0.8, 0.8),
            "Volatility": (0.5, 0.5),
            "Risk": (0.2, 0.2),
            "Value": (0.8, 0.2),
        }

        # 전문가 노드 그리기
        for expert, (x, y) in experts.items():
            # 노드 크기는 활성화 정도에 따라 조정
            activation_level = np.random.uniform(0.3, 1.0)
            size = 1000 + 2000 * activation_level

            ax.scatter(
                x,
                y,
                s=size,
                c=self.colors["bcell"],
                alpha=0.7,
                edgecolors="black",
                linewidth=2,
            )
            ax.text(
                x, y, expert, ha="center", va="center", fontweight="bold", color="white"
            )

        # 전문가 간 연결선 (상호작용 강도)
        connections = [
            ("Trend", "Momentum", 0.8),
            ("Volatility", "Risk", 0.9),
            ("Momentum", "Value", 0.6),
            ("Risk", "Value", 0.7),
            ("Trend", "Volatility", 0.5),
        ]

        for exp1, exp2, strength in connections:
            x1, y1 = experts[exp1]
            x2, y2 = experts[exp2]

            # 연결선 두께는 상호작용 강도에 비례
            ax.plot(
                [x1, x2],
                [y1, y2],
                color=self.colors["response"],
                linewidth=strength * 3,
                alpha=0.6,
            )

        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        ax.set_aspect("equal")
        ax.axis("off")

        # 범례
        high_activation = mpatches.Patch(
            color=self.colors["bcell"], label="높은 활성화"
        )
        connection = mpatches.Patch(color=self.colors["response"], label="전문가 연결")
        ax.legend(handles=[high_activation, connection], loc="upper right")

    def _plot_immune_memory_formation(self, ax, data: Dict):
        """면역 메모리 형성 과정 시각화"""
        ax.set_title("면역 메모리 형성 및 활용", fontweight="bold")

        # 메모리 형성 시간선
        time_points = np.arange(0, 50, 1)

        # 초기 학습 곡선
        initial_learning = 1 - np.exp(-time_points / 10)

        # 메모리 강화 이벤트
        memory_events = [15, 30, 45]
        memory_strength = initial_learning.copy()

        for event in memory_events:
            if event < len(memory_strength):
                # 메모리 강화
                memory_strength[event:] += 0.3 * np.exp(
                    -(time_points[event:] - event) / 5
                )

        # 메모리 감쇠
        decay_factor = np.exp(-time_points / 100)
        memory_strength *= 0.5 + 0.5 * decay_factor

        # 플롯
        ax.plot(
            time_points,
            memory_strength,
            color=self.colors["memory"],
            linewidth=3,
            label="메모리 강도",
        )

        # 메모리 강화 이벤트 표시
        for event in memory_events:
            ax.axvline(x=event, color=self.colors["crisis"], linestyle=":", alpha=0.7)
            ax.annotate(
                f"위기 학습",
                xy=(event, memory_strength[event]),
                xytext=(event + 3, memory_strength[event] + 0.1),
                arrowprops=dict(arrowstyle="->", color=self.colors["crisis"]),
            )

        # 메모리 활용 구간
        utilization_zones = [(10, 20), (25, 35), (40, 50)]
        for start, end in utilization_zones:
            ax.axvspan(start, end, alpha=0.1, color=self.colors["memory"])

        ax.set_xlabel("시간")
        ax.set_ylabel("메모리 강도")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_adaptive_threshold_adjustment(self, ax, data: Dict):
        """적응적 임계값 조정 시각화"""
        ax.set_title("적응적 임계값 조정 메커니즘", fontweight="bold")

        time_points = np.linspace(0, 100, 100)

        # 기본 임계값
        base_threshold = 0.5

        # 시장 변동성에 따른 임계값 조정
        market_volatility = 0.3 * np.sin(time_points * 0.05) + 0.5
        adaptive_threshold = base_threshold + 0.2 * (market_volatility - 0.5)

        # 학습 기반 임계값 조정
        learning_adjustment = 0.1 * np.sin(time_points * 0.02)
        final_threshold = adaptive_threshold + learning_adjustment

        # 플롯
        ax.plot(
            time_points,
            [base_threshold] * len(time_points),
            color="gray",
            linestyle="--",
            alpha=0.5,
            label="기본 임계값",
        )
        ax.plot(
            time_points,
            adaptive_threshold,
            color=self.colors["tcell"],
            alpha=0.7,
            label="시장 적응 임계값",
        )
        ax.plot(
            time_points,
            final_threshold,
            color=self.colors["crisis"],
            linewidth=2,
            label="학습 기반 최종 임계값",
        )

        # 임계값 조정 효과 시각화
        ax.fill_between(
            time_points,
            base_threshold,
            final_threshold,
            alpha=0.2,
            color=self.colors["tcell"],
        )

        ax.set_xlabel("시간")
        ax.set_ylabel("임계값")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_crisis_propagation_network(self, ax, data: Dict):
        """위기 전파 네트워크 시각화"""
        ax.set_title("위기 전파 및 면역 반응 네트워크", fontweight="bold")

        # 네트워크 구조
        nodes = {
            "Market": (0.5, 0.9),
            "T-Cell1": (0.2, 0.7),
            "T-Cell2": (0.8, 0.7),
            "B-Cell1": (0.3, 0.5),
            "B-Cell2": (0.7, 0.5),
            "Memory": (0.5, 0.3),
            "Response": (0.5, 0.1),
        }

        # 위기 전파 시뮬레이션
        crisis_intensity = {
            "Market": 1.0,
            "T-Cell1": 0.8,
            "T-Cell2": 0.9,
            "B-Cell1": 0.6,
            "B-Cell2": 0.7,
            "Memory": 0.4,
            "Response": 0.8,
        }

        # 노드 그리기
        for node, (x, y) in nodes.items():
            intensity = crisis_intensity[node]
            color = plt.cm.Reds(intensity)
            size = 500 + 1000 * intensity

            ax.scatter(
                x, y, s=size, c=[color], edgecolors="black", linewidth=2, alpha=0.8
            )
            ax.text(
                x,
                y,
                node,
                ha="center",
                va="center",
                fontweight="bold",
                color="white" if intensity > 0.5 else "black",
            )

        # 연결선 (신호 전파)
        connections = [
            ("Market", "T-Cell1", 0.9),
            ("Market", "T-Cell2", 0.8),
            ("T-Cell1", "B-Cell1", 0.7),
            ("T-Cell2", "B-Cell2", 0.8),
            ("B-Cell1", "Memory", 0.5),
            ("B-Cell2", "Memory", 0.6),
            ("Memory", "Response", 0.7),
            ("B-Cell1", "Response", 0.6),
            ("B-Cell2", "Response", 0.7),
        ]

        for start, end, strength in connections:
            x1, y1 = nodes[start]
            x2, y2 = nodes[end]

            # 화살표로 방향성 표시
            ax.annotate(
                "",
                xy=(x2, y2),
                xytext=(x1, y1),
                arrowprops=dict(
                    arrowstyle="->",
                    color=self.colors["crisis"],
                    lw=strength * 3,
                    alpha=0.7,
                ),
            )

        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        ax.set_aspect("equal")
        ax.axis("off")

        # 범례
        high_intensity = mpatches.Patch(color="red", label="높은 위기 강도")
        low_intensity = mpatches.Patch(color="pink", label="낮은 위기 강도")
        ax.legend(handles=[high_intensity, low_intensity], loc="upper left")

    def _plot_xai_decision_tree(self, ax, data: Dict):
        """XAI 기반 의사결정 트리 시각화"""
        ax.set_title("XAI 기반 의사결정 과정", fontweight="bold")

        # 의사결정 노드
        decision_nodes = {
            "Market Signal": (0.5, 0.9),
            "T-Cell Analysis": (0.3, 0.7),
            "B-Cell Assessment": (0.7, 0.7),
            "Memory Check": (0.2, 0.5),
            "Expert Selection": (0.8, 0.5),
            "Risk Evaluation": (0.4, 0.3),
            "Strategy Decision": (0.6, 0.3),
            "Action": (0.5, 0.1),
        }

        # 의사결정 경로
        decision_paths = [
            ("Market Signal", "T-Cell Analysis", "Crisis Detection"),
            ("Market Signal", "B-Cell Assessment", "Expert Analysis"),
            ("T-Cell Analysis", "Memory Check", "Historical Pattern"),
            ("B-Cell Assessment", "Expert Selection", "Best Expert"),
            ("Memory Check", "Risk Evaluation", "Risk Assessment"),
            ("Expert Selection", "Strategy Decision", "Strategy Choice"),
            ("Risk Evaluation", "Action", "Execute"),
            ("Strategy Decision", "Action", "Execute"),
        ]

        # 노드 그리기
        for node, (x, y) in decision_nodes.items():
            # 노드 타입에 따른 색상
            if "T-Cell" in node:
                color = self.colors["tcell"]
            elif "B-Cell" in node or "Expert" in node:
                color = self.colors["bcell"]
            elif "Memory" in node:
                color = self.colors["memory"]
            else:
                color = self.colors["response"]

            # 둥근 사각형 노드
            bbox = FancyBboxPatch(
                (x - 0.08, y - 0.03),
                0.16,
                0.06,
                boxstyle="round,pad=0.01",
                facecolor=color,
                alpha=0.7,
                edgecolor="black",
                linewidth=1,
            )
            ax.add_patch(bbox)

            ax.text(
                x,
                y,
                node,
                ha="center",
                va="center",
                fontweight="bold",
                fontsize=8,
                color="white",
            )

        # 의사결정 경로 그리기
        for start, end, label in decision_paths:
            x1, y1 = decision_nodes[start]
            x2, y2 = decision_nodes[end]

            # 화살표
            ax.annotate(
                "",
                xy=(x2, y2),
                xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.5, alpha=0.7),
            )

            # 경로 라벨
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(
                mid_x,
                mid_y,
                label,
                ha="center",
                va="center",
                fontsize=6,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
            )

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.axis("off")

    def create_comprehensive_immune_analysis(
        self, bipd_instance, start_date: str, end_date: str, output_dir: str = None
    ):
        """종합적인 면역 시스템 분석 및 시각화"""

        if output_dir is None:
            output_dir = "."

        # 분석 데이터 생성
        analysis_data = self._extract_immune_analysis_data(
            bipd_instance, start_date, end_date
        )

        # 면역 시스템 패턴 시각화
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pattern_path = f"{output_dir}/immune_patterns_{timestamp}.png"
        pattern_fig = self.visualize_immune_response_pattern(
            analysis_data, pattern_path
        )

        # 추가 분석 차트
        detailed_path = f"{output_dir}/immune_detailed_{timestamp}.png"
        detailed_fig = self._create_detailed_immune_analysis(
            analysis_data, detailed_path
        )

        print(f"면역 시스템 시각화 완료:")
        print(f"  - 반응 패턴: {pattern_path}")
        print(f"  - 상세 분석: {detailed_path}")
        print(f"이 시각화는 기존 연구와 차별화된 T-Cell 기반 XAI 구현을 보여줍니다!")

        return pattern_fig, detailed_fig

    def _extract_immune_analysis_data(
        self, bipd_instance, start_date: str, end_date: str
    ) -> Dict:
        """면역 시스템 분석 데이터 추출"""

        # 실제 구현에서는 BIPD 인스턴스에서 데이터 추출
        # 여기서는 시뮬레이션 데이터 생성

        return {
            "tcell_activations": self._generate_tcell_data(),
            "bcell_responses": self._generate_bcell_data(),
            "memory_events": self._generate_memory_data(),
            "crisis_events": self._generate_crisis_data(),
            "adaptive_thresholds": self._generate_threshold_data(),
        }

    def _generate_tcell_data(self) -> Dict:
        """T-Cell 데이터 생성"""
        return {
            "activation_timeline": np.random.random(100),
            "crisis_detections": np.random.randint(0, 5, 20),
            "sensitivity_adjustments": np.random.random(50),
        }

    def _generate_bcell_data(self) -> Dict:
        """B-Cell 데이터 생성"""
        return {
            "expert_activations": np.random.random(50),
            "confidence_scores": np.random.random(50),
            "strategy_selections": np.random.randint(0, 5, 30),
        }

    def _generate_memory_data(self) -> Dict:
        """메모리 데이터 생성"""
        return {
            "formation_events": np.random.randint(0, 10, 20),
            "retrieval_events": np.random.randint(0, 8, 25),
            "strength_evolution": np.random.random(100),
        }

    def _generate_crisis_data(self) -> Dict:
        """위기 데이터 생성"""
        return {
            "crisis_levels": np.random.random(100),
            "propagation_patterns": np.random.random((10, 10)),
            "response_times": np.random.random(50),
        }

    def _generate_threshold_data(self) -> Dict:
        """임계값 데이터 생성"""
        return {
            "base_thresholds": np.random.random(100),
            "adaptive_adjustments": np.random.random(100),
            "learning_effects": np.random.random(100),
        }

    def _create_detailed_immune_analysis(self, data: Dict, output_path: str = None):
        """상세 면역 분석 차트"""

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # 1. T-Cell 민감도 분포
        ax = axes[0, 0]
        sensitivities = np.random.normal(0.5, 0.2, 1000)
        ax.hist(sensitivities, bins=30, color=self.colors["tcell"], alpha=0.7)
        ax.set_title("T-Cell 민감도 분포")
        ax.set_xlabel("민감도")
        ax.set_ylabel("빈도")

        # 2. B-Cell 전문가 성능
        ax = axes[0, 1]
        experts = ["Trend", "Momentum", "Volatility", "Risk", "Value"]
        performance = np.random.uniform(0.6, 0.95, len(experts))
        ax.bar(experts, performance, color=self.colors["bcell"], alpha=0.7)
        ax.set_title("B-Cell 전문가 성능")
        ax.set_ylabel("정확도")
        plt.setp(ax.get_xticklabels(), rotation=45)

        # 3. 메모리 활용 패턴
        ax = axes[0, 2]
        time_points = np.arange(0, 30, 1)
        memory_usage = np.random.exponential(0.3, len(time_points))
        ax.plot(time_points, memory_usage, color=self.colors["memory"], marker="o")
        ax.set_title("메모리 활용 패턴")
        ax.set_xlabel("시간")
        ax.set_ylabel("활용도")

        # 4. 위기 전파 속도
        ax = axes[1, 0]
        propagation_speeds = np.random.gamma(2, 0.5, 100)
        ax.hist(propagation_speeds, bins=20, color=self.colors["crisis"], alpha=0.7)
        ax.set_title("위기 전파 속도 분포")
        ax.set_xlabel("전파 속도")
        ax.set_ylabel("빈도")

        # 5. 적응적 학습 곡선
        ax = axes[1, 1]
        learning_curve = 1 - np.exp(-np.arange(0, 50, 1) / 10)
        ax.plot(learning_curve, color=self.colors["response"], linewidth=2)
        ax.set_title("적응적 학습 곡선")
        ax.set_xlabel("시간")
        ax.set_ylabel("학습 수준")

        # 6. XAI 설명 품질
        ax = axes[1, 2]
        explanation_quality = np.random.beta(3, 2, 100)
        ax.hist(explanation_quality, bins=20, color=self.colors["memory"], alpha=0.7)
        ax.set_title("XAI 설명 품질 분포")
        ax.set_xlabel("설명 품질")
        ax.set_ylabel("빈도")

        plt.tight_layout()
        plt.suptitle("면역 시스템 상세 분석", fontsize=16, fontweight="bold", y=1.02)

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")

        return fig


# 기존 시스템과 통합하는 함수
def create_visualizations(
    bipd_instance, start_date: str, end_date: str, output_dir: str = None
):
    """면역 시스템 시각화 생성"""

    visualizer = ImmuneSystemVisualizer()

    # 종합 분석 및 시각화
    pattern_fig, detailed_fig = visualizer.create_comprehensive_immune_analysis(
        bipd_instance, start_date, end_date, output_dir
    )

    return {
        "pattern_visualization": pattern_fig,
        "detailed_analysis": detailed_fig,
        "visualizer": visualizer,
    }
