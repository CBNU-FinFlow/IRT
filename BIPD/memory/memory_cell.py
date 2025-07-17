"""
메모리 시스템
생체면역 시스템의 메모리 세포를 모델링한 학습 및 기억 컴포넌트
"""

import numpy as np
from typing import Dict, List, Any, Optional
from collections import deque
from datetime import datetime
import json


class MemoryCell:
    """
    면역 메모리 세포
    과거 경험을 저장하고 유사 상황에서 재활용
    """
    
    def __init__(self, cell_id: str, memory_capacity: int = 100):
        self.cell_id = cell_id
        self.memory_capacity = memory_capacity
        self.similarity_threshold = 0.8
        self.memory_strength = 0.5
        
        # 메모리 저장소
        self.crisis_memories = deque(maxlen=memory_capacity)
        self.success_patterns = deque(maxlen=50)
        self.failure_patterns = deque(maxlen=50)
        
        # 활성화 상태
        self.activation_level = 0.0
        self.last_recall_time = None
        
        # 통계
        self.recall_count = 0
        self.match_count = 0
    
    def store_memory(self, 
                    market_features: np.ndarray,
                    portfolio_weights: np.ndarray,
                    performance: float,
                    crisis_level: float,
                    strategy_type: str = "general"):
        """
        메모리 저장
        
        Args:
            market_features: 시장 특성
            portfolio_weights: 포트폴리오 가중치
            performance: 성과
            crisis_level: 위기 수준
            strategy_type: 전략 유형
        """
        memory_entry = {
            'timestamp': datetime.now().isoformat(),
            'market_features': market_features.copy(),
            'portfolio_weights': portfolio_weights.copy(),
            'performance': performance,
            'crisis_level': crisis_level,
            'strategy_type': strategy_type,
            'recall_count': 0,
            'success_rate': 1.0 if performance > 0 else 0.0
        }
        
        # 위기 메모리 저장
        self.crisis_memories.append(memory_entry)
        
        # 성공/실패 패턴 분류
        if performance > 0.01:  # 1% 이상 수익
            self.success_patterns.append(memory_entry)
        elif performance < -0.01:  # 1% 이상 손실
            self.failure_patterns.append(memory_entry)
    
    def recall_memory(self, current_features: np.ndarray, crisis_level: float) -> Optional[Dict[str, Any]]:
        """
        메모리 회상
        
        Args:
            current_features: 현재 시장 특성
            crisis_level: 현재 위기 수준
            
        Returns:
            유사한 과거 경험 또는 None
        """
        self.recall_count += 1
        
        if not self.crisis_memories:
            return None
        
        # 유사한 메모리 검색
        best_match = None
        best_similarity = 0.0
        
        for memory in self.crisis_memories:
            # 시장 특성 유사도 계산
            feature_similarity = self._calculate_similarity(
                current_features, memory['market_features']
            )
            
            # 위기 수준 유사도 계산
            crisis_similarity = 1.0 - abs(crisis_level - memory['crisis_level'])
            
            # 통합 유사도 (70% 시장 특성, 30% 위기 수준)
            total_similarity = 0.7 * feature_similarity + 0.3 * crisis_similarity
            
            if total_similarity > best_similarity and total_similarity > self.similarity_threshold:
                best_similarity = total_similarity
                best_match = memory
        
        if best_match:
            self.match_count += 1
            self.activation_level = best_similarity
            self.last_recall_time = datetime.now()
            
            # 회상 횟수 증가
            best_match['recall_count'] += 1
            
            # 메모리 강화
            self._strengthen_memory(best_match)
            
            return {
                'memory': best_match,
                'similarity': best_similarity,
                'recall_timestamp': datetime.now().isoformat()
            }
        
        return None
    
    def _calculate_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """특성 벡터 간 유사도 계산"""
        try:
            # 길이 맞춤
            min_len = min(len(features1), len(features2))
            f1 = features1[:min_len]
            f2 = features2[:min_len]
            
            # 코사인 유사도
            dot_product = np.dot(f1, f2)
            norm1 = np.linalg.norm(f1)
            norm2 = np.linalg.norm(f2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return max(0.0, similarity)  # 음수 제거
            
        except Exception:
            return 0.0
    
    def _strengthen_memory(self, memory: Dict[str, Any]):
        """메모리 강화"""
        # 회상 횟수에 따른 강화
        recall_boost = min(0.1 * memory['recall_count'], 0.5)
        
        # 성과에 따른 강화
        performance_boost = memory['performance'] * 0.1
        
        # 메모리 강도 업데이트
        memory['memory_strength'] = min(
            memory.get('memory_strength', 0.5) + recall_boost + performance_boost,
            1.0
        )
    
    def get_successful_patterns(self, top_k: int = 5) -> List[Dict[str, Any]]:
        """성공 패턴 반환"""
        if not self.success_patterns:
            return []
        
        # 성과 순으로 정렬
        sorted_patterns = sorted(
            self.success_patterns, 
            key=lambda x: x['performance'], 
            reverse=True
        )
        
        return list(sorted_patterns[:top_k])
    
    def get_failure_patterns(self, top_k: int = 5) -> List[Dict[str, Any]]:
        """실패 패턴 반환"""
        if not self.failure_patterns:
            return []
        
        # 손실 순으로 정렬
        sorted_patterns = sorted(
            self.failure_patterns, 
            key=lambda x: x['performance']
        )
        
        return list(sorted_patterns[:top_k])
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """메모리 통계 반환"""
        total_memories = len(self.crisis_memories)
        
        if total_memories == 0:
            return {
                'total_memories': 0,
                'recall_rate': 0.0,
                'match_rate': 0.0,
                'avg_performance': 0.0,
                'memory_utilization': 0.0
            }
        
        performances = [m['performance'] for m in self.crisis_memories]
        recall_counts = [m['recall_count'] for m in self.crisis_memories]
        
        return {
            'total_memories': total_memories,
            'recall_rate': self.recall_count / max(total_memories, 1),
            'match_rate': self.match_count / max(self.recall_count, 1),
            'avg_performance': np.mean(performances),
            'memory_utilization': len(self.crisis_memories) / self.memory_capacity,
            'avg_recall_count': np.mean(recall_counts),
            'success_patterns': len(self.success_patterns),
            'failure_patterns': len(self.failure_patterns)
        }
    
    def clear_memory(self):
        """메모리 초기화"""
        self.crisis_memories.clear()
        self.success_patterns.clear()
        self.failure_patterns.clear()
        self.activation_level = 0.0
        self.recall_count = 0
        self.match_count = 0
        self.last_recall_time = None
    
    def export_memory(self, filepath: str):
        """메모리 내보내기"""
        memory_data = {
            'cell_id': self.cell_id,
            'crisis_memories': [
                {
                    'timestamp': m['timestamp'],
                    'market_features': m['market_features'].tolist(),
                    'portfolio_weights': m['portfolio_weights'].tolist(),
                    'performance': m['performance'],
                    'crisis_level': m['crisis_level'],
                    'strategy_type': m['strategy_type'],
                    'recall_count': m['recall_count'],
                    'success_rate': m['success_rate']
                }
                for m in self.crisis_memories
            ],
            'statistics': self.get_memory_statistics()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(memory_data, f, ensure_ascii=False, indent=2)
    
    def import_memory(self, filepath: str):
        """메모리 가져오기"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                memory_data = json.load(f)
            
            # 메모리 복원
            for m in memory_data['crisis_memories']:
                memory_entry = {
                    'timestamp': m['timestamp'],
                    'market_features': np.array(m['market_features']),
                    'portfolio_weights': np.array(m['portfolio_weights']),
                    'performance': m['performance'],
                    'crisis_level': m['crisis_level'],
                    'strategy_type': m['strategy_type'],
                    'recall_count': m['recall_count'],
                    'success_rate': m['success_rate']
                }
                
                self.crisis_memories.append(memory_entry)
                
                # 성공/실패 패턴 분류
                if memory_entry['performance'] > 0.01:
                    self.success_patterns.append(memory_entry)
                elif memory_entry['performance'] < -0.01:
                    self.failure_patterns.append(memory_entry)
            
            print(f"메모리 복원 완료: {len(self.crisis_memories)}개 메모리")
            
        except Exception as e:
            print(f"메모리 가져오기 실패: {str(e)}")


class MemorySystem:
    """
    통합 메모리 시스템
    여러 메모리 세포를 관리하고 조정
    """
    
    def __init__(self, n_memory_cells: int = 3):
        self.memory_cells = [
            MemoryCell(f"Memory-{i+1:02d}") 
            for i in range(n_memory_cells)
        ]
        
        # 메모리 분배 전략
        self.memory_allocation = {
            0: 'general',      # 일반 상황
            1: 'crisis',       # 위기 상황
            2: 'recovery'      # 회복 상황
        }
    
    def store_experience(self, 
                        market_features: np.ndarray,
                        portfolio_weights: np.ndarray,
                        performance: float,
                        crisis_level: float):
        """경험 저장"""
        # 상황별 메모리 셀 선택
        target_cell = self._select_memory_cell(crisis_level, performance)
        
        # 전략 유형 결정
        strategy_type = self._determine_strategy_type(crisis_level, performance)
        
        # 메모리 저장
        target_cell.store_memory(
            market_features, portfolio_weights, performance, crisis_level, strategy_type
        )
    
    def recall_experience(self, current_features: np.ndarray, crisis_level: float) -> Optional[Dict[str, Any]]:
        """경험 회상"""
        best_recall = None
        best_similarity = 0.0
        
        # 모든 메모리 셀에서 검색
        for cell in self.memory_cells:
            recall_result = cell.recall_memory(current_features, crisis_level)
            
            if recall_result and recall_result['similarity'] > best_similarity:
                best_similarity = recall_result['similarity']
                best_recall = recall_result
                best_recall['source_cell'] = cell.cell_id
        
        return best_recall
    
    def _select_memory_cell(self, crisis_level: float, performance: float) -> MemoryCell:
        """메모리 셀 선택"""
        if crisis_level > 0.5:
            return self.memory_cells[1]  # 위기 상황
        elif performance > 0.02:
            return self.memory_cells[2]  # 회복 상황
        else:
            return self.memory_cells[0]  # 일반 상황
    
    def _determine_strategy_type(self, crisis_level: float, performance: float) -> str:
        """전략 유형 결정"""
        if crisis_level > 0.7:
            return 'crisis_defense'
        elif crisis_level > 0.4:
            return 'risk_management'
        elif performance > 0.02:
            return 'growth_opportunity'
        else:
            return 'general'
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """시스템 통계 반환"""
        total_memories = sum(len(cell.crisis_memories) for cell in self.memory_cells)
        total_recalls = sum(cell.recall_count for cell in self.memory_cells)
        total_matches = sum(cell.match_count for cell in self.memory_cells)
        
        cell_stats = {}
        for i, cell in enumerate(self.memory_cells):
            cell_stats[f'cell_{i+1}'] = cell.get_memory_statistics()
        
        return {
            'total_memories': total_memories,
            'total_recalls': total_recalls,
            'total_matches': total_matches,
            'overall_match_rate': total_matches / max(total_recalls, 1),
            'memory_cells': cell_stats
        }
    
    def clear_all_memories(self):
        """모든 메모리 초기화"""
        for cell in self.memory_cells:
            cell.clear_memory()
    
    def export_all_memories(self, base_filepath: str):
        """모든 메모리 내보내기"""
        for i, cell in enumerate(self.memory_cells):
            filepath = f"{base_filepath}_cell_{i+1}.json"
            cell.export_memory(filepath)
    
    def import_all_memories(self, base_filepath: str):
        """모든 메모리 가져오기"""
        for i, cell in enumerate(self.memory_cells):
            filepath = f"{base_filepath}_cell_{i+1}.json"
            cell.import_memory(filepath)