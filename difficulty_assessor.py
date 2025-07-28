import numpy as np
import math
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from sklearn.preprocessing import MinMaxScaler  # type: ignore
import scipy.stats as stats  # type: ignore

@dataclass
class DifficultyMetrics:
    """난이도 평가를 위한 메트릭 클래스"""
    
    # 기본 플레이 메트릭
    total_steps: int = 0
    total_reward: float = 0.0
    box_pushes: int = 0
    solved: bool = False
    
    # 정책 관련 메트릭
    mean_entropy: float = 0.0
    std_entropy: float = 0.0
    mean_value: float = 0.0
    std_value: float = 0.0
    
    # 시간 관련 메트릭
    mean_inference_time: float = 0.0
    
    # 레벨 구조 메트릭
    level_size: Tuple[int, int] = (0, 0)
    num_boxes: int = 0
    num_targets: int = 0
    wall_ratio: float = 0.0
    box_density: float = 0.0
    level_complexity: float = 0.0


class DifficultyAssessor:
    """
    소코반 레벨의 난이도를 다차원적으로 평가하는 클래스
    
    평가 차원:
    1. 구조적 복잡도: 레벨의 기하학적 구조
    2. 인지적 부하: 에이전트의 정책 엔트로피 및 추론 시간
    3. 해결 경로 복잡도: 스텝 수, 박스 푸시 횟수 등
    4. 성공 확률: 해결 가능성
    """
    
    def __init__(self, 
                 weight_structural: float = 0.3,
                 weight_cognitive: float = 0.3,
                 weight_solution: float = 0.3,
                 weight_success: float = 0.1):
        """
        Args:
            weight_structural: 구조적 복잡도 가중치
            weight_cognitive: 인지적 부하 가중치
            weight_solution: 해결 경로 복잡도 가중치
            weight_success: 성공 확률 가중치
        """
        self.weight_structural = weight_structural
        self.weight_cognitive = weight_cognitive
        self.weight_solution = weight_solution
        self.weight_success = weight_success
        
        # 가중치 정규화
        total_weight = sum([weight_structural, weight_cognitive, weight_solution, weight_success])
        self.weight_structural /= total_weight
        self.weight_cognitive /= total_weight
        self.weight_solution /= total_weight
        self.weight_success /= total_weight
        
        # 메트릭 정규화를 위한 히스토리
        self.metric_history: List[DifficultyMetrics] = []
        self.max_history_size = 100
        
        # 정규화 스케일러
        self.scaler_initialized = False
        self.metric_ranges: Dict[str, Any] = {}
    
    def assess_difficulty(self, play_data: Dict[str, Any], level_data: Dict[str, Any]) -> Dict[str, float]:
        """
        레벨의 난이도를 평가합니다.
        
        Args:
            play_data: 에이전트의 플레이 데이터
            level_data: 레벨의 구조적 데이터
            
        Returns:
            난이도 점수 및 세부 메트릭
        """
        # 메트릭 객체 생성
        metrics = self._create_metrics(play_data, level_data)
        
        # 각 차원별 난이도 계산
        structural_difficulty = self._assess_structural_difficulty(metrics)
        cognitive_difficulty = self._assess_cognitive_difficulty(metrics)
        solution_difficulty = self._assess_solution_difficulty(metrics)
        success_difficulty = self._assess_success_difficulty(metrics)
        
        # 종합 난이도 계산
        total_difficulty = (
            self.weight_structural * structural_difficulty +
            self.weight_cognitive * cognitive_difficulty +
            self.weight_solution * solution_difficulty +
            self.weight_success * success_difficulty
        )
        
        # 0-1 범위로 클램핑
        total_difficulty = np.clip(total_difficulty, 0.0, 1.0)
        
        # 히스토리 업데이트
        self._update_history(metrics)
        
        return {
            'total_difficulty': total_difficulty,
            'structural_difficulty': structural_difficulty,
            'cognitive_difficulty': cognitive_difficulty,
            'solution_difficulty': solution_difficulty,
            'success_difficulty': success_difficulty,
            'metrics': self._metrics_to_dict(metrics)
        }
    
    def _create_metrics(self, play_data: Dict[str, Any], level_data: Dict[str, Any]) -> DifficultyMetrics:
        """플레이 데이터와 레벨 데이터로부터 메트릭 객체 생성"""
        return DifficultyMetrics(
            total_steps=play_data.get('total_steps', 0),
            total_reward=play_data.get('total_reward', 0.0),
            box_pushes=play_data.get('box_pushes', 0),
            solved=play_data.get('solved', False),
            mean_entropy=play_data.get('mean_entropy', 0.0),
            std_entropy=play_data.get('std_entropy', 0.0),
            mean_value=play_data.get('mean_value', 0.0),
            std_value=play_data.get('std_value', 0.0),
            mean_inference_time=play_data.get('mean_inference_time', 0.0),
            level_size=level_data.get('size', (0, 0)),
            num_boxes=level_data.get('num_boxes', 0),
            num_targets=level_data.get('num_targets', 0),
            wall_ratio=self._calculate_wall_ratio(level_data),
            box_density=self._calculate_box_density(level_data),
            level_complexity=level_data.get('level_complexity', 0.0)
        )
    
    def _assess_structural_difficulty(self, metrics: DifficultyMetrics) -> float:
        """구조적 복잡도 평가"""
        # 레벨 크기 복잡도
        size_complexity = math.sqrt(metrics.level_size[0] * metrics.level_size[1]) / 20.0
        size_complexity = min(size_complexity, 1.0)
        
        # 박스 개수 복잡도
        box_complexity = metrics.num_boxes / 10.0
        box_complexity = min(box_complexity, 1.0)
        
        # 벽 비율 복잡도 (적당한 벽 비율이 중간 난이도)
        wall_complexity = 1.0 - abs(metrics.wall_ratio - 0.4) * 2.5
        wall_complexity = max(wall_complexity, 0.0)
        
        # 박스 밀도 복잡도
        density_complexity = metrics.box_density
        
        # 기본 레벨 복잡도
        base_complexity = metrics.level_complexity
        
        # 가중 평균
        structural_score = (
            0.2 * size_complexity +
            0.3 * box_complexity +
            0.2 * wall_complexity +
            0.15 * density_complexity +
            0.15 * base_complexity
        )
        
        return min(structural_score, 1.0)
    
    def _assess_cognitive_difficulty(self, metrics: DifficultyMetrics) -> float:
        """인지적 부하 평가"""
        # 정책 엔트로피 (높을수록 불확실성 증가)
        entropy_score = min(metrics.mean_entropy / 1.5, 1.0)  # 최대 엔트로피는 log(4) ≈ 1.39
        
        # 엔트로피 변동성 (불안정성)
        entropy_variability = min(metrics.std_entropy / 0.5, 1.0)
        
        # 가치 함수 불확실성
        value_uncertainty = min(abs(metrics.mean_value) / 50.0, 1.0)
        value_variability = min(metrics.std_value / 25.0, 1.0)
        
        # 추론 시간 (복잡할수록 더 오래 걸림)
        inference_complexity = min(metrics.mean_inference_time / 0.01, 1.0)
        
        # 가중 평균
        cognitive_score = (
            0.3 * entropy_score +
            0.2 * entropy_variability +
            0.2 * value_uncertainty +
            0.15 * value_variability +
            0.15 * inference_complexity
        )
        
        return min(cognitive_score, 1.0)
    
    def _assess_solution_difficulty(self, metrics: DifficultyMetrics) -> float:
        """해결 경로 복잡도 평가"""
        # 스텝 수 복잡도
        steps_complexity = min(metrics.total_steps / 200.0, 1.0)
        
        # 박스 푸시 복잡도
        push_complexity = min(metrics.box_pushes / (metrics.num_boxes * 3), 1.0) if metrics.num_boxes > 0 else 0
        
        # 효율성 (보상 대비 스텝 수)
        efficiency = metrics.total_reward / max(metrics.total_steps, 1)
        inefficiency_score = 1.0 - min(efficiency / 5.0, 1.0)  # 낮은 효율성 = 높은 난이도
        
        # 가중 평균
        solution_score = (
            0.4 * steps_complexity +
            0.3 * push_complexity +
            0.3 * inefficiency_score
        )
        
        return min(solution_score, 1.0)
    
    def _assess_success_difficulty(self, metrics: DifficultyMetrics) -> float:
        """성공 확률 기반 난이도 평가"""
        if metrics.solved:
            # 해결했지만 어려웠던 정도를 다른 메트릭으로 추정
            base_success_difficulty = 0.3
            
            # 많은 스텝이 필요했다면 어려웠던 것
            if metrics.total_steps > 100:
                base_success_difficulty += 0.3
            
            # 박스를 많이 밀었다면 어려웠던 것
            if metrics.num_boxes > 0 and metrics.box_pushes > metrics.num_boxes * 2:
                base_success_difficulty += 0.2
                
            return min(base_success_difficulty, 1.0)
        else:
            # 해결하지 못함 = 높은 난이도
            return 0.8
    
    def _calculate_wall_ratio(self, level_data: Dict[str, Any]) -> float:
        """벽 비율 계산"""
        level = level_data.get('level')
        if level is None:
            return 0.0
        
        total_cells = level.size
        wall_cells = np.sum(level == 1)  # 벽은 1로 표시
        return wall_cells / total_cells
    
    def _calculate_box_density(self, level_data: Dict[str, Any]) -> float:
        """박스 밀도 계산"""
        level = level_data.get('level')
        if level is None:
            return 0.0
        
        total_cells = level.size
        box_cells = np.sum((level == 2) | (level == 5))  # 박스는 2, 목표 위 박스는 5
        return box_cells / total_cells
    
    def _update_history(self, metrics: DifficultyMetrics):
        """메트릭 히스토리 업데이트"""
        self.metric_history.append(metrics)
        
        # 최대 히스토리 크기 유지
        if len(self.metric_history) > self.max_history_size:
            self.metric_history = self.metric_history[-self.max_history_size:]
        
        # 정규화 범위 업데이트
        self._update_normalization_ranges()
    
    def _update_normalization_ranges(self):
        """정규화를 위한 메트릭 범위 업데이트"""
        if len(self.metric_history) < 5:  # 최소 5개 샘플 필요
            return
        
        metrics_array = []
        for m in self.metric_history:
            metrics_array.append([
                m.total_steps, m.box_pushes, m.mean_entropy, m.std_entropy,
                m.mean_value, m.std_value, m.mean_inference_time,
                m.level_size[0] * m.level_size[1], m.num_boxes, m.wall_ratio, m.box_density
            ])
        
        metrics_array = np.array(metrics_array)
        
        # 최소, 최대값 계산
        self.metric_ranges = {
            'min_values': np.min(metrics_array, axis=0),
            'max_values': np.max(metrics_array, axis=0)
        }
        
        self.scaler_initialized = True
    
    def _metrics_to_dict(self, metrics: DifficultyMetrics) -> Dict[str, Any]:
        """메트릭 객체를 딕셔너리로 변환"""
        return {
            'total_steps': metrics.total_steps,
            'total_reward': metrics.total_reward,
            'box_pushes': metrics.box_pushes,
            'solved': metrics.solved,
            'mean_entropy': metrics.mean_entropy,
            'std_entropy': metrics.std_entropy,
            'mean_value': metrics.mean_value,
            'std_value': metrics.std_value,
            'mean_inference_time': metrics.mean_inference_time,
            'level_size': metrics.level_size,
            'num_boxes': metrics.num_boxes,
            'num_targets': metrics.num_targets,
            'wall_ratio': metrics.wall_ratio,
            'box_density': metrics.box_density,
            'level_complexity': metrics.level_complexity
        }
    
    def get_difficulty_distribution(self) -> Dict[str, float]:
        """난이도 분포 통계 반환"""
        if len(self.metric_history) < 2:
            return {'mean': 0.5, 'std': 0.0, 'count': len(self.metric_history)}
        
        # 각 메트릭에 대해 간단한 난이도 점수 계산
        difficulties = []
        for metrics in self.metric_history:
            structural = self._assess_structural_difficulty(metrics)
            cognitive = self._assess_cognitive_difficulty(metrics)
            solution = self._assess_solution_difficulty(metrics)
            success = self._assess_success_difficulty(metrics)
            
            total = (
                self.weight_structural * structural +
                self.weight_cognitive * cognitive +
                self.weight_solution * solution +
                self.weight_success * success
            )
            difficulties.append(total)
        
        return {
            'mean': np.mean(difficulties),
            'std': np.std(difficulties),
            'min': np.min(difficulties),
            'max': np.max(difficulties),
            'count': len(difficulties)
        }
    
    def calibrate_weights(self, target_distribution: Dict[str, float] = None):
        """가중치를 목표 분포에 맞게 조정"""
        if target_distribution is None:
            target_distribution = {'mean': 0.5, 'std': 0.2}
        
        current_dist = self.get_difficulty_distribution()
        
        if current_dist['count'] < 10:
            return  # 충분한 데이터가 없음
        
        # 간단한 조정 로직
        if current_dist['mean'] > target_distribution['mean'] + 0.1:
            # 너무 어려움 - 구조적/해결 경로 가중치 감소
            self.weight_structural *= 0.9
            self.weight_solution *= 0.9
            self.weight_cognitive *= 1.1
            self.weight_success *= 1.1
        elif current_dist['mean'] < target_distribution['mean'] - 0.1:
            # 너무 쉬움 - 구조적/해결 경로 가중치 증가
            self.weight_structural *= 1.1
            self.weight_solution *= 1.1
            self.weight_cognitive *= 0.9
            self.weight_success *= 0.9
        
        # 가중치 정규화
        total_weight = self.weight_structural + self.weight_cognitive + self.weight_solution + self.weight_success
        self.weight_structural /= total_weight
        self.weight_cognitive /= total_weight
        self.weight_solution /= total_weight
        self.weight_success /= total_weight


class AdaptiveDifficultyAssessor(DifficultyAssessor):
    """
    적응형 난이도 평가기
    시간에 따라 평가 기준을 조정하여 더 정확한 난이도 측정을 수행합니다.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.assessment_history = []
        self.learning_rate = 0.01
        
    def assess_difficulty_with_feedback(self, 
                                      play_data: Dict[str, Any], 
                                      level_data: Dict[str, Any],
                                      human_difficulty: Optional[float] = None) -> Dict[str, float]:
        """
        피드백을 포함한 난이도 평가
        
        Args:
            play_data: 플레이 데이터
            level_data: 레벨 데이터
            human_difficulty: 인간이 평가한 난이도 (0-1, 선택적)
        """
        assessment = self.assess_difficulty(play_data, level_data)
        
        # 인간 피드백이 있으면 가중치 조정
        if human_difficulty is not None:
            self._update_weights_with_feedback(assessment, human_difficulty)
        
        # 평가 히스토리 업데이트
        self.assessment_history.append({
            'assessment': assessment,
            'human_difficulty': human_difficulty,
            'play_data': play_data,
            'level_data': level_data
        })
        
        # 히스토리 크기 제한
        if len(self.assessment_history) > 50:
            self.assessment_history = self.assessment_history[-50:]
        
        return assessment
    
    def _update_weights_with_feedback(self, assessment: Dict[str, float], human_difficulty: float):
        """인간 피드백을 바탕으로 가중치 업데이트"""
        predicted_difficulty = assessment['total_difficulty']
        error = human_difficulty - predicted_difficulty
        
        # 각 차원별 기여도에 따라 가중치 조정
        if abs(error) > 0.1:  # 오차가 충분히 클 때만 조정
            structural_contrib = assessment['structural_difficulty'] * self.weight_structural
            cognitive_contrib = assessment['cognitive_difficulty'] * self.weight_cognitive
            solution_contrib = assessment['solution_difficulty'] * self.weight_solution
            success_contrib = assessment['success_difficulty'] * self.weight_success
            
            # 기여도가 높은 차원의 가중치를 더 많이 조정
            total_contrib = structural_contrib + cognitive_contrib + solution_contrib + success_contrib
            
            if total_contrib > 0:
                self.weight_structural += self.learning_rate * error * (structural_contrib / total_contrib)
                self.weight_cognitive += self.learning_rate * error * (cognitive_contrib / total_contrib)
                self.weight_solution += self.learning_rate * error * (solution_contrib / total_contrib)
                self.weight_success += self.learning_rate * error * (success_contrib / total_contrib)
                
                # 가중치 정규화 및 양수 보장
                weights = np.array([self.weight_structural, self.weight_cognitive, 
                                  self.weight_solution, self.weight_success])
                weights = np.maximum(weights, 0.01)  # 최소값 보장
                weights = weights / np.sum(weights)  # 정규화
                
                self.weight_structural, self.weight_cognitive, self.weight_solution, self.weight_success = weights
    
    def get_assessment_accuracy(self) -> Dict[str, float]:
        """평가 정확도 통계 반환"""
        if not self.assessment_history:
            return {'mae': 0.0, 'rmse': 0.0, 'correlation': 0.0, 'count': 0}
        
        human_scores = []
        predicted_scores = []
        
        for entry in self.assessment_history:
            if entry['human_difficulty'] is not None:
                human_scores.append(entry['human_difficulty'])
                predicted_scores.append(entry['assessment']['total_difficulty'])
        
        if len(human_scores) < 2:
            return {'mae': 0.0, 'rmse': 0.0, 'correlation': 0.0, 'count': len(human_scores)}
        
        human_scores = np.array(human_scores)
        predicted_scores = np.array(predicted_scores)
        
        mae = np.mean(np.abs(human_scores - predicted_scores))
        rmse = np.sqrt(np.mean((human_scores - predicted_scores) ** 2))
        correlation = np.corrcoef(human_scores, predicted_scores)[0, 1]
        
        if np.isnan(correlation):
            correlation = 0.0
        
        return {
            'mae': mae,
            'rmse': rmse,
            'correlation': correlation,
            'count': len(human_scores)
        } 