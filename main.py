#!/usr/bin/env python3
"""
소코반 자동 레벨 디자인 AI - 메인 파이프라인
Author: AI Sokoban Level Designer
"""

import os
import sys
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Any
import json
from datetime import datetime
import logging

# 프로젝트 모듈 import
from sokoban_env import SokobanEnv
from level_generator import LevelGenerator, AdaptiveLevelGenerator
from rl_agent import PPOAgent
from difficulty_assessor import DifficultyAssessor, AdaptiveDifficultyAssessor

class SokobanPipeline:
    """
    소코반 자동 레벨 디자인 파이프라인
    
    창작 → 해결 → 평가 → 피드백의 자동화된 루프를 관리합니다.
    """
    
    def __init__(self, 
                 use_adaptive: bool = True,
                 agent_model_path: str = "sokoban_ppo.pth",
                 results_dir: str = "results"):
        
        self.use_adaptive = use_adaptive
        self.agent_model_path = agent_model_path
        self.results_dir = results_dir
        
        # 결과 디렉토리 생성
        os.makedirs(results_dir, exist_ok=True)
        
        # 로깅 설정
        self._setup_logging()
        
        # 컴포넌트 초기화
        self._initialize_components()
        
        # 파이프라인 통계
        self.pipeline_stats = {
            'total_levels_generated': 0,
            'successful_evaluations': 0,
            'average_difficulty': 0.0,
            'generation_times': [],
            'evaluation_times': [],
            'difficulty_history': []
        }
    
    def _setup_logging(self):
        """로깅 시스템 설정"""
        log_file = os.path.join(self.results_dir, f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("소코반 자동 레벨 디자인 파이프라인이 초기화되었습니다.")
    
    def _initialize_components(self):
        """파이프라인 컴포넌트 초기화"""
        self.logger.info("컴포넌트 초기화 중...")
        
        # 1. 레벨 생성기 초기화
        if self.use_adaptive:
            self.level_generator = AdaptiveLevelGenerator()
            self.logger.info("적응형 레벨 생성기를 초기화했습니다.")
        else:
            self.level_generator = LevelGenerator()
            self.logger.info("기본 레벨 생성기를 초기화했습니다.")
        
        # 2. 환경 초기화 (임시)
        temp_env = SokobanEnv()
        
        # 3. RL 에이전트 초기화
        self.agent = PPOAgent(temp_env.observation_space.shape)
        
        # 사전 훈련된 모델 로드 시도
        if os.path.exists(self.agent_model_path):
            self.agent.load_model(self.agent_model_path)
            self.logger.info(f"사전 훈련된 모델을 로드했습니다: {self.agent_model_path}")
        else:
            self.logger.warning(f"모델 파일을 찾을 수 없습니다: {self.agent_model_path}")
            self.logger.info("무작위 초기화된 에이전트를 사용합니다.")
        
        # 4. 난이도 평가기 초기화
        if self.use_adaptive:
            self.difficulty_assessor = AdaptiveDifficultyAssessor()
            self.logger.info("적응형 난이도 평가기를 초기화했습니다.")
        else:
            self.difficulty_assessor = DifficultyAssessor()
            self.logger.info("기본 난이도 평가기를 초기화했습니다.")
        
        self.logger.info("모든 컴포넌트가 성공적으로 초기화되었습니다.")
    
    def run_pipeline(self, 
                     num_iterations: int = 10,
                     target_difficulty: Optional[float] = None,
                     save_levels: bool = True,
                     visualize: bool = False) -> Dict[str, Any]:
        """
        파이프라인 실행
        
        Args:
            num_iterations: 실행할 반복 횟수
            target_difficulty: 목표 난이도 (None이면 자동)
            save_levels: 생성된 레벨 저장 여부
            visualize: 결과 시각화 여부
        
        Returns:
            파이프라인 실행 결과
        """
        self.logger.info(f"파이프라인 실행 시작: {num_iterations}회 반복")
        
        results = []
        
        for iteration in range(num_iterations):
            self.logger.info(f"=== 반복 {iteration + 1}/{num_iterations} ===")
            
            try:
                # 1. 레벨 생성
                level_result = self._generate_level(target_difficulty)
                
                # 2. 에이전트로 레벨 해결 시도
                play_result = self._solve_level(level_result['level'])
                
                # 3. 난이도 평가
                assessment_result = self._assess_difficulty(
                    play_result['play_data'], 
                    level_result['level_data']
                )
                
                # 4. 피드백 적용
                self._apply_feedback(
                    level_result['level'], 
                    assessment_result['total_difficulty']
                )
                
                # 결과 통합
                iteration_result = {
                    'iteration': iteration + 1,
                    'level_generation': level_result,
                    'play_result': play_result,
                    'difficulty_assessment': assessment_result,
                    'timestamp': datetime.now().isoformat()
                }
                
                results.append(iteration_result)
                
                # 통계 업데이트
                self._update_statistics(iteration_result)
                
                # 레벨 저장
                if save_levels:
                    self._save_level(iteration_result, iteration + 1)
                
                self.logger.info(f"반복 {iteration + 1} 완료 - 난이도: {assessment_result['total_difficulty']:.3f}")
                
            except Exception as e:
                self.logger.error(f"반복 {iteration + 1} 중 오류 발생: {e}")
                continue
        
        # 최종 결과 생성
        final_results = self._generate_final_results(results)
        
        # 결과 저장
        self._save_results(final_results)
        
        # 시각화
        if visualize:
            self._visualize_results(results)
        
        self.logger.info("파이프라인 실행 완료!")
        
        return final_results
    
    def _generate_level(self, target_difficulty: Optional[float] = None) -> Dict[str, Any]:
        """레벨 생성"""
        start_time = time.time()
        
        try:
            if target_difficulty is not None and hasattr(self.level_generator, 'generate_level_with_target_difficulty'):
                level = self.level_generator.generate_level_with_target_difficulty(target_difficulty)
            else:
                level = self.level_generator.generate_level()
            
            level_data = self.level_generator.get_level_stats(level)
            generation_time = time.time() - start_time
            
            self.pipeline_stats['generation_times'].append(generation_time)
            self.pipeline_stats['total_levels_generated'] += 1
            
            return {
                'level': level,
                'level_data': level_data,
                'generation_time': generation_time,
                'target_difficulty': target_difficulty
            }
            
        except Exception as e:
            self.logger.error(f"레벨 생성 중 오류: {e}")
            raise
    
    def _solve_level(self, level: np.ndarray) -> Dict[str, Any]:
        """에이전트로 레벨 해결 시도"""
        start_time = time.time()
        
        try:
            # 레벨로 환경 생성
            env = SokobanEnv(level=level, max_steps=500)
            
            # 에이전트가 플레이하며 데이터 수집
            play_data = self.agent.get_play_data(env)
            
            solving_time = time.time() - start_time
            
            env.close()
            
            return {
                'play_data': play_data,
                'solving_time': solving_time,
                'success': play_data.get('solved', False)
            }
            
        except Exception as e:
            self.logger.error(f"레벨 해결 시도 중 오류: {e}")
            raise
    
    def _assess_difficulty(self, play_data: Dict[str, Any], level_data: Dict[str, Any]) -> Dict[str, Any]:
        """난이도 평가"""
        start_time = time.time()
        
        try:
            assessment = self.difficulty_assessor.assess_difficulty(play_data, level_data)
            assessment_time = time.time() - start_time
            
            self.pipeline_stats['evaluation_times'].append(assessment_time)
            self.pipeline_stats['successful_evaluations'] += 1
            self.pipeline_stats['difficulty_history'].append(assessment['total_difficulty'])
            
            assessment['assessment_time'] = assessment_time
            
            return assessment
            
        except Exception as e:
            self.logger.error(f"난이도 평가 중 오류: {e}")
            raise
    
    def _apply_feedback(self, level: np.ndarray, measured_difficulty: float):
        """피드백 적용"""
        try:
            if hasattr(self.level_generator, 'update_difficulty_feedback'):
                self.level_generator.update_difficulty_feedback(level, measured_difficulty)
            
            if hasattr(self.difficulty_assessor, 'calibrate_weights'):
                self.difficulty_assessor.calibrate_weights()
                
        except Exception as e:
            self.logger.error(f"피드백 적용 중 오류: {e}")
    
    def _update_statistics(self, iteration_result: Dict[str, Any]):
        """통계 업데이트"""
        difficulty = iteration_result['difficulty_assessment']['total_difficulty']
        
        # 평균 난이도 업데이트
        total_assessments = len(self.pipeline_stats['difficulty_history'])
        if total_assessments > 0:
            self.pipeline_stats['average_difficulty'] = sum(self.pipeline_stats['difficulty_history']) / total_assessments
    
    def _save_level(self, iteration_result: Dict[str, Any], iteration: int):
        """개별 레벨 저장"""
        try:
            level_dir = os.path.join(self.results_dir, "levels")
            os.makedirs(level_dir, exist_ok=True)
            
            level_file = os.path.join(level_dir, f"level_{iteration:03d}.json")
            
            # numpy 배열을 리스트로 변환
            save_data = {
                'iteration': iteration,
                'level': iteration_result['level_generation']['level'].tolist(),
                'level_stats': iteration_result['level_generation']['level_data'],
                'play_data': iteration_result['play_result']['play_data'],
                'difficulty_assessment': iteration_result['difficulty_assessment'],
                'timestamp': iteration_result['timestamp']
            }
            
            with open(level_file, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            self.logger.error(f"레벨 저장 중 오류: {e}")
    
    def _generate_final_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """최종 결과 생성"""
        if not results:
            return {'error': '결과가 없습니다.'}
        
        # 성공률 계산
        successful_solves = sum(1 for r in results if r['play_result']['success'])
        success_rate = successful_solves / len(results)
        
        # 난이도 분포
        difficulties = [r['difficulty_assessment']['total_difficulty'] for r in results]
        
        # 세부 난이도 분석
        structural_difficulties = [r['difficulty_assessment']['structural_difficulty'] for r in results]
        cognitive_difficulties = [r['difficulty_assessment']['cognitive_difficulty'] for r in results]
        solution_difficulties = [r['difficulty_assessment']['solution_difficulty'] for r in results]
        success_difficulties = [r['difficulty_assessment']['success_difficulty'] for r in results]
        
        return {
            'summary': {
                'total_iterations': len(results),
                'success_rate': success_rate,
                'average_difficulty': np.mean(difficulties),
                'difficulty_std': np.std(difficulties),
                'min_difficulty': np.min(difficulties),
                'max_difficulty': np.max(difficulties)
            },
            'detailed_analysis': {
                'structural_difficulty': {
                    'mean': np.mean(structural_difficulties),
                    'std': np.std(structural_difficulties)
                },
                'cognitive_difficulty': {
                    'mean': np.mean(cognitive_difficulties),
                    'std': np.std(cognitive_difficulties)
                },
                'solution_difficulty': {
                    'mean': np.mean(solution_difficulties),
                    'std': np.std(solution_difficulties)
                },
                'success_difficulty': {
                    'mean': np.mean(success_difficulties),
                    'std': np.std(success_difficulties)
                }
            },
            'performance_metrics': {
                'average_generation_time': np.mean(self.pipeline_stats['generation_times']),
                'average_evaluation_time': np.mean(self.pipeline_stats['evaluation_times']),
                'total_levels_generated': self.pipeline_stats['total_levels_generated'],
                'successful_evaluations': self.pipeline_stats['successful_evaluations']
            },
            'pipeline_stats': self.pipeline_stats,
            'individual_results': results
        }
    
    def _save_results(self, final_results: Dict[str, Any]):
        """최종 결과 저장"""
        try:
            results_file = os.path.join(self.results_dir, f"pipeline_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(final_results, f, indent=2, ensure_ascii=False, default=str)
            
            self.logger.info(f"결과가 저장되었습니다: {results_file}")
            
        except Exception as e:
            self.logger.error(f"결과 저장 중 오류: {e}")
    
    def _visualize_results(self, results: List[Dict[str, Any]]):
        """결과 시각화"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('소코반 자동 레벨 디자인 파이프라인 결과', fontsize=16)
            
            # 1. 난이도 분포
            difficulties = [r['difficulty_assessment']['total_difficulty'] for r in results]
            axes[0, 0].hist(difficulties, bins=20, alpha=0.7, color='blue')
            axes[0, 0].set_title('난이도 분포')
            axes[0, 0].set_xlabel('난이도')
            axes[0, 0].set_ylabel('빈도')
            
            # 2. 난이도 진화
            iterations = [r['iteration'] for r in results]
            axes[0, 1].plot(iterations, difficulties, 'o-', color='red')
            axes[0, 1].set_title('난이도 변화')
            axes[0, 1].set_xlabel('반복')
            axes[0, 1].set_ylabel('난이도')
            
            # 3. 성공률
            successes = [1 if r['play_result']['success'] else 0 for r in results]
            axes[1, 0].plot(iterations, np.cumsum(successes) / np.arange(1, len(successes) + 1), 'o-', color='green')
            axes[1, 0].set_title('누적 성공률')
            axes[1, 0].set_xlabel('반복')
            axes[1, 0].set_ylabel('성공률')
            
            # 4. 차원별 난이도
            structural = [r['difficulty_assessment']['structural_difficulty'] for r in results]
            cognitive = [r['difficulty_assessment']['cognitive_difficulty'] for r in results]
            solution = [r['difficulty_assessment']['solution_difficulty'] for r in results]
            
            axes[1, 1].plot(iterations, structural, 'o-', label='구조적', alpha=0.7)
            axes[1, 1].plot(iterations, cognitive, 's-', label='인지적', alpha=0.7)
            axes[1, 1].plot(iterations, solution, '^-', label='해결 경로', alpha=0.7)
            axes[1, 1].set_title('차원별 난이도')
            axes[1, 1].set_xlabel('반복')
            axes[1, 1].set_ylabel('난이도')
            axes[1, 1].legend()
            
            plt.tight_layout()
            
            # 저장
            plot_file = os.path.join(self.results_dir, f"pipeline_visualization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.show()
            
            self.logger.info(f"시각화 결과가 저장되었습니다: {plot_file}")
            
        except Exception as e:
            self.logger.error(f"시각화 중 오류: {e}")
    
    def train_agent(self, num_episodes: int = 500):
        """에이전트 훈련"""
        self.logger.info(f"에이전트 훈련 시작: {num_episodes} 에피소드")
        
        # 기본 환경으로 훈련
        env = SokobanEnv()
        stats = self.agent.train(env, num_episodes=num_episodes, model_path=self.agent_model_path)
        
        self.logger.info("에이전트 훈련 완료!")
        return stats


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='소코반 자동 레벨 디자인 AI 파이프라인')
    
    parser.add_argument('--mode', type=str, choices=['train', 'generate', 'pipeline'], default='pipeline',
                       help='실행 모드: train(에이전트 훈련), generate(레벨 생성만), pipeline(전체 파이프라인)')
    parser.add_argument('--iterations', type=int, default=10,
                       help='파이프라인 반복 횟수')
    parser.add_argument('--target-difficulty', type=float, default=None,
                       help='목표 난이도 (0.0-1.0)')
    parser.add_argument('--model-path', type=str, default='sokoban_ppo.pth',
                       help='에이전트 모델 파일 경로')
    parser.add_argument('--results-dir', type=str, default='results',
                       help='결과 저장 디렉토리')
    parser.add_argument('--train-episodes', type=int, default=500,
                       help='훈련 에피소드 수')
    parser.add_argument('--adaptive', action='store_true',
                       help='적응형 컴포넌트 사용')
    parser.add_argument('--visualize', action='store_true',
                       help='결과 시각화')
    
    args = parser.parse_args()
    
    # 파이프라인 초기화
    pipeline = SokobanPipeline(
        use_adaptive=args.adaptive,
        agent_model_path=args.model_path,
        results_dir=args.results_dir
    )
    
    if args.mode == 'train':
        # 에이전트 훈련만
        pipeline.train_agent(args.train_episodes)
        
    elif args.mode == 'generate':
        # 레벨 생성만
        level = pipeline.level_generator.generate_level()
        print("생성된 레벨:")
        env = SokobanEnv(level=level)
        print(env.render())
        
    elif args.mode == 'pipeline':
        # 전체 파이프라인 실행
        results = pipeline.run_pipeline(
            num_iterations=args.iterations,
            target_difficulty=args.target_difficulty,
            save_levels=True,
            visualize=args.visualize
        )
        
        print("\n=== 파이프라인 실행 결과 ===")
        print(f"총 반복 횟수: {results['summary']['total_iterations']}")
        print(f"성공률: {results['summary']['success_rate']:.1%}")
        print(f"평균 난이도: {results['summary']['average_difficulty']:.3f}")
        print(f"난이도 표준편차: {results['summary']['difficulty_std']:.3f}")


if __name__ == "__main__":
    main() 