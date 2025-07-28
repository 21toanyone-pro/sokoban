#!/usr/bin/env python3
"""
소코반 100단계 커리큘럼 생성기
점진적으로 난이도가 증가하는 100단계의 레벨을 생성합니다.
"""

import numpy as np
import time
import sys
from typing import Dict, List, Any, Tuple
import math

# pygame 의존성 해결
sys.modules['pygame'] = type(sys)('fake_pygame')
sys.modules['pygame'].init = lambda: None
sys.modules['pygame'].display = type(sys)('fake_display')
sys.modules['pygame'].display.init = lambda: None
sys.modules['pygame'].display.set_mode = lambda x: None
sys.modules['pygame'].display.set_caption = lambda x: None
sys.modules['pygame'].display.flip = lambda: None
sys.modules['pygame'].time = type(sys)('fake_time')
sys.modules['pygame'].time.Clock = lambda: type('', (), {'tick': lambda x: None})()
sys.modules['pygame'].draw = type(sys)('fake_draw')
sys.modules['pygame'].draw.rect = lambda *args: None
sys.modules['pygame'].Rect = lambda *args: None
sys.modules['pygame'].display.quit = lambda: None
sys.modules['pygame'].quit = lambda: None

from sokoban_env_simple import SokobanEnvSimple
from level_generator import LevelGenerator, AdaptiveLevelGenerator
from difficulty_assessor import DifficultyAssessor
from level_manager import LevelManager

class CurriculumGenerator:
    """100단계 커리큘럼 레벨 생성기"""
    
    def __init__(self):
        self.generator = AdaptiveLevelGenerator()
        self.assessor = DifficultyAssessor()
        self.level_manager = LevelManager()
        
        # 커리큘럼 설정
        self.total_stages = 100
        self.levels_per_stage = 3  # 각 단계마다 3개의 레벨 생성
        
    def calculate_stage_difficulty(self, stage: int) -> float:
        """
        단계별 목표 난이도 계산
        
        Args:
            stage: 커리큘럼 단계 (1-100)
            
        Returns:
            목표 난이도 (0.0-1.0)
        """
        # 시그모이드 커브를 사용한 점진적 난이도 증가
        # 초기: 매우 쉬움 (0.1)
        # 중간: 보통 (0.5) 
        # 최종: 매우 어려움 (0.9)
        
        # 1-100을 0-1로 정규화
        normalized_stage = (stage - 1) / (self.total_stages - 1)
        
        # 시그모이드 변형: 0.1에서 0.9까지 부드럽게 증가
        sigmoid = 1 / (1 + math.exp(-6 * (normalized_stage - 0.5)))
        difficulty = 0.1 + 0.8 * sigmoid
        
        return min(0.9, max(0.1, difficulty))
    
    def calculate_stage_parameters(self, stage: int, target_difficulty: float) -> Dict[str, Any]:
        """
        단계별 생성 파라미터 계산
        
        Args:
            stage: 커리큘럼 단계
            target_difficulty: 목표 난이도
            
        Returns:
            생성 파라미터 딕셔너리
        """
        # 박스 개수: 1-6개 (단계에 따라 증가)
        max_boxes = min(6, 1 + (stage - 1) // 15)  # 15단계마다 박스 1개씩 증가
        num_boxes = max(1, min(max_boxes, int(1 + target_difficulty * 5)))
        
        # 복잡도: 난이도에 비례
        complexity = max(0.1, min(0.9, target_difficulty))
        
        # 레벨 크기: 단계에 따라 점진적 증가
        base_size = 5
        size_increase = (stage - 1) // 20  # 20단계마다 크기 1씩 증가
        level_size = min(12, base_size + size_increase)
        
        return {
            'num_boxes': num_boxes,
            'complexity': complexity,
            'min_size': (level_size, level_size),
            'max_size': (level_size + 2, level_size + 2),
            'target_difficulty': target_difficulty
        }
    
    def simple_random_agent(self, env: SokobanEnvSimple, max_steps: int = 100) -> Dict[str, Any]:
        """간단한 랜덤 에이전트로 레벨 테스트"""
        import random
        
        play_data = {
            'total_steps': 0,
            'total_reward': 0.0,
            'box_pushes': 0,
            'solved': False,
            'policy_entropy': [],
            'value_estimates': [],
            'action_sequence': [],
            'inference_times': []
        }
        
        state, info = env.reset()
        
        for step in range(max_steps):
            action = random.randint(0, 3)
            
            start_time = time.time()
            next_state, reward, terminated, truncated, info = env.step(action)
            inference_time = time.time() - start_time
            
            play_data['total_steps'] += 1
            play_data['total_reward'] += reward
            play_data['policy_entropy'].append(1.0)
            play_data['value_estimates'].append(0.0)
            play_data['action_sequence'].append(action)
            play_data['inference_times'].append(inference_time)
            
            if reward >= 1:
                play_data['box_pushes'] += 1
            
            state = next_state
            
            if terminated:
                play_data['solved'] = info.get('is_solved', False)
                break
            elif truncated:
                break
        
        # 통계 계산
        if play_data['policy_entropy']:
            play_data['mean_entropy'] = np.mean(play_data['policy_entropy'])
            play_data['std_entropy'] = np.std(play_data['policy_entropy'])
        
        if play_data['value_estimates']:
            play_data['mean_value'] = np.mean(play_data['value_estimates'])
            play_data['std_value'] = np.std(play_data['value_estimates'])
        
        if play_data['inference_times']:
            play_data['mean_inference_time'] = np.mean(play_data['inference_times'])
        
        return play_data
    
    def generate_stage_levels(self, stage: int) -> List[Dict[str, Any]]:
        """
        특정 단계의 레벨들을 생성합니다.
        
        Args:
            stage: 커리큘럼 단계 (1-100)
            
        Returns:
            생성된 레벨들의 정보 리스트
        """
        target_difficulty = self.calculate_stage_difficulty(stage)
        params = self.calculate_stage_parameters(stage, target_difficulty)
        
        print(f"단계 {stage}: 목표 난이도 {target_difficulty:.3f}")
        print(f"   파라미터: 박스 {params['num_boxes']}개, 복잡도 {params['complexity']:.2f}, 크기 {params['min_size']}")
        
        stage_levels = []
        
        for level_idx in range(self.levels_per_stage):
            print(f"   레벨 {level_idx + 1}/{self.levels_per_stage} 생성 중...")
            
            # 레벨 생성 시도 (최대 5번)
            level = None
            for attempt in range(5):
                try:
                    # 생성기 파라미터 설정
                    self.generator.min_size = params['min_size']
                    self.generator.max_size = params['max_size']
                    
                    level = self.generator.generate_level(
                        num_boxes=params['num_boxes'],
                        complexity=params['complexity']
                    )
                    
                                         if level is not None:
                         break
                         
                 except Exception as e:
                     print(f"     생성 시도 {attempt + 1} 실패: {e}")
                     continue
             
             if level is None:
                 print(f"     레벨 생성 실패")
                 continue
            
            # 환경 생성 및 테스트
            try:
                env = SokobanEnvSimple(level=level)
                play_data = self.simple_random_agent(env, max_steps=50)
                
                # 레벨 통계
                level_stats = self.generator.get_level_stats(level)
                
                # 난이도 평가
                difficulty_result = self.assessor.assess_difficulty(play_data, level_stats)
                
                # 메타데이터 구성
                metadata = {
                    'curriculum_stage': stage,
                    'target_difficulty': target_difficulty,
                    'measured_difficulty': difficulty_result['total_difficulty'],
                    'generation_parameters': params,
                    'level_stats': level_stats,
                    'play_data': play_data,
                    'difficulty_breakdown': difficulty_result,
                    'generation_time': time.time()
                }
                
                # 레벨 저장
                level_id = f"stage_{stage:03d}_level_{level_idx + 1:02d}"
                saved_id = self.level_manager.save_level(level, metadata, level_id)
                
                if saved_id:
                    stage_levels.append({
                        'level_id': saved_id,
                        'level': level,
                        'metadata': metadata,
                        'target_difficulty': target_difficulty,
                        'measured_difficulty': difficulty_result['total_difficulty'],
                        'solved': play_data['solved']
                    })
                    
                    print(f"     ✅ 레벨 저장: {saved_id} (난이도: {difficulty_result['total_difficulty']:.3f})")
                else:
                    print(f"     ❌ 레벨 저장 실패")
                    
            except Exception as e:
                print(f"     ❌ 레벨 테스트 실패: {e}")
                continue
        
        return stage_levels
    
    def generate_full_curriculum(self, start_stage: int = 1, end_stage: int = 100) -> Dict[str, Any]:
        """
        전체 커리큘럼을 생성합니다.
        
        Args:
            start_stage: 시작 단계
            end_stage: 종료 단계
            
        Returns:
            커리큘럼 생성 결과
        """
        print(f"=== 소코반 100단계 커리큘럼 생성 시작 ===")
        print(f"생성 범위: 단계 {start_stage} ~ {end_stage}")
        print(f"단계당 레벨 수: {self.levels_per_stage}개")
        print(f"총 예상 레벨 수: {(end_stage - start_stage + 1) * self.levels_per_stage}개\n")
        
        curriculum_results = {
            'stages': {},
            'statistics': {
                'total_stages': 0,
                'total_levels': 0,
                'success_rate': 0.0,
                'difficulty_progression': [],
                'generation_times': []
            },
            'start_time': time.time()
        }
        
        for stage in range(start_stage, end_stage + 1):
            stage_start_time = time.time()
            
            print(f"\n🔄 === 단계 {stage}/{end_stage} ===")
            
            try:
                stage_levels = self.generate_stage_levels(stage)
                stage_time = time.time() - stage_start_time
                
                # 단계 통계
                stage_stats = {
                    'levels_generated': len(stage_levels),
                    'target_difficulty': self.calculate_stage_difficulty(stage),
                    'average_measured_difficulty': 0.0,
                    'success_rate': 0.0,
                    'generation_time': stage_time
                }
                
                if stage_levels:
                    difficulties = [l['measured_difficulty'] for l in stage_levels]
                    successes = [l['solved'] for l in stage_levels]
                    
                    stage_stats['average_measured_difficulty'] = np.mean(difficulties)
                    stage_stats['success_rate'] = np.mean(successes)
                
                curriculum_results['stages'][stage] = {
                    'levels': stage_levels,
                    'statistics': stage_stats
                }
                
                curriculum_results['statistics']['total_stages'] += 1
                curriculum_results['statistics']['total_levels'] += len(stage_levels)
                curriculum_results['statistics']['difficulty_progression'].append(stage_stats['average_measured_difficulty'])
                curriculum_results['statistics']['generation_times'].append(stage_time)
                
                print(f"✅ 단계 {stage} 완료: {len(stage_levels)}개 레벨, 평균 난이도 {stage_stats['average_measured_difficulty']:.3f}")
                
                # 중간 진행률 표시 (10단계마다)
                if stage % 10 == 0:
                    progress = (stage - start_stage + 1) / (end_stage - start_stage + 1) * 100
                    print(f"📊 전체 진행률: {progress:.1f}% ({stage}/{end_stage} 단계)")
                
            except Exception as e:
                print(f"❌ 단계 {stage} 생성 실패: {e}")
                continue
        
        # 최종 통계 계산
        total_time = time.time() - curriculum_results['start_time']
        
        if curriculum_results['statistics']['total_levels'] > 0:
            all_difficulties = []
            all_successes = []
            
            for stage_data in curriculum_results['stages'].values():
                for level_info in stage_data['levels']:
                    all_difficulties.append(level_info['measured_difficulty'])
                    all_successes.append(level_info['solved'])
            
            curriculum_results['statistics']['success_rate'] = np.mean(all_successes)
            curriculum_results['statistics']['difficulty_range'] = {
                'min': np.min(all_difficulties),
                'max': np.max(all_difficulties),
                'mean': np.mean(all_difficulties),
                'std': np.std(all_difficulties)
            }
        
        curriculum_results['statistics']['total_time'] = total_time
        curriculum_results['end_time'] = time.time()
        
        # 결과 요약 출력
        self._print_curriculum_summary(curriculum_results)
        
        return curriculum_results
    
    def _print_curriculum_summary(self, results: Dict[str, Any]):
        """커리큘럼 생성 결과 요약 출력"""
        stats = results['statistics']
        
        print(f"\n🎉 === 커리큘럼 생성 완료! ===")
        print(f"🎯 총 단계 수: {stats['total_stages']}")
        print(f"📦 총 레벨 수: {stats['total_levels']}")
        print(f"🏆 전체 성공률: {stats['success_rate']:.1%}")
        print(f"⏱️  총 소요 시간: {stats['total_time']:.1f}초")
        
        if 'difficulty_range' in stats:
            diff_range = stats['difficulty_range']
            print(f"📊 난이도 범위: {diff_range['min']:.3f} ~ {diff_range['max']:.3f}")
            print(f"📈 평균 난이도: {diff_range['mean']:.3f} (±{diff_range['std']:.3f})")
        
        print(f"\n💾 저장된 레벨들은 saved_levels/ 폴더에서 확인할 수 있습니다!")
        print(f"🔍 레벨 검색: level_manager.py 사용")
        print(f"🎮 레벨 플레이: sokoban_env_simple.py 사용")


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='소코반 100단계 커리큘럼 생성기')
    parser.add_argument('--start', type=int, default=1, help='시작 단계 (기본: 1)')
    parser.add_argument('--end', type=int, default=100, help='종료 단계 (기본: 100)')
    parser.add_argument('--quick', action='store_true', help='빠른 테스트 (1-10단계만)')
    
    args = parser.parse_args()
    
    if args.quick:
        start_stage, end_stage = 1, 10
        print("🚀 빠른 테스트 모드: 1-10단계만 생성합니다.")
    else:
        start_stage, end_stage = args.start, args.end
    
    # 커리큘럼 생성기 실행
    generator = CurriculumGenerator()
    results = generator.generate_full_curriculum(start_stage, end_stage)
    
    return results


if __name__ == "__main__":
    main() 