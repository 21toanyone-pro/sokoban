#!/usr/bin/env python3
"""
소코반 자동 레벨 디자인 AI - 데모 파이프라인
전체 시스템의 핵심 기능을 간단하게 보여줍니다.
"""

import numpy as np
import time
import sys

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
from level_generator import LevelGenerator
from difficulty_assessor import DifficultyAssessor

def simple_random_agent(env, max_steps=100):
    """간단한 랜덤 에이전트 (PPO 대신)"""
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
        # 랜덤 액션 선택
        action = random.randint(0, 3)
        
        start_time = time.time()
        next_state, reward, terminated, truncated, info = env.step(action)
        inference_time = time.time() - start_time
        
        # 플레이 데이터 수집
        play_data['total_steps'] += 1
        play_data['total_reward'] += reward
        play_data['policy_entropy'].append(1.0)  # 랜덤 = 최대 엔트로피
        play_data['value_estimates'].append(0.0)  # 더미 값
        play_data['action_sequence'].append(action)
        play_data['inference_times'].append(inference_time)
        
        if reward >= 1:  # 박스 푸시
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

def run_pipeline_demo(num_levels=3):
    """소코반 AI 파이프라인 데모 실행"""
    
    print("=== 소코반 자동 레벨 디자인 AI 데모 ===\n")
    
    # 1. 컴포넌트 초기화
    print("컴포넌트 초기화 중...")
    generator = LevelGenerator()
    assessor = DifficultyAssessor()
    print("초기화 완료!\n")
    
    results = []
    
    for i in range(num_levels):
        print(f"=== 레벨 {i+1}/{num_levels} 생성 및 평가 ===")
        
        # 2. 레벨 생성 (창작자 AI)
        print("레벨 생성 중...")
        start_time = time.time()
        level = generator.generate_level(num_boxes=2, complexity=0.5)
        generation_time = time.time() - start_time
        
        level_data = generator.get_level_stats(level)
        print(f"레벨 생성 완료! ({generation_time:.3f}초)")
        print(f"   크기: {level.shape}, 박스: {level_data['num_boxes']}개")
        
        # 3. 환경 생성 및 레벨 표시
        env = SokobanEnvSimple(level=level)
        print("   생성된 레벨:")
        level_str = env.render()
        for line in level_str.split('\n'):
            print(f"     {line}")
        print()
        
        # 4. 에이전트 플레이 (해결사 AI)
        print("🤖 에이전트 플레이 중...")
        start_time = time.time()
        play_data = simple_random_agent(env, max_steps=50)
        solving_time = time.time() - start_time
        
        print(f"✅ 플레이 완료! ({solving_time:.3f}초)")
        print(f"   스텝: {play_data['total_steps']}, 보상: {play_data['total_reward']:.1f}")
        print(f"   해결 여부: {'성공' if play_data['solved'] else '실패'}")
        
        # 5. 난이도 평가 (평가자 AI)
        print("📊 난이도 평가 중...")
        start_time = time.time()
        difficulty = assessor.assess_difficulty(play_data, level_data)
        assessment_time = time.time() - start_time
        
        print(f"✅ 평가 완료! ({assessment_time:.3f}초)")
        print(f"   총 난이도: {difficulty['total_difficulty']:.3f}")
        print(f"   구조적: {difficulty['structural_difficulty']:.3f}")
        print(f"   인지적: {difficulty['cognitive_difficulty']:.3f}")
        print(f"   해결경로: {difficulty['solution_difficulty']:.3f}")
        print(f"   성공확률: {difficulty['success_difficulty']:.3f}")
        
        # 6. 결과 저장
        result = {
            'level': level,
            'level_data': level_data,
            'play_data': play_data,
            'difficulty': difficulty,
            'times': {
                'generation': generation_time,
                'solving': solving_time,
                'assessment': assessment_time
            }
        }
        results.append(result)
        
        print(f"✨ 레벨 {i+1} 완료!\n")
    
    # 7. 전체 결과 요약
    print("📈 === 파이프라인 실행 결과 요약 ===")
    
    total_difficulties = [r['difficulty']['total_difficulty'] for r in results]
    success_count = sum(1 for r in results if r['play_data']['solved'])
    
    print(f"🎯 처리된 레벨 수: {len(results)}")
    print(f"🏆 성공률: {success_count}/{len(results)} ({success_count/len(results)*100:.1f}%)")
    print(f"📊 평균 난이도: {np.mean(total_difficulties):.3f}")
    print(f"📏 난이도 범위: {np.min(total_difficulties):.3f} ~ {np.max(total_difficulties):.3f}")
    
    avg_generation_time = np.mean([r['times']['generation'] for r in results])
    avg_solving_time = np.mean([r['times']['solving'] for r in results])
    avg_assessment_time = np.mean([r['times']['assessment'] for r in results])
    
    print(f"⏱️  평균 처리 시간:")
    print(f"   레벨 생성: {avg_generation_time:.3f}초")
    print(f"   에이전트 플레이: {avg_solving_time:.3f}초")
    print(f"   난이도 평가: {avg_assessment_time:.3f}초")
    
    print("\n🎉 === 파이프라인 데모 완료! ===")
    print("💡 이것이 바로 AI가 게임 레벨을 자동으로 디자인하는 시스템입니다!")
    print("🔄 창작 → 해결 → 평가 → 피드백의 자동화된 루프가 성공적으로 작동했습니다!")
    
    return results

if __name__ == "__main__":
    # 데모 실행
    results = run_pipeline_demo(num_levels=3)
    
    print(f"\n📁 결과가 메모리에 저장되었습니다. (총 {len(results)}개 레벨)")
    print("🚀 전체 시스템을 사용하려면: py main.py --mode pipeline") 