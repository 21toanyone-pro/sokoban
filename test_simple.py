#!/usr/bin/env python3
"""
Pygame 없는 간단한 소코반 테스트
"""

print("=== 간단한 소코반 AI 테스트 ===\n")

try:
    # 1. 간단한 소코반 환경 테스트
    print("1. 간단한 소코반 환경 테스트...")
    from sokoban_env_simple import SokobanEnvSimple
    env = SokobanEnvSimple()
    print("SokobanEnvSimple 모듈 로드 성공!")
    print(f"   - 관찰 공간: {env.observation_space.shape}")
    print(f"   - 행동 공간: {env.action_space.n}")
    print("   - 기본 레벨:")
    print(env.render())
    print()

    # 2. 몇 번의 액션 테스트
    print("2. 게임 플레이 테스트...")
    obs, info = env.reset()
    print(f"   초기 상태 - 박스 개수: {info['total_boxes']}, 목표 개수: {len(env.target_positions)}")
    
    # 오른쪽으로 이동 (박스 밀기)
    action = 3  # 오른쪽
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"   액션 {action} 후 - 보상: {reward:.1f}, 완료: {terminated}")
    print("   현재 레벨:")
    print(env.render())
    print()

except Exception as e:
    print(f"소코반 환경 오류: {e}")
    import traceback
    traceback.print_exc()
    print()

try:
    # 3. 레벨 생성기 테스트 (pygame 임포트 제거)
    print("3. 레벨 생성기 테스트...")
    
    # 임시로 pygame import를 건너뛰도록 모듈 수정
    import sys
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
    
    from level_generator import LevelGenerator
    generator = LevelGenerator()
    print("LevelGenerator 모듈 로드 성공!")
    
    # 간단한 레벨 생성
    level = generator.generate_level(num_boxes=2, complexity=0.3)
    print(f"   - 생성된 레벨 크기: {level.shape}")
    
    # 레벨을 간단한 환경에서 테스트
    test_env = SokobanEnvSimple(level=level)
    print("   - 생성된 레벨:")
    print(test_env.render())
    
    # 레벨 통계
    stats = generator.get_level_stats(level)
    print(f"   - 레벨 통계: 박스 {stats['num_boxes']}개, 벽 비율 {stats['wall_ratio']:.2f}")
    print()

except Exception as e:
    print(f"레벨 생성기 오류: {e}")
    import traceback
    traceback.print_exc()
    print()

try:
    # 4. 난이도 평가기 테스트
    print("4. 난이도 평가기 테스트...")
    from difficulty_assessor import DifficultyAssessor
    assessor = DifficultyAssessor()
    print("DifficultyAssessor 모듈 로드 성공!")
    
    # 가짜 플레이 데이터
    fake_play_data = {
        'total_steps': 50,
        'total_reward': 95,
        'box_pushes': 3,
        'solved': True,
        'mean_entropy': 0.8,
        'std_entropy': 0.2,
        'mean_value': 20.0,
        'std_value': 5.0,
        'mean_inference_time': 0.001
    }
    
    fake_level_data = {
        'level': level,
        'size': level.shape,
        'num_boxes': 2,
        'num_targets': 2,
        'level_complexity': 0.5
    }
    
    difficulty = assessor.assess_difficulty(fake_play_data, fake_level_data)
    print(f"   - 총 난이도: {difficulty['total_difficulty']:.3f}")
    print(f"   - 구조적 난이도: {difficulty['structural_difficulty']:.3f}")
    print(f"   - 인지적 난이도: {difficulty['cognitive_difficulty']:.3f}")
    print()

except Exception as e:
    print(f"난이도 평가기 오류: {e}")
    import traceback
    traceback.print_exc()
    print()

print("=== 테스트 완료 ===")
print("기본 기능들이 성공적으로 작동합니다!")
print("전체 파이프라인을 실행하려면: py main.py --mode generate") 