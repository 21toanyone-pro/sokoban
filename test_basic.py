#!/usr/bin/env python3
"""
기본 모듈 테스트 스크립트
"""

print("=== 소코반 AI 모듈 테스트 시작 ===\n")

try:
    # 1. 소코반 환경 테스트
    print("1. 소코반 환경 테스트...")
    from sokoban_env import SokobanEnv
    env = SokobanEnv()
    print("SokobanEnv 모듈 로드 성공!")
    print(f"   - 관찰 공간: {env.observation_space.shape}")
    print(f"   - 행동 공간: {env.action_space.n}")
    print("   - 기본 레벨:")
    print(env.render())
    print()

except Exception as e:
    print(f"SokobanEnv 오류: {e}")
    print()

try:
    # 2. 레벨 생성기 테스트
    print("2. 레벨 생성기 테스트...")
    from level_generator import LevelGenerator
    generator = LevelGenerator()
    print("LevelGenerator 모듈 로드 성공!")
    
    # 간단한 레벨 생성
    level = generator.generate_level(num_boxes=2, complexity=0.3)
    print(f"   - 생성된 레벨 크기: {level.shape}")
    
    # 레벨을 환경에서 테스트
    test_env = SokobanEnv(level=level)
    print("   - 생성된 레벨:")
    print(test_env.render())
    print()

except Exception as e:
    print(f"LevelGenerator 오류: {e}")
    print()

try:
    # 3. 강화학습 에이전트 테스트 (모델 없이)
    print("3. RL 에이전트 테스트...")
    from rl_agent import PPOAgent
    agent = PPOAgent((4, 5))  # 기본 레벨 크기
    print("PPOAgent 모듈 로드 성공!")
    print(f"   - 디바이스: {agent.device}")
    print()

except Exception as e:
    print(f"PPOAgent 오류: {e}")
    print()

try:
    # 4. 난이도 평가기 테스트
    print("4. 난이도 평가기 테스트...")
    from difficulty_assessor import DifficultyAssessor
    assessor = DifficultyAssessor()
    print("DifficultyAssessor 모듈 로드 성공!")
    print()

except Exception as e:
    print(f"DifficultyAssessor 오류: {e}")
    print()

print("=== 모듈 테스트 완료 ===")
print("🎉 모든 모듈이 성공적으로 로드되었다면 main.py를 실행할 수 있습니다!") 