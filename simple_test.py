#!/usr/bin/env python3
"""
간단한 라이브러리 테스트
"""

print("=== 간단한 라이브러리 테스트 ===")

# 기본 라이브러리 테스트
try:
    import numpy as np
    print("NumPy 설치됨:", np.__version__)
except ImportError:
    print("NumPy가 설치되지 않음")

try:
    import gymnasium as gym
    print("Gymnasium 설치됨:", gym.__version__)
except ImportError:
    print("Gymnasium이 설치되지 않음")

try:
    import torch
    print("PyTorch 설치됨:", torch.__version__)
except ImportError:
    print("PyTorch가 설치되지 않음")

try:
    import matplotlib
    print("Matplotlib 설치됨:", matplotlib.__version__)
except ImportError:
    print("Matplotlib이 설치되지 않음")

# 기본 소코반 환경 테스트
print("\n=== 소코반 환경 테스트 ===")
try:
    import numpy as np
    
    # 간단한 소코반 레벨 정의
    level = np.array([
        [1, 1, 1, 1, 1],
        [1, 4, 0, 2, 1], 
        [1, 0, 0, 3, 1],
        [1, 1, 1, 1, 1]
    ])
    
    print("기본 레벨 생성 성공!")
    print("레벨:")
    
    # 텍스트 렌더링
    symbols = {0: ' ', 1: '#', 2: '$', 3: '.', 4: '@'}
    for row in level:
        line = ''.join(symbols.get(cell, '?') for cell in row)
        print(f"  {line}")
    
    print("\n# = 벽, @ = 플레이어, $ = 박스, . = 목표")
    
except Exception as e:
    print(f"기본 테스트 실패: {e}")

print("\n=== 테스트 완료 ===")
print("라이브러리가 정상적으로 설치되었다면 main.py를 실행할 수 있습니다.") 