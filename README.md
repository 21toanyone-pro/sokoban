# 소코반 자동 레벨 디자인 AI

파이썬과 강화학습을 이용한 소코반 자동 레벨 디자인 AI 시스템

## 목차

- [프로젝트 개요](#프로젝트-개요)
- [시스템 아키텍처](#시스템-아키텍처)
- [설치 및 환경 설정](#설치-및-환경-설정)
- [사용 방법](#사용-방법)
- [모듈별 상세 설명](#모듈별-상세-설명)
- [결과 및 평가](#결과-및-평가)
- [향후 개발 계획](#향후-개발-계획)

## 프로젝트 개요

본 프로젝트는 **AI가 스스로 소코반 게임 레벨을 디자인하고 그 난이도를 평가하는 자동화된 시스템**입니다. 두 가지 핵심 AI인 **해결사 에이전트(Solver Agent)**와 **창작자 에이전트(Creator Agent)**의 유기적인 상호작용을 통해 작동합니다.

### 핵심 특징

- **자동화된 폐쇄 루프**: 창작 → 해결 → 평가 → 피드백
- **역방향 생성 알고리즘**: 풀이 가능성 100% 보장
- **다차원 난이도 평가**: 구조적/인지적/해결경로/성공확률 복합 평가
- **적응형 학습**: 사용자 피드백을 통한 지속적 개선

## 시스템 아키텍처

```
Level Generator → RL Agent (Solver) → Difficulty Assessor → Feedback Loop
     ↑                                                           ↓
     ←←←←←←←←←← Adaptive Parameters ←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←
```

### 핵심 모듈

1. **Level Generator** (창작자): 역방향 생성으로 풀이 가능한 레벨 생성
2. **RL Agent (Solver)** (해결사): PPO 알고리즘으로 퍼즐 해결
3. **Difficulty Assessor** (평가자): 다차원 난이도 정량화
4. **Sokoban Environment**: OpenAI Gym 호환 게임 환경
5. **Main Pipeline**: 전체 시스템 오케스트레이션

## 설치 및 환경 설정

### 필수 요구사항

- Python 3.8 이상
- CUDA 지원 GPU (선택사항, CPU로도 실행 가능)

### 1. 저장소 클론

```bash
git clone https://github.com/your-username/sokoban-ai-designer.git
cd sokoban-ai-designer
```

### 2. 가상환경 생성 및 활성화

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate
```

### 3. 필요 라이브러리 설치

```bash
pip install -r requirements.txt
```

## 사용 방법

### 기본 파이프라인 실행

```bash
python main.py --mode pipeline --iterations 10 --adaptive --visualize
```

### 에이전트 훈련

```bash
python main.py --mode train --train-episodes 500
```

### 레벨 생성만 실행

```bash
python main.py --mode generate
```

### 고급 옵션

```bash
python main.py \
    --mode pipeline \
    --iterations 20 \
    --target-difficulty 0.7 \
    --adaptive \
    --visualize \
    --results-dir my_results
```

### 명령행 옵션

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--mode` | 실행 모드 (train/generate/pipeline) | pipeline |
| `--iterations` | 파이프라인 반복 횟수 | 10 |
| `--target-difficulty` | 목표 난이도 (0.0-1.0) | None |
| `--adaptive` | 적응형 컴포넌트 사용 | False |
| `--visualize` | 결과 시각화 | False |
| `--model-path` | 에이전트 모델 파일 경로 | sokoban_ppo.pth |
| `--results-dir` | 결과 저장 디렉토리 | results |
| `--train-episodes` | 훈련 에피소드 수 | 500 |

## 모듈별 상세 설명

### SokobanEnv (`sokoban_env.py`)

OpenAI Gym 호환 소코반 게임 환경

**주요 기능:**
- 표준화된 게임 상태 관리
- 다양한 렌더링 모드 (텍스트/Pygame/RGB 배열)
- 보상 함수 정의 및 게임 종료 조건 처리

**게임 요소:**
- 0: 빈 공간, 1: 벽, 2: 박스, 3: 목표 지점
- 4: 플레이어, 5: 목표 위 박스, 6: 목표 위 플레이어

### LevelGenerator (`level_generator.py`)

역방향 생성 알고리즘을 사용한 레벨 자동 생성

**핵심 알고리즘:**
1. 기본 방 구조 생성
2. 목표 상태에서 역방향으로 박스 이동
3. 연결성 검증 및 유효성 확인

**특징:**
- 풀이 가능성 100% 보장
- 복잡도 조절 가능
- 적응형 생성 파라미터 조정

### PPOAgent (`rl_agent.py`)

Proximal Policy Optimization 기반 강화학습 에이전트

**신경망 구조:**
- CNN 기반 Actor-Critic 아키텍처
- 입력: 게임 보드 상태 (높이 × 너비)
- 출력: 행동 확률 분포 + 상태 가치

**훈련 기능:**
- 경험 리플레이 버퍼
- PPO 클리핑 및 GAE (Generalized Advantage Estimation)
- 커리큘럼 학습 지원

### DifficultyAssessor (`difficulty_assessor.py`)

다차원 난이도 평가 시스템

**평가 차원:**
1. **구조적 복잡도**: 레벨 크기, 박스 개수, 벽 비율
2. **인지적 부하**: 정책 엔트로피, 가치 불확실성, 추론 시간
3. **해결 경로 복잡도**: 스텝 수, 박스 푸시 횟수, 효율성
4. **성공 확률**: 해결 가능성 및 시도 여부

**적응형 평가:**
- 사용자 피드백 학습
- 가중치 자동 조정
- 평가 정확도 모니터링

### SokobanPipeline (`main.py`)

전체 시스템 오케스트레이션 및 파이프라인 관리

**파이프라인 단계:**
1. **레벨 생성**: 목표 난이도에 맞는 레벨 생성
2. **해결 시도**: 에이전트가 레벨 플레이 및 데이터 수집
3. **난이도 평가**: 플레이 데이터 기반 난이도 정량화
4. **피드백 적용**: 결과를 생성기에 반영하여 개선

## 결과 및 평가

### 결과 파일 구조

```
results/
├── pipeline_YYYYMMDD_HHMMSS.log          # 실행 로그
├── pipeline_results_YYYYMMDD_HHMMSS.json # 종합 결과
├── pipeline_visualization_YYYYMMDD_HHMMSS.png # 시각화
└── levels/
    ├── level_001.json                     # 개별 레벨 데이터
    ├── level_002.json
    └── ...
```

### 평가 지표

- **생성 성공률**: 유효한 레벨 생성 비율
- **해결 성공률**: 에이전트의 퍼즐 해결 비율
- **난이도 정확도**: 목표 난이도 대비 실제 측정 난이도
- **다양성 점수**: 생성된 레벨들의 구조적 다양성

### 시각화 차트

1. **난이도 분포**: 생성된 레벨들의 난이도 히스토그램
2. **난이도 진화**: 반복에 따른 난이도 변화 추이
3. **성공률 변화**: 누적 해결 성공률 그래프
4. **차원별 분석**: 구조적/인지적/해결경로 난이도 비교

## 고급 사용법

### 커스텀 환경 설정

```python
from sokoban_env import SokobanEnv
import numpy as np

# 커스텀 레벨 정의
custom_level = np.array([
    [1, 1, 1, 1, 1],
    [1, 4, 0, 2, 1],
    [1, 0, 0, 3, 1],
    [1, 1, 1, 1, 1]
])

env = SokobanEnv(level=custom_level, max_steps=200)
```

### 에이전트 평가

```python
from rl_agent import PPOAgent

agent = PPOAgent((4, 5))  # 레벨 크기에 맞게 조정
agent.load_model("sokoban_ppo.pth")

results = agent.evaluate(env, num_episodes=10, render=True)
print(f"성공률: {results['success_rate']:.1%}")
```

### 난이도 평가기 커스터마이징

```python
from difficulty_assessor import DifficultyAssessor

# 가중치 조정
assessor = DifficultyAssessor(
    weight_structural=0.4,    # 구조적 복잡도
    weight_cognitive=0.3,     # 인지적 부하
    weight_solution=0.2,      # 해결 경로
    weight_success=0.1        # 성공 확률
)
```

## 핵심 기술 및 알고리즘

### 1. 역방향 생성 (Reverse Generation)

```python
# 해결된 상태에서 시작
final_state = create_solved_state()

# 역방향으로 박스 이동하여 초기 상태 생성
for _ in range(num_moves):
    final_state = reverse_move(final_state)

return final_state
```

### 2. PPO (Proximal Policy Optimization)

```python
# 중요도 비율 계산
ratio = torch.exp(new_log_probs - old_log_probs)

# 클리핑된 목적 함수
surr1 = ratio * advantages
surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantages
policy_loss = -torch.min(surr1, surr2).mean()
```

### 3. 다차원 난이도 평가

```python
total_difficulty = (
    w_structural * structural_score +
    w_cognitive * cognitive_score +
    w_solution * solution_score +
    w_success * success_score
)
```

## 개발 및 디버깅

### 로깅 활용

시스템은 상세한 로깅을 제공합니다:

```bash
tail -f results/pipeline_YYYYMMDD_HHMMSS.log
```

### 개별 모듈 테스트

```bash
# 환경 테스트
python -c "from sokoban_env import SokobanEnv; env = SokobanEnv(); print('Environment OK')"

# 레벨 생성기 테스트
python -c "from level_generator import LevelGenerator; gen = LevelGenerator(); level = gen.generate_level(); print('Generator OK')"

# 에이전트 테스트 (모델 없이)
python -c "from rl_agent import PPOAgent; agent = PPOAgent((5,5)); print('Agent OK')"
```

### 성능 모니터링

파이프라인은 다음 성능 지표를 추적합니다:

- 레벨 생성 시간
- 에이전트 해결 시간
- 난이도 평가 시간
- 메모리 사용량

## 알려진 제한사항

1. **계산 복잡도**: 대형 레벨(15×15 이상)에서 성능 저하 가능
2. **학습 안정성**: 초기 에이전트 학습 시 불안정할 수 있음
3. **플랫폼 의존성**: Pygame 렌더링은 GUI 환경 필요

## 기여 방법

1. Fork 생성
2. Feature 브랜치 생성 (`git checkout -b feature/AmazingFeature`)
3. 변경사항 커밋 (`git commit -m 'Add some AmazingFeature'`)
4. 브랜치에 Push (`git push origin feature/AmazingFeature`)
5. Pull Request 생성

## 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 `LICENSE` 파일을 참조하세요.

## 문의 및 지원

- 이슈 리포트: [GitHub Issues](https://github.com/your-username/sokoban-ai-designer/issues)
- 기능 제안: [GitHub Discussions](https://github.com/your-username/sokoban-ai-designer/discussions)

## 참고문헌

1. Schulman, J., et al. "Proximal Policy Optimization Algorithms" (2017)
2. Silver, D., et al. "Mastering the game of Go with deep neural networks and tree search" (2016)
3. Justesen, N., et al. "Procedural Content Generation via Machine Learning" (2018)

---

**소코반 자동 레벨 디자인 AI** - AI의 창의성과 게임 디자인의 만남 