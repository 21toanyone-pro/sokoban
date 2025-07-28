import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Tuple, Dict, Any

class SokobanEnvSimple(gym.Env):
    """
    소코반 게임 환경 - OpenAI Gym 인터페이스 (Pygame 없는 간단 버전)
    
    게임 요소:
    - 0: 빈 공간
    - 1: 벽
    - 2: 박스
    - 3: 목표 지점
    - 4: 플레이어
    - 5: 목표 지점 위의 박스
    - 6: 목표 지점 위의 플레이어
    """
    
    # 게임 요소 상수
    EMPTY = 0
    WALL = 1
    BOX = 2
    TARGET = 3
    PLAYER = 4
    BOX_ON_TARGET = 5
    PLAYER_ON_TARGET = 6
    
    # 행동 정의
    ACTIONS = {
        0: (-1, 0),  # 위
        1: (1, 0),   # 아래
        2: (0, -1),  # 왼쪽
        3: (0, 1)    # 오른쪽
    }
    
    def __init__(self, level: Optional[np.ndarray] = None, max_steps: int = 200, render_mode: Optional[str] = None):
        super().__init__()
        
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.step_count = 0
        
        # 기본 레벨 설정 (level이 없으면 간단한 테스트 레벨 사용)
        if level is None:
            self.original_level = self._create_default_level()
        else:
            self.original_level = level.copy()
        
        self.level = None
        self.player_pos = None
        self.box_positions = set()
        self.target_positions = set()
        
        # 행동 공간: 상하좌우
        self.action_space = spaces.Discrete(4)
        
        # 관찰 공간: 게임 보드의 각 셀 상태
        self.observation_space = spaces.Box(
            low=0, high=6, 
            shape=self.original_level.shape, 
            dtype=np.int32
        )
        
        self.reset()
    
    def _create_default_level(self) -> np.ndarray:
        """기본 테스트 레벨 생성"""
        return np.array([
            [1, 1, 1, 1, 1],
            [1, 4, 0, 2, 1],
            [1, 0, 0, 3, 1],
            [1, 1, 1, 1, 1]
        ])
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        
        self.level = self.original_level.copy()
        self.step_count = 0
        
        # 플레이어, 박스, 목표 위치 찾기
        self._parse_level()
        
        return self._get_observation(), self._get_info()
    
    def _parse_level(self):
        """레벨에서 플레이어, 박스, 목표 위치 추출"""
        self.player_pos = None
        self.box_positions = set()
        self.target_positions = set()
        
        for i in range(self.level.shape[0]):
            for j in range(self.level.shape[1]):
                cell = self.level[i, j]
                if cell == self.PLAYER:
                    self.player_pos = (i, j)
                elif cell == self.PLAYER_ON_TARGET:
                    self.player_pos = (i, j)
                    self.target_positions.add((i, j))
                elif cell == self.BOX:
                    self.box_positions.add((i, j))
                elif cell == self.BOX_ON_TARGET:
                    self.box_positions.add((i, j))
                    self.target_positions.add((i, j))
                elif cell == self.TARGET:
                    self.target_positions.add((i, j))
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        if self.player_pos is None:
            raise ValueError("게임이 리셋되지 않았습니다.")
        
        self.step_count += 1
        reward = 0
        terminated = False
        truncated = False
        
        # 행동 수행
        move_reward, push_reward = self._move_player(action)
        reward = move_reward + push_reward
        
        # 게임 종료 조건 확인
        if self._is_solved():
            terminated = True
            reward += 100  # 게임 완료 보너스
        elif self.step_count >= self.max_steps:
            truncated = True
            reward -= 10  # 시간 초과 페널티
        
        # 레벨 상태 업데이트
        self._update_level()
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()
    
    def _move_player(self, action: int) -> Tuple[float, float]:
        """플레이어 이동 처리"""
        if action not in self.ACTIONS:
            return -1, 0  # 잘못된 행동 페널티
        
        dy, dx = self.ACTIONS[action]
        new_pos = (self.player_pos[0] + dy, self.player_pos[1] + dx)
        
        # 경계 확인
        if not self._is_valid_position(new_pos):
            return -1, 0  # 벽으로 이동 시도 페널티
        
        move_reward = -0.1  # 기본 이동 비용
        push_reward = 0
        
        # 목표 위치의 셀 확인
        if new_pos in self.box_positions:
            # 박스가 있는 경우 - 박스 밀기 시도
            box_new_pos = (new_pos[0] + dy, new_pos[1] + dx)
            
            if (self._is_valid_position(box_new_pos) and 
                box_new_pos not in self.box_positions):
                # 박스 이동 가능
                self.box_positions.remove(new_pos)
                self.box_positions.add(box_new_pos)
                self.player_pos = new_pos
                
                # 박스가 목표에 도달했는지 확인
                if box_new_pos in self.target_positions:
                    push_reward = 10  # 목표에 박스 놓기 보너스
                else:
                    push_reward = 1   # 박스 이동 보너스
            else:
                # 박스 이동 불가능
                return -1, 0
        else:
            # 빈 공간으로 이동
            self.player_pos = new_pos
            
            # 목표 지점에 도달하면 약간의 보너스
            if new_pos in self.target_positions:
                move_reward = 0.1
        
        return move_reward, push_reward
    
    def _is_valid_position(self, pos: Tuple[int, int]) -> bool:
        """유효한 위치인지 확인"""
        y, x = pos
        if (y < 0 or y >= self.level.shape[0] or 
            x < 0 or x >= self.level.shape[1]):
            return False
        return self.original_level[y, x] != self.WALL
    
    def _is_solved(self) -> bool:
        """게임이 완료되었는지 확인"""
        return self.box_positions == self.target_positions
    
    def _update_level(self):
        """현재 상태에 따라 레벨 업데이트"""
        # 원본 레벨에서 시작 (벽과 목표만)
        self.level = np.where(
            self.original_level == self.WALL, 
            self.WALL, 
            self.EMPTY
        )
        
        # 목표 지점 표시
        for pos in self.target_positions:
            if pos not in self.box_positions and pos != self.player_pos:
                self.level[pos] = self.TARGET
        
        # 박스 위치 표시
        for pos in self.box_positions:
            if pos in self.target_positions:
                self.level[pos] = self.BOX_ON_TARGET
            else:
                self.level[pos] = self.BOX
        
        # 플레이어 위치 표시
        if self.player_pos in self.target_positions:
            self.level[self.player_pos] = self.PLAYER_ON_TARGET
        else:
            self.level[self.player_pos] = self.PLAYER
    
    def _get_observation(self) -> np.ndarray:
        """현재 게임 상태 반환"""
        return self.level.astype(np.int32)
    
    def _get_info(self) -> Dict:
        """추가 정보 반환"""
        return {
            'step_count': self.step_count,
            'is_solved': self._is_solved(),
            'boxes_on_target': len(self.box_positions & self.target_positions),
            'total_boxes': len(self.box_positions),
            'player_position': self.player_pos
        }
    
    def render(self) -> str:
        """게임 상태 시각화 (텍스트만)"""
        symbols = {
            self.EMPTY: ' ',
            self.WALL: '#',
            self.BOX: '$',
            self.TARGET: '.',
            self.PLAYER: '@',
            self.BOX_ON_TARGET: '*',
            self.PLAYER_ON_TARGET: '+'
        }
        
        result = []
        for row in self.level:
            result.append(''.join(symbols[cell] for cell in row))
        
        return '\n'.join(result)
    
    def close(self):
        """환경 정리"""
        pass
    
    def get_level_data(self) -> Dict[str, Any]:
        """레벨 데이터 반환 (난이도 평가용)"""
        return {
            'level': self.original_level.copy(),
            'size': self.original_level.shape,
            'num_boxes': len(self.box_positions),
            'num_targets': len(self.target_positions),
            'level_complexity': self._calculate_complexity()
        }
    
    def _calculate_complexity(self) -> float:
        """레벨의 구조적 복잡도 계산"""
        # 간단한 복잡도 지표
        total_cells = self.original_level.size
        wall_ratio = np.sum(self.original_level == self.WALL) / total_cells
        box_target_ratio = len(self.box_positions) / max(1, len(self.target_positions))
        
        return wall_ratio * 0.5 + box_target_ratio * 0.5 