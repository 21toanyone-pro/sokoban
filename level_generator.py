import numpy as np
import random
from typing import List, Tuple, Set, Optional, Dict, cast
from copy import deepcopy
from sokoban_env import SokobanEnv

class LevelGenerator:
    """
    소코반 레벨 자동 생성기
    역방향 생성(Reverse Generation) 알고리즘을 사용하여 
    풀이 가능성을 100% 보장하는 레벨을 생성합니다.
    """
    
    def __init__(self, min_size: Tuple[int, int] = (5, 5), max_size: Tuple[int, int] = (12, 12)):
        self.min_size = min_size
        self.max_size = max_size
        
        # 역방향 생성을 위한 행동 (플레이어 이동의 역순)
        self.reverse_actions = [
            (-1, 0),  # 위
            (1, 0),   # 아래  
            (0, -1),  # 왼쪽
            (0, 1)    # 오른쪽
        ]
        
    def generate_level(self, num_boxes: Optional[int] = None, complexity: float = 0.5, 
                      target_difficulty: Optional[float] = None) -> np.ndarray:
        """
        새로운 소코반 레벨 생성
        
        Args:
            num_boxes: 박스 개수 (None이면 자동 결정)
            complexity: 레벨 복잡도 (0.0-1.0)
            target_difficulty: 목표 난이도 (None이면 무시)
        
        Returns:
            생성된 레벨의 numpy 배열
        """
        attempts = 0
        max_attempts = 10
        
        while attempts < max_attempts:
            try:
                level = self._generate_level_internal(num_boxes, complexity)
                if self._validate_level(level):
                    return level
            except Exception as e:
                print(f"레벨 생성 시도 {attempts + 1} 실패: {e}")
            
            attempts += 1
        
        # 모든 시도가 실패하면 간단한 기본 레벨 반환
        print("레벨 생성 실패, 기본 레벨을 반환합니다.")
        return self._create_fallback_level()
    
    def _generate_level_internal(self, num_boxes: Optional[int], complexity: float) -> np.ndarray:
        """내부 레벨 생성 로직"""
        # 레벨 크기 결정
        height = random.randint(self.min_size[0], self.max_size[0])
        width = random.randint(self.min_size[1], self.max_size[1])
        
        # 박스 개수 자동 결정
        if num_boxes is None:
            max_boxes = min(6, (height * width) // 8)  # 공간의 1/8 정도
            num_boxes = random.randint(1, max(1, max_boxes))
        
        # 1단계: 기본 방 구조 생성
        level = self._create_room_structure(height, width, complexity)
        
        # 2단계: 역방향 생성으로 풀이 가능한 퍼즐 생성
        level = self._reverse_generate_puzzle(level, num_boxes)
        
        return level
    
    def _create_room_structure(self, height: int, width: int, complexity: float) -> np.ndarray:
        """기본적인 방 구조 생성"""
        level = np.full((height, width), SokobanEnv.WALL, dtype=np.int32)
        
        # 중앙 영역을 빈 공간으로 만들기
        inner_height = max(3, height - 2)
        inner_width = max(3, width - 2)
        
        start_y = (height - inner_height) // 2
        start_x = (width - inner_width) // 2
        
        # 기본 방 만들기
        for y in range(start_y, start_y + inner_height):
            for x in range(start_x, start_x + inner_width):
                level[y, x] = SokobanEnv.EMPTY
        
        # 복잡도에 따라 내부 벽 추가
        if complexity > 0.3:
            self._add_internal_walls(level, complexity)
        
        return level
    
    def _add_internal_walls(self, level: np.ndarray, complexity: float):
        """복잡도에 따라 내부 벽 추가"""
        height, width = level.shape
        
        # 추가할 벽의 개수 결정
        max_walls = int((height * width * complexity) // 10)
        
        for _ in range(max_walls):
            # 랜덤 위치에 벽 추가 (가장자리는 제외)
            y = random.randint(1, height - 2)
            x = random.randint(1, width - 2)
            
            if level[y, x] == SokobanEnv.EMPTY:
                # 벽을 추가해도 연결성이 유지되는지 확인
                level[y, x] = SokobanEnv.WALL
                if not self._check_connectivity(level):
                    level[y, x] = SokobanEnv.EMPTY  # 연결성이 깨지면 되돌림
    
    def _check_connectivity(self, level: np.ndarray) -> bool:
        """레벨의 모든 빈 공간이 연결되어 있는지 확인"""
        height, width = level.shape
        empty_cells = [(y, x) for y in range(height) for x in range(width) 
                       if level[y, x] == SokobanEnv.EMPTY]
        
        if not empty_cells:
            return False
        
        # BFS로 연결성 확인
        visited = set()
        queue = [empty_cells[0]]
        visited.add(empty_cells[0])
        
        while queue:
            y, x = queue.pop(0)
            for dy, dx in self.reverse_actions:
                ny, nx = y + dy, x + dx
                if (0 <= ny < height and 0 <= nx < width and 
                    (ny, nx) not in visited and level[ny, nx] == SokobanEnv.EMPTY):
                    visited.add((ny, nx))
                    queue.append((ny, nx))
        
        return len(visited) == len(empty_cells)
    
    def _reverse_generate_puzzle(self, level: np.ndarray, num_boxes: int) -> np.ndarray:
        """역방향 생성 알고리즘으로 퍼즐 생성"""
        height, width = level.shape
        
        # 빈 공간 찾기
        empty_positions = [(y, x) for y in range(height) for x in range(width) 
                          if level[y, x] == SokobanEnv.EMPTY]
        
        if len(empty_positions) < num_boxes + 1:  # 플레이어 + 박스들
            raise ValueError("공간이 부족합니다.")
        
        # 목표 위치들을 무작위로 선택
        target_positions = random.sample(empty_positions, num_boxes)
        
        # 플레이어 초기 위치 선택 (목표와 겹치지 않는 곳)
        remaining_positions = [pos for pos in empty_positions if pos not in target_positions]
        player_start = cast(Tuple[int, int], random.choice(remaining_positions))
        
        # 역방향 생성 시작: 해결된 상태에서 시작
        state = {
            'player_pos': player_start,
            'box_positions': set(target_positions),
            'target_positions': set(target_positions)
        }
        
        # 역방향으로 박스들을 이동시켜 초기 상태 만들기
        moves = random.randint(10, 30)  # 이동 횟수
        for _ in range(moves):
            state = self._reverse_move(level, state)
        
        # 최종 레벨 구성
        final_level = level.copy()
        
        # 목표 지점 배치
        for pos in target_positions:
            final_level[pos] = SokobanEnv.TARGET
        
        # 박스 위치 배치
        for pos in state['box_positions']:
            if pos in target_positions:
                final_level[pos] = SokobanEnv.BOX_ON_TARGET
            else:
                final_level[pos] = SokobanEnv.BOX
        
        # 플레이어 위치 배치
        player_pos = state['player_pos']
        if player_pos in target_positions:
            final_level[player_pos] = SokobanEnv.PLAYER_ON_TARGET
        else:
            final_level[player_pos] = SokobanEnv.PLAYER
        
        return final_level
    
    def _reverse_move(self, level: np.ndarray, state: Dict) -> Dict:
        """역방향 이동 수행"""
        height, width = level.shape
        player_pos = state['player_pos']
        box_positions = state['box_positions'].copy()
        
        # 가능한 역방향 이동 찾기
        possible_moves = []
        
        for dy, dx in self.reverse_actions:
            # 플레이어가 이동할 수 있는 위치
            new_player_pos = (player_pos[0] + dy, player_pos[1] + dx)
            
            if not self._is_valid_move_position(level, new_player_pos):
                continue
            
            # 박스를 당기는 경우
            pull_from_pos = (player_pos[0] - dy, player_pos[1] - dx)
            
            if (self._is_valid_move_position(level, pull_from_pos) and 
                pull_from_pos in box_positions and
                new_player_pos not in box_positions):
                
                possible_moves.append(('pull', new_player_pos, pull_from_pos, player_pos))
            
            # 일반 이동 (박스 없이)
            elif new_player_pos not in box_positions:
                possible_moves.append(('move', new_player_pos, (-1, -1), (-1, -1)))
        
        if not possible_moves:
            return state  # 이동 불가능하면 현재 상태 유지
        
        # 랜덤하게 이동 선택
        move_type, new_player_pos, box_from, box_to = random.choice(possible_moves)
        
        new_state = {
            'player_pos': new_player_pos,
            'box_positions': box_positions,
            'target_positions': state['target_positions']
        }
        
        if move_type == 'pull':
            box_positions.remove(box_from)
            box_positions.add(box_to)
        
        return new_state
    
    def _is_valid_move_position(self, level: np.ndarray, pos: Tuple[int, int]) -> bool:
        """이동 가능한 위치인지 확인"""
        y, x = pos
        height, width = level.shape
        
        if y < 0 or y >= height or x < 0 or x >= width:
            return False
        
        return level[y, x] != SokobanEnv.WALL
    
    def _validate_level(self, level: np.ndarray) -> bool:
        """생성된 레벨이 유효한지 검증"""
        # 기본 요소들이 모두 있는지 확인
        has_player = np.any((level == SokobanEnv.PLAYER) | (level == SokobanEnv.PLAYER_ON_TARGET))
        has_box = np.any((level == SokobanEnv.BOX) | (level == SokobanEnv.BOX_ON_TARGET))
        has_target = np.any((level == SokobanEnv.TARGET) | 
                           (level == SokobanEnv.BOX_ON_TARGET) | 
                           (level == SokobanEnv.PLAYER_ON_TARGET))
        
        if not (has_player and has_box and has_target):
            return False
        
        # 박스와 목표 개수가 같은지 확인
        num_boxes = np.sum((level == SokobanEnv.BOX) | (level == SokobanEnv.BOX_ON_TARGET))
        num_targets = np.sum((level == SokobanEnv.TARGET) | 
                            (level == SokobanEnv.BOX_ON_TARGET) | 
                            (level == SokobanEnv.PLAYER_ON_TARGET))
        
        return num_boxes == num_targets
    
    def _create_fallback_level(self) -> np.ndarray:
        """기본 대체 레벨 생성"""
        return np.array([
            [1, 1, 1, 1, 1, 1],
            [1, 4, 0, 2, 3, 1],
            [1, 0, 0, 0, 0, 1],
            [1, 3, 2, 0, 0, 1],
            [1, 1, 1, 1, 1, 1]
        ])
    
    def generate_batch(self, batch_size: int = 5, **kwargs) -> List[np.ndarray]:
        """여러 레벨을 한 번에 생성"""
        levels = []
        for _ in range(batch_size):
            level = self.generate_level(**kwargs)
            levels.append(level)
        return levels
    
    def get_level_stats(self, level: np.ndarray) -> Dict:
        """레벨의 통계 정보 반환"""
        height, width = level.shape
        
        num_walls = np.sum(level == SokobanEnv.WALL)
        num_boxes = np.sum((level == SokobanEnv.BOX) | (level == SokobanEnv.BOX_ON_TARGET))
        num_targets = np.sum((level == SokobanEnv.TARGET) | 
                            (level == SokobanEnv.BOX_ON_TARGET) | 
                            (level == SokobanEnv.PLAYER_ON_TARGET))
        num_empty = np.sum(level == SokobanEnv.EMPTY)
        
        return {
            'size': (height, width),
            'total_cells': height * width,
            'num_walls': num_walls,
            'num_boxes': num_boxes,
            'num_targets': num_targets,
            'num_empty': num_empty,
            'wall_ratio': num_walls / (height * width),
            'box_density': num_boxes / max(1, num_empty + num_boxes)
        }


class AdaptiveLevelGenerator(LevelGenerator):
    """
    적응형 레벨 생성기
    난이도 피드백을 받아서 목표 난이도에 맞는 레벨을 생성합니다.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.difficulty_history = []
        self.generation_params = {
            'complexity': 0.5,
            'num_boxes': 3,
            'moves': 20
        }
    
    def update_difficulty_feedback(self, generated_level: np.ndarray, 
                                 measured_difficulty: float):
        """난이도 피드백 업데이트"""
        level_stats = self.get_level_stats(generated_level)
        
        self.difficulty_history.append({
            'level_stats': level_stats,
            'measured_difficulty': measured_difficulty,
            'generation_params': self.generation_params.copy()
        })
        
        # 최근 10개 기록만 유지
        if len(self.difficulty_history) > 10:
            self.difficulty_history = self.difficulty_history[-10:]
    
    def generate_level_with_target_difficulty(self, target_difficulty: float) -> np.ndarray:
        """목표 난이도에 맞는 레벨 생성"""
        # 과거 데이터를 바탕으로 생성 파라미터 조정
        self._adjust_parameters(target_difficulty)
        
        # 조정된 파라미터로 레벨 생성
        return self.generate_level(
            num_boxes=self.generation_params['num_boxes'],
            complexity=self.generation_params['complexity']
        )
    
    def _adjust_parameters(self, target_difficulty: float):
        """목표 난이도에 맞게 생성 파라미터 조정"""
        if not self.difficulty_history:
            # 초기 상태: 기본값 사용
            return
        
        # 간단한 선형 조정 로직
        if target_difficulty > 0.7:  # 어려운 레벨
            self.generation_params['complexity'] = min(0.8, self.generation_params['complexity'] + 0.1)
            self.generation_params['num_boxes'] = min(6, self.generation_params['num_boxes'] + 1)
        elif target_difficulty < 0.3:  # 쉬운 레벨
            self.generation_params['complexity'] = max(0.2, self.generation_params['complexity'] - 0.1)
            self.generation_params['num_boxes'] = max(1, self.generation_params['num_boxes'] - 1)
        
        # 복잡도 범위 제한
        self.generation_params['complexity'] = np.clip(self.generation_params['complexity'], 0.1, 0.9) 