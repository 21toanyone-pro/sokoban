#!/usr/bin/env python3
"""
소코반 레벨 매니저
레벨 저장, 로드, 검색 기능을 제공합니다.
"""

import json
import os
import numpy as np
import datetime
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

class LevelManager:
    """소코반 레벨 저장 및 관리 시스템"""
    
    def __init__(self, base_dir: str = "saved_levels"):
        self.base_dir = Path(base_dir)
        self.levels_dir = self.base_dir / "levels"
        self.index_file = self.base_dir / "level_index.json"
        
        # 디렉토리 생성
        self.base_dir.mkdir(exist_ok=True)
        self.levels_dir.mkdir(exist_ok=True)
        
        # 인덱스 로드
        self.level_index = self._load_index()
    
    def _load_index(self) -> Dict:
        """레벨 인덱스 파일 로드"""
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"인덱스 로드 실패: {e}")
                
        # 기본 인덱스 구조
        return {
            "total_levels": 0,
            "levels": {},
            "difficulty_ranges": {
                "easy": [],      # 0.0 - 0.3
                "medium": [],    # 0.3 - 0.6
                "hard": [],      # 0.6 - 1.0
            },
            "created_at": datetime.datetime.now().isoformat(),
            "last_updated": datetime.datetime.now().isoformat()
        }
    
    def _save_index(self):
        """레벨 인덱스 파일 저장"""
        self.level_index["last_updated"] = datetime.datetime.now().isoformat()
        try:
            with open(self.index_file, 'w', encoding='utf-8') as f:
                json.dump(self.level_index, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"인덱스 저장 실패: {e}")
    
    def _make_json_serializable(self, obj):
        """객체를 JSON 직렬화 가능하도록 변환"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int32, np.int64, np.int8, np.int16)):
            return int(obj)
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, tuple):
            return list(obj)
        elif isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        else:
            return obj
    
    def save_level(self, level: np.ndarray, metadata: Dict[str, Any], level_id: Optional[str] = None) -> Optional[str]:
        """
        레벨을 저장합니다.
        
        Args:
            level: 소코반 레벨 배열
            metadata: 레벨 메타데이터 (난이도, 통계 등)
            level_id: 사용자 지정 ID (None이면 자동 생성)
        
        Returns:
            저장된 레벨의 ID
        """
        # 레벨 ID 생성
        if level_id is None:
            level_id = f"level_{self.level_index['total_levels'] + 1:04d}"
        
        # 메타데이터 보강 (JSON 직렬화 가능하도록 변환)
        level_data = {
            "id": level_id,
            "level": level.tolist(),  # numpy 배열을 리스트로 변환
            "shape": list(level.shape),  # tuple을 list로 변환
            "created_at": datetime.datetime.now().isoformat(),
            "metadata": self._make_json_serializable(metadata)
        }
        
        # 파일 저장
        level_file = self.levels_dir / f"{level_id}.json"
        try:
            with open(level_file, 'w', encoding='utf-8') as f:
                json.dump(level_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"레벨 저장 실패 ({level_id}): {e}")
            return None
        
        # 인덱스 업데이트
        difficulty = metadata.get('total_difficulty', 0.5)
        
        self.level_index["levels"][level_id] = {
            "file": f"{level_id}.json",
            "difficulty": difficulty,
            "size": level.shape,
            "num_boxes": metadata.get('num_boxes', 0),
            "created_at": level_data["created_at"],
            "curriculum_stage": metadata.get('curriculum_stage', None)
        }
        
        # 난이도별 분류
        difficulty_ranges = self.level_index["difficulty_ranges"]
        if difficulty <= 0.3:
            if "easy" not in difficulty_ranges:
                difficulty_ranges["easy"] = []
            difficulty_ranges["easy"].append(level_id)
        elif difficulty <= 0.6:
            if "medium" not in difficulty_ranges:
                difficulty_ranges["medium"] = []
            difficulty_ranges["medium"].append(level_id)
        else:
            if "hard" not in difficulty_ranges:
                difficulty_ranges["hard"] = []
            difficulty_ranges["hard"].append(level_id)
        
        self.level_index["total_levels"] += 1
        self._save_index()
        
        return level_id
    
    def load_level(self, level_id: str) -> Optional[Tuple[np.ndarray, Dict]]:
        """
        레벨을 로드합니다.
        
        Args:
            level_id: 로드할 레벨 ID
            
        Returns:
            (level_array, metadata) 튜플 또는 None
        """
        if level_id not in self.level_index["levels"]:
            print(f"레벨 ID '{level_id}'를 찾을 수 없습니다.")
            return None
        
        level_file = self.levels_dir / self.level_index["levels"][level_id]["file"]
        
        try:
            with open(level_file, 'r', encoding='utf-8') as f:
                level_data = json.load(f)
            
            level_array = np.array(level_data["level"])
            metadata = level_data["metadata"]
            
            return level_array, metadata
            
        except Exception as e:
            print(f"레벨 로드 실패 ({level_id}): {e}")
            return None
    
    def get_levels_by_difficulty(self, difficulty_range: str) -> List[str]:
        """
        난이도별 레벨 ID 목록을 반환합니다.
        
        Args:
            difficulty_range: "easy", "medium", "hard"
            
        Returns:
            레벨 ID 리스트
        """
        return self.level_index["difficulty_ranges"].get(difficulty_range, [])
    
    def get_levels_by_stage(self, stage: int) -> List[str]:
        """
        커리큘럼 단계별 레벨 ID 목록을 반환합니다.
        
        Args:
            stage: 커리큘럼 단계 (1-100)
            
        Returns:
            레벨 ID 리스트
        """
        levels = []
        for level_id, info in self.level_index["levels"].items():
            if info.get("curriculum_stage") == stage:
                levels.append(level_id)
        return levels
    
    def search_levels(self, min_difficulty: float = 0.0, max_difficulty: float = 1.0, 
                     min_boxes: int = 0, max_boxes: int = 10) -> List[str]:
        """
        조건에 맞는 레벨들을 검색합니다.
        
        Args:
            min_difficulty: 최소 난이도
            max_difficulty: 최대 난이도
            min_boxes: 최소 박스 개수
            max_boxes: 최대 박스 개수
            
        Returns:
            조건에 맞는 레벨 ID 리스트
        """
        matching_levels = []
        
        for level_id, info in self.level_index["levels"].items():
            difficulty = info.get("difficulty", 0.5)
            num_boxes = info.get("num_boxes", 0)
            
            if (min_difficulty <= difficulty <= max_difficulty and 
                min_boxes <= num_boxes <= max_boxes):
                matching_levels.append(level_id)
        
        return matching_levels
    
    def get_statistics(self) -> Dict[str, Any]:
        """레벨 통계 정보를 반환합니다."""
        if self.level_index["total_levels"] == 0:
            return {"total_levels": 0}
        
        difficulties = [info["difficulty"] for info in self.level_index["levels"].values()]
        box_counts = [info["num_boxes"] for info in self.level_index["levels"].values()]
        
        return {
            "total_levels": self.level_index["total_levels"],
            "difficulty_stats": {
                "min": min(difficulties),
                "max": max(difficulties),
                "avg": sum(difficulties) / len(difficulties)
            },
            "difficulty_distribution": {
                "easy": len(self.level_index["difficulty_ranges"]["easy"]),
                "medium": len(self.level_index["difficulty_ranges"]["medium"]),
                "hard": len(self.level_index["difficulty_ranges"]["hard"])
            },
            "box_stats": {
                "min": min(box_counts),
                "max": max(box_counts),
                "avg": sum(box_counts) / len(box_counts)
            }
        }
    
    def list_levels(self, limit: int = 20) -> List[Dict]:
        """레벨 목록을 반환합니다."""
        levels = []
        count = 0
        
        for level_id, info in self.level_index["levels"].items():
            if count >= limit:
                break
                
            levels.append({
                "id": level_id,
                "difficulty": info["difficulty"],
                "size": info["size"],
                "num_boxes": info["num_boxes"],
                "created_at": info["created_at"],
                "stage": info.get("curriculum_stage", "N/A")
            })
            count += 1
        
        return levels
    
    def export_levels(self, level_ids: List[str], export_file: str):
        """선택한 레벨들을 하나의 파일로 내보냅니다."""
        export_data = {
            "exported_at": datetime.datetime.now().isoformat(),
            "levels": []
        }
        
        for level_id in level_ids:
            level_data = self.load_level(level_id)
            if level_data:
                level_array, metadata = level_data
                export_data["levels"].append({
                    "id": level_id,
                    "level": level_array.tolist(),
                    "metadata": metadata
                })
        
        try:
            with open(export_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            print(f"{len(export_data['levels'])}개 레벨을 {export_file}로 내보냈습니다.")
        except Exception as e:
            print(f"내보내기 실패: {e}")


def demo_level_manager():
    """레벨 매니저 데모"""
    print("=== 레벨 매니저 데모 ===")
    
    # 매니저 초기화
    manager = LevelManager()
    
    # 더미 레벨 생성 및 저장
    dummy_level = np.array([
        [1, 1, 1, 1, 1],
        [1, 4, 0, 2, 1],
        [1, 0, 0, 3, 1],
        [1, 1, 1, 1, 1]
    ])
    
    metadata = {
        "total_difficulty": 0.4,
        "num_boxes": 1,
        "num_targets": 1,
        "curriculum_stage": 1,
        "generation_time": 0.001
    }
    
    level_id = manager.save_level(dummy_level, metadata)
    print(f"레벨 저장 완료: {level_id}")
    
    # 레벨 로드
    loaded_level, loaded_metadata = manager.load_level(level_id)
    print(f"레벨 로드 완료: {loaded_level.shape}")
    
    # 통계 출력
    stats = manager.get_statistics()
    print(f"📊 총 레벨 수: {stats['total_levels']}")
    
    return manager


if __name__ == "__main__":
    demo_level_manager() 