#!/usr/bin/env python3
"""
소코반 레벨 브라우저
저장된 레벨들을 검색하고 플레이할 수 있는 인터페이스를 제공합니다.
"""

import sys
import numpy as np
from typing import Dict, List, Optional, Any

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

from level_manager import LevelManager
from sokoban_env_simple import SokobanEnvSimple

class LevelBrowser:
    """소코반 레벨 브라우저"""
    
    def __init__(self):
        self.level_manager = LevelManager()
        self.current_level = None
        self.current_env = None
        
    def show_statistics(self):
        """전체 레벨 통계 표시"""
        stats = self.level_manager.get_statistics()
        
        print("=== 레벨 라이브러리 통계 ===")
        print(f"총 레벨 수: {stats['total_levels']}")
        
        if stats['total_levels'] > 0:
            print(f"난이도 범위: {stats['difficulty_stats']['min']:.3f} ~ {stats['difficulty_stats']['max']:.3f}")
            print(f"평균 난이도: {stats['difficulty_stats']['avg']:.3f}")
            
            print(f"\n난이도별 분포:")
            print(f"   쉬움 (0.0-0.3): {stats['difficulty_distribution']['easy']}개")
            print(f"   보통 (0.3-0.6): {stats['difficulty_distribution']['medium']}개")
            print(f"   어려움 (0.6-1.0): {stats['difficulty_distribution']['hard']}개")
            
            print(f"\n박스 개수:")
            print(f"   최소: {stats['box_stats']['min']}개")
            print(f"   최대: {stats['box_stats']['max']}개")
            print(f"   평균: {stats['box_stats']['avg']:.1f}개")
        
        print()
    
    def list_levels(self, limit: int = 20):
        """레벨 목록 표시"""
        levels = self.level_manager.list_levels(limit)
        
        if not levels:
            print("저장된 레벨이 없습니다.")
            return
        
        print(f"=== 레벨 목록 (최대 {limit}개) ===")
        print("ID".ljust(20) + "단계".ljust(8) + "난이도".ljust(10) + "크기".ljust(10) + "박스".ljust(8) + "생성일시")
        print("-" * 80)
        
        for level in levels:
            stage = str(level['stage']).ljust(8)
            difficulty = f"{level['difficulty']:.3f}".ljust(10)
            size = f"{level['size'][0]}x{level['size'][1]}".ljust(10)
            boxes = f"{level['num_boxes']}개".ljust(8)
            created = level['created_at'][:19].replace('T', ' ')
            
            print(f"{level['id'].ljust(20)}{stage}{difficulty}{size}{boxes}{created}")
        
        print()
    
    def search_levels(self):
        """대화형 레벨 검색"""
        print("=== 레벨 검색 ===")
        
        # 검색 조건 입력
        print("검색 조건을 입력하세요 (엔터만 누르면 기본값 사용):")
        
        try:
            min_diff = input("최소 난이도 (0.0-1.0, 기본: 0.0): ").strip()
            min_difficulty = float(min_diff) if min_diff else 0.0
            
            max_diff = input("최대 난이도 (0.0-1.0, 기본: 1.0): ").strip()
            max_difficulty = float(max_diff) if max_diff else 1.0
            
            min_box = input("최소 박스 개수 (기본: 0): ").strip()
            min_boxes = int(min_box) if min_box else 0
            
            max_box = input("최대 박스 개수 (기본: 10): ").strip()
            max_boxes = int(max_box) if max_box else 10
            
        except ValueError:
            print("잘못된 입력입니다. 기본값을 사용합니다.")
            min_difficulty, max_difficulty = 0.0, 1.0
            min_boxes, max_boxes = 0, 10
        
        # 검색 실행
        matching_levels = self.level_manager.search_levels(
            min_difficulty, max_difficulty, min_boxes, max_boxes
        )
        
        print(f"\n검색 결과: {len(matching_levels)}개 레벨 발견")
        
        if matching_levels:
            print("\n검색된 레벨들:")
            for i, level_id in enumerate(matching_levels[:10]):  # 최대 10개만 표시
                level_info = self.level_manager.level_index["levels"][level_id]
                print(f"{i+1:2d}. {level_id} (난이도: {level_info['difficulty']:.3f}, 박스: {level_info['num_boxes']}개)")
            
            if len(matching_levels) > 10:
                print(f"    ... 외 {len(matching_levels) - 10}개 더")
        
        print()
        return matching_levels
    
    def load_and_display_level(self, level_id: str) -> bool:
        """레벨을 로드하고 표시"""
        level_data = self.level_manager.load_level(level_id)
        
        if level_data is None:
            print(f"❌ 레벨 '{level_id}'를 찾을 수 없습니다.")
            return False
        
        level_array, metadata = level_data
        
        # 환경 생성
        self.current_level = level_array
        self.current_env = SokobanEnvSimple(level=level_array)
        
        # 레벨 정보 표시
        print(f"🎮 === 레벨: {level_id} ===")
        print(f"📏 크기: {level_array.shape}")
        print(f"📊 난이도: {metadata.get('measured_difficulty', 'N/A'):.3f}")
        print(f"📦 박스 개수: {metadata.get('num_boxes', 'N/A')}")
        print(f"🎯 단계: {metadata.get('curriculum_stage', 'N/A')}")
        
        if 'target_difficulty' in metadata:
            print(f"🎚️  목표 난이도: {metadata['target_difficulty']:.3f}")
        
        print("\n레벨:")
        print(self.current_env.render())
        print()
        
        return True
    
    def play_level_interactive(self):
        """대화형 레벨 플레이"""
        if self.current_env is None:
            print("❌ 먼저 레벨을 로드해주세요.")
            return
        
        print("🎮 === 레벨 플레이 ===")
        print("조작법: w(위), s(아래), a(왼쪽), d(오른쪽), q(종료)")
        print("기호: @ = 플레이어, $ = 박스, . = 목표, * = 목표 위 박스, # = 벽")
        print()
        
        # 게임 리셋
        obs, info = self.current_env.reset()
        print("초기 상태:")
        print(self.current_env.render())
        print(f"스텝: {info['step_count']}, 목표 박스: {info['boxes_on_target']}/{info['total_boxes']}")
        print()
        
        # 게임 루프
        while True:
            try:
                command = input("다음 행동을 입력하세요: ").strip().lower()
                
                if command == 'q':
                    print("게임을 종료합니다.")
                    break
                
                # 행동 매핑
                action_map = {'w': 0, 's': 1, 'a': 2, 'd': 3}
                
                if command not in action_map:
                    print("❌ 잘못된 입력입니다. w/s/a/d/q만 사용하세요.")
                    continue
                
                action = action_map[command]
                
                # 행동 실행
                obs, reward, terminated, truncated, info = self.current_env.step(action)
                
                # 결과 표시
                print(f"\n행동: {command} (보상: {reward:.1f})")
                print(self.current_env.render())
                print(f"스텝: {info['step_count']}, 목표 박스: {info['boxes_on_target']}/{info['total_boxes']}")
                
                # 게임 종료 확인
                if terminated:
                    if info['is_solved']:
                        print("🎉 축하합니다! 퍼즐을 해결했습니다!")
                    else:
                        print("💀 게임 종료")
                    break
                elif truncated:
                    print("⏰ 시간 초과로 게임이 종료되었습니다.")
                    break
                
                print()
                
            except KeyboardInterrupt:
                print("\n게임을 중단합니다.")
                break
            except Exception as e:
                print(f"❌ 오류 발생: {e}")
                continue
    
    def browse_by_stage(self):
        """단계별 레벨 탐색"""
        print("📚 === 단계별 레벨 탐색 ===")
        
        try:
            stage = int(input("탐색할 단계를 입력하세요 (1-100): "))
            
            if not (1 <= stage <= 100):
                print("❌ 단계는 1-100 사이여야 합니다.")
                return
            
        except ValueError:
            print("❌ 숫자를 입력해주세요.")
            return
        
        # 해당 단계의 레벨들 찾기
        stage_levels = self.level_manager.get_levels_by_stage(stage)
        
        if not stage_levels:
            print(f"❌ 단계 {stage}의 레벨이 없습니다.")
            return
        
        print(f"\n🎯 단계 {stage}의 레벨들 ({len(stage_levels)}개):")
        
        for i, level_id in enumerate(stage_levels):
            level_info = self.level_manager.level_index["levels"][level_id]
            print(f"{i+1}. {level_id} (난이도: {level_info['difficulty']:.3f})")
        
        print()
    
    def browse_by_difficulty(self):
        """난이도별 레벨 탐색"""
        print("🎚️  === 난이도별 레벨 탐색 ===")
        print("1. 쉬움 (0.0-0.3)")
        print("2. 보통 (0.3-0.6)")
        print("3. 어려움 (0.6-1.0)")
        
        try:
            choice = int(input("선택하세요 (1-3): "))
            
            difficulty_map = {1: "easy", 2: "medium", 3: "hard"}
            
            if choice not in difficulty_map:
                print("❌ 1-3 중에서 선택해주세요.")
                return
            
            difficulty_range = difficulty_map[choice]
            
        except ValueError:
            print("❌ 숫자를 입력해주세요.")
            return
        
        # 해당 난이도의 레벨들 찾기
        difficulty_levels = self.level_manager.get_levels_by_difficulty(difficulty_range)
        
        if not difficulty_levels:
            print(f"❌ {difficulty_range} 난이도의 레벨이 없습니다.")
            return
        
        range_names = {"easy": "쉬움", "medium": "보통", "hard": "어려움"}
        print(f"\n🎯 {range_names[difficulty_range]} 난이도의 레벨들 ({len(difficulty_levels)}개):")
        
        for i, level_id in enumerate(difficulty_levels[:20]):  # 최대 20개
            level_info = self.level_manager.level_index["levels"][level_id]
            print(f"{i+1:2d}. {level_id} (난이도: {level_info['difficulty']:.3f})")
        
        if len(difficulty_levels) > 20:
            print(f"    ... 외 {len(difficulty_levels) - 20}개 더")
        
        print()
    
    def main_menu(self):
        """메인 메뉴 실행"""
        print("🎮 === 소코반 레벨 브라우저 ===")
        print("저장된 소코반 레벨들을 탐색하고 플레이할 수 있습니다.\n")
        
        # 초기 통계 표시
        self.show_statistics()
        
        while True:
            print("📋 === 메인 메뉴 ===")
            print("1. 레벨 목록 보기")
            print("2. 레벨 검색")
            print("3. 단계별 탐색")
            print("4. 난이도별 탐색")
            print("5. 레벨 로드 및 플레이")
            print("6. 통계 보기")
            print("0. 종료")
            
            try:
                choice = input("\n선택하세요: ").strip()
                
                if choice == '0':
                    print("👋 소코반 레벨 브라우저를 종료합니다.")
                    break
                elif choice == '1':
                    self.list_levels()
                elif choice == '2':
                    self.search_levels()
                elif choice == '3':
                    self.browse_by_stage()
                elif choice == '4':
                    self.browse_by_difficulty()
                elif choice == '5':
                    level_id = input("로드할 레벨 ID를 입력하세요: ").strip()
                    if level_id:
                        if self.load_and_display_level(level_id):
                            play_choice = input("이 레벨을 플레이하시겠습니까? (y/n): ").strip().lower()
                            if play_choice == 'y':
                                self.play_level_interactive()
                elif choice == '6':
                    self.show_statistics()
                else:
                    print("❌ 잘못된 선택입니다.")
                
                print()
                
            except KeyboardInterrupt:
                print("\n👋 소코반 레벨 브라우저를 종료합니다.")
                break
            except Exception as e:
                print(f"❌ 오류 발생: {e}")
                continue


def main():
    """메인 함수"""
    browser = LevelBrowser()
    browser.main_menu()


if __name__ == "__main__":
    main() 