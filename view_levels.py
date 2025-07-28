#!/usr/bin/env python3
"""
저장된 소코반 레벨들을 간단하게 확인하는 뷰어
"""

import json
import numpy as np
from pathlib import Path

def render_level(level_array):
    """레벨을 텍스트로 렌더링"""
    symbols = {
        0: ' ',  # 빈 공간
        1: '#',  # 벽
        2: '$',  # 박스
        3: '.',  # 목표
        4: '@',  # 플레이어
        5: '*',  # 목표 위 박스
        6: '+'   # 목표 위 플레이어
    }
    
    result = []
    for row in level_array:
        line = ''.join(symbols.get(cell, '?') for cell in row)
        result.append(line)
    
    return '\n'.join(result)

def load_and_display_level(level_file):
    """레벨 파일을 로드하고 표시"""
    try:
        with open(level_file, 'r', encoding='utf-8') as f:
            level_data = json.load(f)
        
        level_id = level_data['id']
        level_array = np.array(level_data['level'])
        metadata = level_data['metadata']
        
        print(f"레벨: {level_id}")
        print(f"크기: {level_array.shape}")
        print(f"난이도: {metadata.get('measured_difficulty', 'N/A'):.3f}")
        print(f"단계: {metadata.get('curriculum_stage', 'N/A')}")
        print(f"박스: {metadata.get('level_stats', {}).get('num_boxes', 'N/A')}개")
        print("\n레벨:")
        print(render_level(level_array))
        print()
        
        return level_data
        
    except Exception as e:
        print(f"레벨 로드 실패 ({level_file}): {e}")
        return None

def show_curriculum_overview():
    """커리큘럼 개요 표시"""
    levels_dir = Path("saved_levels/levels")
    
    if not levels_dir.exists():
        print("saved_levels/levels 폴더가 없습니다.")
        return
    
    # 모든 레벨 파일 찾기
    level_files = sorted(levels_dir.glob("*.json"))
    
    if not level_files:
        print("저장된 레벨이 없습니다.")
        return
    
    print(f"=== 소코반 커리큘럼 개요 ===")
    print(f"총 레벨 수: {len(level_files)}")
    print()
    
    # 단계별 정리
    stages = {}
    for level_file in level_files:
        try:
            with open(level_file, 'r', encoding='utf-8') as f:
                level_data = json.load(f)
            
            stage = level_data['metadata'].get('curriculum_stage', 0)
            if stage not in stages:
                stages[stage] = []
            
            stages[stage].append({
                'id': level_data['id'],
                'difficulty': level_data['metadata'].get('measured_difficulty', 0),
                'file': level_file
            })
            
        except Exception as e:
            print(f"파일 읽기 실패: {level_file} - {e}")
            continue
    
    # 단계별 요약 표시
    for stage in sorted(stages.keys()):
        stage_levels = stages[stage]
        difficulties = [l['difficulty'] for l in stage_levels]
        avg_difficulty = sum(difficulties) / len(difficulties)
        
        print(f"단계 {stage}: {len(stage_levels)}개 레벨, 평균 난이도 {avg_difficulty:.3f}")
        for level in stage_levels:
            print(f"   - {level['id']} (난이도: {level['difficulty']:.3f})")
        print()
    
    return stages

def show_specific_levels(stage=None, count=5):
    """특정 단계의 레벨들을 표시"""
    levels_dir = Path("saved_levels/levels")
    
    if stage:
        pattern = f"stage_{stage:03d}_*.json"
        level_files = sorted(levels_dir.glob(pattern))
        print(f"=== 단계 {stage} 레벨들 ===")
    else:
        level_files = sorted(levels_dir.glob("*.json"))[:count]
        print(f"=== 처음 {count}개 레벨들 ===")
    
    if not level_files:
        print("해당하는 레벨이 없습니다.")
        return
    
    for level_file in level_files:
        load_and_display_level(level_file)
        print("-" * 50)

def main():
    """메인 함수"""
    print("=== 소코반 레벨 뷰어 ===\n")
    
    # 1. 커리큘럼 개요
    stages = show_curriculum_overview()
    
    if not stages:
        return
    
    # 2. 몇 개 레벨 예시 표시
    print("=== 레벨 예시 (처음 3개) ===")
    show_specific_levels(count=3)
    
    # 3. 사용법 안내
    print("=== 사용법 ===")
    print("특정 단계 보기: python view_levels.py --stage 1")
    print("레벨 브라우저: python level_browser.py")
    print("더 많은 레벨 생성: python curriculum_generator.py --start 1 --end 100")
    print("레벨 플레이: level_browser.py에서 레벨 로드 후 플레이")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--stage" and len(sys.argv) > 2:
        try:
            stage = int(sys.argv[2])
            show_specific_levels(stage=stage)
        except ValueError:
            print("잘못된 단계 번호입니다.")
    else:
        main() 