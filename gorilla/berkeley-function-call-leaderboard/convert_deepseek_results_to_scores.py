#!/usr/bin/env python3
"""
DeepSeek V3.1 모델들의 result 파일을 score로 변환하는 스크립트
기존 BFCL 평가 시스템을 정확히 사용하여 100% 호환성 보장

Usage:
    python convert_deepseek_results_to_scores.py
"""

import os
import sys
from pathlib import Path

# BFCL 모듈 경로 추가
bfcl_path = Path(__file__).parent / "bfcl_eval"
sys.path.insert(0, str(bfcl_path))

# 필요한 패키지 설치 확인
def check_and_install_requirements():
    """필요한 패키지들을 확인하고 설치"""
    required_packages = [
        "typer",
        "python-dotenv", 
        "tabulate",
        "tqdm",
        "numpy",
        "pandas",
        "overrides",
        "pydantic",
        "anthropic",
        "openai"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"필요한 패키지 설치 중: {', '.join(missing_packages)}")
        import subprocess
        for package in missing_packages:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"[OK] {package} 설치 완료")
            except subprocess.CalledProcessError:
                print(f"[FAIL] {package} 설치 실패")
                return False
    return True

def main():
    """
    메인 함수 - BFCL의 evaluate 명령을 사용해서 정확한 score 생성
    """
    # Windows 콘솔 인코딩 설정
    if sys.platform == "win32":
        import locale
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except:
            pass
    
    print("DeepSeek V3.1 모델 result -> score 변환 시작")
    print("=" * 60)
    
    # 패키지 설치 확인
    if not check_and_install_requirements():
        print("[WARN] 필요한 패키지 설치에 실패했습니다.")
        return
    
    # BFCL 모듈 import
    try:
        from bfcl_eval.eval_checker.eval_runner import main as evaluation_main
        from bfcl_eval.constants.eval_config import RESULT_PATH, SCORE_PATH
        print("[OK] BFCL 평가 모듈 로드 성공")
    except ImportError as e:
        print(f"[FAIL] BFCL 모듈 로드 실패: {e}")
        print("bfcl_eval 폴더가 올바른 위치에 있는지 확인하세요.")
        return
    
    # DeepSeek V3.1 모델들
    target_models = [
        "deepseek-ai_DeepSeek-V3.1-thinking-off",
        "deepseek-ai_DeepSeek-V3.1-thinking-on"
    ]
    
    base_path = Path(__file__).parent
    result_path = base_path / "result"
    score_path = base_path / "score"
    
    # 각 모델별로 처리
    for model_name in target_models:
        print(f"\n[MODEL] {model_name}")
        print("-" * 50)
        
        model_result_path = result_path / model_name
        model_score_path = score_path / model_name
        
        if not model_result_path.exists():
            print(f"  [WARN] Result 폴더가 존재하지 않습니다: {model_result_path}")
            continue
            
        # result 파일들 확인
        result_files = list(model_result_path.glob("BFCL_v3_*_result.json"))
        print(f"  [INFO] {len(result_files)}개의 result 파일 발견")
        
        if not result_files:
            print(f"  [WARN] Result 파일을 찾을 수 없습니다.")
            continue
        
        # score 폴더 생성
        model_score_path.mkdir(parents=True, exist_ok=True)
        
        try:
            print(f"  [RUN] BFCL 평가 시스템 실행 중...")
            
            # BFCL의 evaluation_main 함수 호출
            # 이 함수가 result 파일들을 읽고 정확한 score 파일들을 생성함
            evaluation_main(
                model=[model_name],  # 특정 모델만 평가
                test_categories=["all"],  # 모든 카테고리 평가
                result_dir=str(result_path),  # result 폴더 경로
                score_dir=str(score_path)     # score 폴더 경로
            )
            
            # 생성된 score 파일들 확인
            score_files = list(model_score_path.glob("*.json"))
            print(f"  [OK] 완료: {len(score_files)}개의 score 파일 생성됨")
            
        except Exception as e:
            print(f"  [FAIL] 평가 실패: {e}")
            print(f"     상세 오류: {type(e).__name__}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("[COMPLETE] 변환 완료!")
    
    # 최종 결과 요약
    print("\n[SUMMARY] 생성된 score 파일 요약:")
    for model_name in target_models:
        model_score_path = score_path / model_name
        if model_score_path.exists():
            score_files = list(model_score_path.glob("*.json"))
            print(f"  [DIR] {model_name}: {len(score_files)}개 파일")
            for score_file in sorted(score_files):
                try:
                    import json
                    with open(score_file, 'r', encoding='utf-8') as f:
                        score_data = json.load(f)
                    
                    if isinstance(score_data, dict) and 'accuracy' in score_data:
                        accuracy = score_data['accuracy']
                        total = score_data.get('total_count', 0)
                        correct = score_data.get('correct_count', 0)
                        print(f"    - {score_file.name}: 정확도 {accuracy:.3f} ({correct}/{total})")
                    else:
                        print(f"    - {score_file.name}: 생성됨")
                except:
                    print(f"    - {score_file.name}: 생성됨 (읽기 오류)")
        else:
            print(f"  [WARN] {model_name}: score 폴더 없음")

if __name__ == "__main__":
    main()