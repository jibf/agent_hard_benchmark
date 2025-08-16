import json
import sys
import os
import signal
from multiprocessing import Manager
from methods.evaluator import evaluator
from methods.collect_result import execute_code, process_code

def timeout_handler(signum, frame):
    raise TimeoutError("Code execution timed out")

def safe_execute_code(code, timeout=5):
    """안전한 코드 실행 (타임아웃 포함)"""
    try:
        # 타임아웃 설정
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)
        
        result = execute_code(code)
        signal.alarm(0)  # 타임아웃 해제
        return result
    except TimeoutError:
        return "Timeout: Code execution took too long"
    except Exception as e:
        return f"Error: {str(e)}"
    finally:
        signal.alarm(0)  # 확실히 타임아웃 해제

def make_json_serializable(obj):
    """복잡한 객체를 JSON 직렬화 가능하게 변환"""
    if hasattr(obj, '__dict__'):
        return str(obj)
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    else:
        return obj

def evaluate_file_fast(file_path):
    """빠른 evaluate (타임아웃 방지)"""
    print(f"🔄 {file_path} 빠른 evaluate 시작...")
    
    try:
        # 파일 로드
        with open(file_path, 'r') as f:
            results = json.load(f)
        
        print(f"   📊 로드된 결과: {len(results)}개 태스크")
        
        # 각 결과에 대해 Groundpath와 Testpath 생성 (타임아웃 포함)
        for i, result in enumerate(results):
            if i % 200 == 0:
                print(f"   진행률: {i}/{len(results)} ({(i/len(results)*100):.1f}%)")
            
            # Groundtruth 실행하여 Groundpath 생성 (5초 타임아웃)
            try:
                ground_result = safe_execute_code(result['Groundtruth'], timeout=5)
                result['Groundpath'] = make_json_serializable(ground_result)
            except Exception as e:
                result['Groundpath'] = f"Error: {str(e)}"
            
            # Response_code 실행하여 Testpath 생성 (5초 타임아웃)
            try:
                test_result = safe_execute_code(result['Response_code'], timeout=5)
                result['Testpath'] = make_json_serializable(test_result)
            except Exception as e:
                result['Testpath'] = f"Error: {str(e)}"
        
        # Evaluate 수행
        print(f"   🎯 Task_score 계산 중...")
        eval_results = Manager().list()
        
        for i, result in enumerate(results):
            try:
                evaluator(file_path, eval_results, result)
                if (i + 1) % 200 == 0:
                    print(f"   Evaluate 진행률: {i+1}/{len(results)} ({((i+1)/len(results)*100):.1f}%)")
            except Exception as e:
                print(f"   ⚠️ Evaluate 오류 (인덱스 {i}): {e}")
                # 오류가 있어도 계속 진행
                continue
        
        # Task_score 추가
        eval_list = list(eval_results)
        for i, result in enumerate(results):
            if i < len(eval_list):
                result['Task_score'] = eval_list[i]['Task_score']
        
        # 결과 저장
        output_path = f"evaluated_fast_{os.path.basename(file_path)}"
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✅ {file_path} 빠른 evaluate 완료: {output_path}")
        return True
        
    except Exception as e:
        print(f"❌ {file_path} evaluate 실패: {e}")
        return False

def main():
    """메인 함수"""
    print("=== 🚀 업로드된 파일들 빠른 Evaluate 시작 ===")
    print("💡 타임아웃: 5초, 진행률 표시: 200개마다")
    
    # 업로드된 파일들
    files = [
        "deepseek-ai_DeepSeek-R1-0528_2025-08-13-15-30_All.json",
        "deepseek-ai_DeepSeek-V3-0324_2025-08-13-15-30_All.json", 
        "xai_grok-4_2025-08-13-15-30_All.json"
    ]
    
    success_count = 0
    for file_path in files:
        if os.path.exists(file_path):
            if evaluate_file_fast(file_path):
                success_count += 1
        else:
            print(f"❌ 파일을 찾을 수 없음: {file_path}")
    
    print(f"\n🎉 완료! {success_count}/{len(files)}개 파일 빠른 evaluate 성공")

if __name__ == "__main__":
    main()
