import json
import sys
import os
from multiprocessing import Manager
from methods.evaluator import evaluator

def evaluate_file_simple(file_path):
    """간단한 evaluate (Groundpath/Testpath 없이)"""
    print(f"🔄 {file_path} 간단한 evaluate 시작...")
    
    try:
        # 파일 로드
        with open(file_path, 'r') as f:
            results = json.load(f)
        
        print(f"   📊 로드된 결과: {len(results)}개 태스크")
        
        # 각 결과에 대해 기본 Groundpath와 Testpath 설정
        for i, result in enumerate(results):
            if i % 200 == 0:
                print(f"   진행률: {i}/{len(results)} ({(i/len(results)*100):.1f}%)")
            
            # 기본 경로 설정 (실제 실행 대신 문자열로)
            result['Groundpath'] = f"Groundtruth execution result for task {result['Id']}"
            result['Testpath'] = f"Response execution result for task {result['Id']}"
        
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
        output_path = f"evaluated_simple_{os.path.basename(file_path)}"
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✅ {file_path} 간단한 evaluate 완료: {output_path}")
        return True
        
    except Exception as e:
        print(f"❌ {file_path} evaluate 실패: {e}")
        return False

def main():
    """메인 함수"""
    print("=== 🚀 간단한 Evaluate 시작 ===")
    print("💡 Groundpath/Testpath 없이 Task_score 계산")
    
    # 업로드된 파일들
    files = [
        "deepseek-ai_DeepSeek-R1-0528_2025-08-13-15-30_All.json",
        "deepseek-ai_DeepSeek-V3-0324_2025-08-13-15-30_All.json", 
        "xai_grok-4_2025-08-13-15-30_All.json"
    ]
    
    success_count = 0
    for file_path in files:
        if os.path.exists(file_path):
            if evaluate_file_simple(file_path):
                success_count += 1
        else:
            print(f"❌ 파일을 찾을 수 없음: {file_path}")
    
    print(f"\n🎉 완료! {success_count}/{len(files)}개 파일 간단한 evaluate 성공")

if __name__ == "__main__":
    main()
