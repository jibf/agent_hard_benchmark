#!/usr/bin/env python3
import json
import shutil

# 원본 파일 백업
input_file = "/Users/seojune/Desktop/ComplexFuncBench/result/togetherai/moonshotai/Kimi-K2-Instruct/full-1000.jsonl"
backup_file = "/Users/seojune/Desktop/ComplexFuncBench/result/togetherai/moonshotai/Kimi-K2-Instruct/full-1000_backup.jsonl"
output_file = "/Users/seojune/Desktop/ComplexFuncBench/result/togetherai/moonshotai/Kimi-K2-Instruct/full-1000_filtered.jsonl"

# 백업 생성
shutil.copy2(input_file, backup_file)
print(f"백업 파일 생성됨: {backup_file}")

# Cross-200부터 Cross-233까지의 라인 제거
removed_count = 0
total_lines = 0

with open(input_file, 'r', encoding='utf-8') as infile, \
     open(output_file, 'w', encoding='utf-8') as outfile:
    
    for line in infile:
        total_lines += 1
        try:
            # JSON 파싱해서 id 확인
            data = json.loads(line.strip())
            cross_id = data.get('id', '')
            
            # Cross-200부터 Cross-233 범위인지 확인
            if cross_id.startswith('Cross-'):
                try:
                    number = int(cross_id.split('-')[1])
                    if 200 <= number <= 233:
                        removed_count += 1
                        print(f"제거됨: {cross_id}")
                        continue  # 이 라인은 출력 파일에 쓰지 않음
                except (ValueError, IndexError):
                    pass
            
            # 해당 범위가 아니면 출력 파일에 쓰기
            outfile.write(line)
            
        except json.JSONDecodeError:
            # JSON 파싱 실패시에도 라인 유지
            outfile.write(line)

print(f"\n처리 완료:")
print(f"전체 라인 수: {total_lines}")
print(f"제거된 라인 수: {removed_count}")
print(f"남은 라인 수: {total_lines - removed_count}")
print(f"\n파일 저장됨: {output_file}")
print(f"백업 파일: {backup_file}")