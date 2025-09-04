# BFCL 불공정 평가 이슈 식별 및 분류 시스템

BFCL (Berkeley Function Calling Leaderboard) 벤치마크에서 불공정한 평가 사례를 자동으로 탐지하고 분류하는 시스템입니다.

## 🎯 시스템 목적

모델의 진짜 능력과 무관한 기술적/인프라 문제로 인한 불공정한 평가를 식별하여 벤치마크의 신뢰성을 향상시킵니다.

## 📁 파일 구조

### 핵심 구현 파일
- **`unfair_evaluation_detector.py`** - 불공정 평가 탐지 메인 클래스
- **`data_loader.py`** - BFCL 데이터 로딩 시스템
- **`main_unfair_evaluation_analysis.py`** - 완전한 메인 실행 스크립트
- **`run_analysis.py`** - 빠른 분석용 실행 스크립트
- **`test_system.py`** - 시스템 테스트 스크립트

### 분석 결과 파일
- **`unfair_evaluation_analysis.csv`** - 전체 평가 데이터 + 불공정 분류
- **`issue_classification_summary.csv`** - 이슈 카테고리별 요약
- **`model_issue_breakdown.csv`** - 모델별 이슈 분석
- **`priority_fixes_required.csv`** - P0 (Critical) 긴급 수정 항목
- **`detection_results.json`** - JSON 형식 상세 결과

## 🔍 탐지 카테고리

### Priority P0 (Critical) - 즉시 수정 필요
- **Technical Errors**: API 오류, 토큰 실패, 타임아웃 등
- **API Configuration Bias**: 모델별 API 설정 차별 (예: Claude max_tokens 요구사항)

### Priority P1 (High)  
- **Parsing Failures**: 올바른 답변의 파싱 실패
- **State Management Issues**: Multi-turn 대화 상태 관리 문제

### Priority P2 (Medium)
- **Infrastructure Dependencies**: 모델별 인프라 의존성
- **Format Discrimination**: 의미적으로 올바르지만 형식이 다른 경우

## 🚀 사용법

### 1. 빠른 분석 (샘플 데이터)
```bash
python run_analysis.py
```

### 2. 전체 분석
```bash
python main_unfair_evaluation_analysis.py
```

### 3. 옵션을 사용한 분석
```bash
# 특정 모델만 분석
python main_unfair_evaluation_analysis.py --models anthropic_claude-4-sonnet-thinking-off

# 특정 카테고리만 분석  
python main_unfair_evaluation_analysis.py --test-categories irrelevance simple

# 출력 디렉토리 지정
python main_unfair_evaluation_analysis.py --output-dir "./results"
```

### 4. 시스템 테스트
```bash
python test_system.py
```

## 📊 주요 발견사항

### Claude-4-Sonnet 체계적 실패
- **샘플 1000개 케이스 중 100% 불공정 평가** 탐지
- 모든 케이스가 P0 (Critical) 기술적 오류:
  - `zero_token_failure`: 입력/출력 토큰 모두 0
  - `timeout_before_execution`: 처리 전 타임아웃  
  - `systematic_model_failure`: 체계적 실패 패턴

### 전체 데이터세트 통계
- **총 69,604개 평가 기록** 로드 성공
- **16개 모델** 분석 완료
- JSONLines 형식 데이터 처리

## ⚠️ 긴급 조치 필요 사항

**CRITICAL**: Claude-4-Sonnet에서 발견된 체계적 실패는 모델 능력과 무관한 기술적 문제입니다. 이런 평가는 벤치마크 신뢰성을 심각하게 훼손하므로 즉시 수정이 필요합니다.

## 🔧 시스템 기능

- ✅ 체계적 실패 패턴 자동 탐지
- ✅ API 설정 편향 식별 
- ✅ 파싱 실패 vs 실제 성능 구분
- ✅ 우선순위 기반 분류 (P0~P2)
- ✅ 종합 리포트 자동 생성
- ✅ 다양한 출력 형식 (CSV, JSON)

## 💡 확장 가능성

이 시스템은 다른 벤치마크에도 적용 가능하도록 설계되었으며, 새로운 불공정 패턴이 발견될 때 쉽게 확장할 수 있습니다.

---

**생성일**: 2025년 9월 2일  
**시스템 버전**: 1.0  
**분석 대상**: BFCL v3 벤치마크 결과