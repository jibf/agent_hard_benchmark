# BFCL 벤치마크 패턴 분석 결과

이 폴더는 Berkeley Function Calling Leaderboard (BFCL) 벤치마크의 불공정한 평가 패턴을 분석한 결과를 담고 있습니다.

## 📁 파일 목록

### 📊 주요 보고서
- **`UNFAIR_EVALUATION_SUMMARY.md`** - 전체 분석 요약 보고서 (한국어/영어)
- **`bfcl_unfair_evaluation_detailed.txt`** - 상세한 텍스트 보고서
- **`bfcl_unfair_evaluation_report.png`** - 시각화 보고서

### 📈 데이터 분석 파일
- **`bfcl_analysis_detailed.csv`** - 19,449개 평가 기록 상세 분석 (51MB)
- **`performance_inversions.csv`** - 13개 성능 역전 사례 분석
- **`irrelevance_analysis.json`** - Irrelevance 테스트 상세 분석
- **`model_family_patterns.json`** - 모델 패밀리별 패턴 분석

### 🔧 분석 스크립트
- **`bfcl_analysis.py`** - 메인 분석 스크립트
- **`bfcl_detailed_report.py`** - 상세 보고서 생성 스크립트

## 🚨 주요 발견 사항

### 1. Irrelevance Test의 심각한 결함
- 모든 모델에서 100% "decoder_success" 오류율
- 적절한 function calling을 오류로 분류하는 문제

### 2. 성능 역전 현상
- 13개 테스트 카테고리에서 약한 모델이 강한 모델을 이기는 현상
- 최대 50% 성능 차이 발견

### 3. Claude-4-Sonnet의 Multi-turn 테스트 문제
- Multi-turn 테스트에서 0.5-2.5% 극도로 낮은 성능
- 평가 인프라 호환성 문제로 판단

### 4. 기술적/포맷 오류 분류 문제
- 3,167개 format error를 모델 능력 부족으로 잘못 분류

## 🎯 권장 사항

1. **Irrelevance 테스트 스코어링 로직 수정**
2. **Multi-turn 테스트 구현 재검토**
3. **인프라 오류와 모델 능력 구분**
4. **"decoder_success" 분류 기준 재검토**

## 📊 분석 규모

- **평가 기록**: 19,449개
- **모델 수**: 15개
- **테스트 카테고리**: 17개
- **식별된 불공정 패턴**: 다수

## 📞 문의

이 분석에 대한 질문이나 추가 정보가 필요하시면 언제든 연락주세요.

---
*분석 수행일: 2025년 9월 2일*