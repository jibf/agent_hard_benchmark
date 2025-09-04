#!/usr/bin/env python3
"""
BFCL 벤치마크 분석 결과 시각화 생성
팀원 공유용 핵심 차트 생성
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# 한글 폰트 설정
plt.rcParams['font.family'] = ['Arial Unicode MS', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

def create_summary_charts():
    """핵심 분석 결과를 시각화한 차트들 생성"""
    
    # 1. 전체 Infrastructure Health 상태
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('BFCL Benchmark 불공정 평가 이슈 분석 결과 요약', fontsize=16, fontweight='bold')
    
    # Chart 1: 전체 성공률 (Pie Chart)
    success_data = [81.5, 18.5]
    colors = ['#2E8B57', '#DC143C']  # 성공: 초록, 실패: 빨강
    wedges, texts, autotexts = ax1.pie(success_data, 
                                       labels=['성공 (81.5%)', '실패 (18.5%)'], 
                                       colors=colors, 
                                       autopct='%1.1f%%',
                                       startangle=90,
                                       textprops={'fontsize': 10})
    ax1.set_title('전체 Infrastructure 상태\n✅ GOOD (81.5% 성공률)', fontsize=12, fontweight='bold')
    
    # Chart 2: 카테고리별 성공률
    categories = ['Multi-turn\nConversation', 'Simple\nFunction', 'Multiple\nFunctions', 
                 'Parallel\nFunctions', 'Parallel\nMultiple', 'Function\nRelevance',
                 'REST API', 'SQL', 'Java', 'JavaScript', 'Executable', 'AST', 'Relevance']
    success_rates = [0.0, 100.0, 100.0, 100.0, 90.0, 100.0, 87.5, 100.0, 100.0, 100.0, 86.7, 90.0, 83.3]
    
    bars = ax2.bar(range(len(categories)), success_rates, 
                   color=['#DC143C' if rate == 0.0 else '#FFD700' if rate < 90.0 else '#2E8B57' for rate in success_rates])
    ax2.set_title('카테고리별 성공률\n❌ Multi-turn 완전 실패 (0%)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('성공률 (%)')
    ax2.set_xticks(range(len(categories)))
    ax2.set_xticklabels(categories, rotation=45, ha='right', fontsize=9)
    ax2.set_ylim(0, 110)
    
    # 성공률 라벨 추가
    for i, (bar, rate) in enumerate(zip(bars, success_rates)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=8)
    
    # Chart 3: 모델별 성능 순위 (상위 8개)
    model_names = ['claude-4-sonnet-thinking-off', 'claude-4-sonnet-thinking-on-10k', 
                   'claude-3-5-sonnet-20241022', 'gpt-4o-2024-08-06', 
                   'gemini-1.5-pro-002', 'qwen2.5-72b-instruct', 
                   'pixtral-12b-2409', 'llama-3.1-8b-instruct']
    model_scores = [84.2, 84.0, 83.8, 82.5, 80.1, 78.3, 76.5, 73.1]
    
    bars = ax3.barh(range(len(model_names)), model_scores, color='#4682B4')
    ax3.set_title('모델별 성능 순위 (상위 8개)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('성공률 (%)')
    ax3.set_yticks(range(len(model_names)))
    ax3.set_yticklabels([name.replace('anthropic_', '').replace('openai_', '').replace('google_', '') 
                        for name in model_names], fontsize=9)
    ax3.set_xlim(70, 90)
    
    # 점수 라벨 추가
    for i, (bar, score) in enumerate(zip(bars, model_scores)):
        width = bar.get_width()
        ax3.text(width + 0.5, bar.get_y() + bar.get_height()/2.,
                f'{score:.1f}%', ha='left', va='center', fontsize=9)
    
    # Chart 4: Critical Issues 요약
    ax4.axis('off')
    
    # Critical Issues 텍스트 박스
    critical_text = """🚨 CRITICAL ISSUES

1. Multi-turn Conversation 완전 실패
   • 성공률: 0.0%
   • 영향: 모든 Multi-turn 테스트
   • 상태: 긴급 수정 필요

2. 초기 분석 오류 (해결됨)
   • 원인: 잘못된 field mapping
   • 결과: 100% → 81.5% 수정
   • 교훈: JSON 구조 사전 검증 필수

✅ GOOD NEWS
• Performance Inversion: 0건
• 모델 패밀리 편향: 없음
• Infrastructure: 안정적 (81.5%)"""
    
    ax4.text(0.05, 0.95, critical_text, transform=ax4.transAxes, 
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#FFF8DC", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('E:\\Users\\김현준\\Downloads\\agent_hard_benchmark_2\\gorilla\\berkeley-function-call-leaderboard\\BFCL_Analysis_Summary_Charts.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Priority Issues 히트맵
    create_priority_heatmap()

def create_priority_heatmap():
    """우선순위별 이슈 분포 히트맵 생성"""
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Priority matrix 데이터
    categories = ['Multi-turn Conv', 'Simple Func', 'Multiple Func', 'Parallel Func', 
                 'Function Relevance', 'REST API', 'SQL', 'Executable', 'AST', 'Relevance']
    priorities = ['P0 Critical', 'P1 High', 'P2 Medium', 'No Issue']
    
    # 실제 분석 결과를 바탕으로 한 우선순위 매트릭스
    priority_matrix = np.array([
        [1, 0, 0, 0],  # Multi-turn: P0 Critical
        [0, 0, 0, 1],  # Simple: No Issue
        [0, 0, 0, 1],  # Multiple: No Issue
        [0, 0, 0, 1],  # Parallel: No Issue
        [0, 0, 0, 1],  # Function Relevance: No Issue
        [0, 0, 1, 0],  # REST API: P2 Medium
        [0, 0, 0, 1],  # SQL: No Issue
        [0, 0, 1, 0],  # Executable: P2 Medium
        [0, 0, 1, 0],  # AST: P2 Medium
        [0, 0, 1, 0],  # Relevance: P2 Medium
    ])
    
    # 히트맵 생성
    sns.heatmap(priority_matrix, 
                xticklabels=priorities, 
                yticklabels=categories,
                annot=True, 
                cmap=['white', '#FFE4B5', '#FFA500', '#DC143C'],  # 흰색, 베이지, 주황, 빨강
                cbar_kws={'label': 'Issue Count'},
                ax=ax)
    
    ax.set_title('BFCL Benchmark 우선순위별 이슈 분포\n(P0: Critical, P1: High, P2: Medium)', 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('우선순위 레벨', fontsize=12)
    ax.set_ylabel('테스트 카테고리', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('E:\\Users\\김현준\\Downloads\\agent_hard_benchmark_2\\gorilla\\berkeley-function-call-leaderboard\\BFCL_Priority_Issues_Heatmap.png', 
                dpi=300, bbox_inches='tight')
    plt.close()

def create_action_plan_chart():
    """액션 플랜 타임라인 차트"""
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    # 액션 아이템들
    actions = [
        'Multi-turn Conversation 로직 수정',
        'JSON 구조 문서화',
        'Error Handling 강화', 
        '모니터링 대시보드 구축',
        'Regression 테스트 자동화',
        '성능 트렌드 분석 시스템'
    ]
    
    priorities = ['P0', 'P1', 'P1', 'P1', 'P2', 'P2']
    timelines = ['즉시', '1주일', '1주일', '2주일', '1개월', '2개월']
    
    # 우선순위별 색상
    colors = {'P0': '#DC143C', 'P1': '#FFA500', 'P2': '#32CD32'}
    
    # 바 차트
    bars = ax.barh(range(len(actions)), [1, 7, 7, 14, 30, 60], 
                   color=[colors[p] for p in priorities])
    
    ax.set_title('BFCL Benchmark 개선 액션 플랜 타임라인', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('예상 소요 시간 (일)', fontsize=12)
    ax.set_yticks(range(len(actions)))
    ax.set_yticklabels([f"{p}: {action}" for p, action in zip(priorities, actions)], fontsize=10)
    
    # 범례 추가
    legend_elements = [mpatches.Patch(color=colors[p], label=p) for p in ['P0', 'P1', 'P2']]
    ax.legend(handles=legend_elements, title='우선순위', loc='lower right')
    
    plt.tight_layout()
    plt.savefig('E:\\Users\\김현준\\Downloads\\agent_hard_benchmark_2\\gorilla\\berkeley-function-call-leaderboard\\BFCL_Action_Plan_Timeline.png', 
                dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    print("BFCL Benchmark 분석 결과 시각화 생성 중...")
    
    try:
        create_summary_charts()
        print("✅ 요약 차트 생성 완료: BFCL_Analysis_Summary_Charts.png")
        
        print("✅ 우선순위 히트맵 생성 완료: BFCL_Priority_Issues_Heatmap.png")
        
        create_action_plan_chart()
        print("✅ 액션 플랜 타임라인 생성 완료: BFCL_Action_Plan_Timeline.png")
        
        print("\n📊 모든 시각화 차트가 성공적으로 생성되었습니다!")
        
    except Exception as e:
        print(f"❌ 시각화 생성 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()