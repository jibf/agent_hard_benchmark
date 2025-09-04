#!/usr/bin/env python3
"""
BFCL Benchmark Analysis Results Visualization
Create key charts for team sharing
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# Set style
plt.style.use('default')
sns.set_palette("husl")

def create_summary_charts():
    """Create key analysis result visualization charts"""
    
    # 1. Overall Infrastructure Health Status
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('BFCL Benchmark Unfair Evaluation Issues Analysis Summary', fontsize=16, fontweight='bold')
    
    # Chart 1: Overall Success Rate (Pie Chart)
    success_data = [81.5, 18.5]
    colors = ['#2E8B57', '#DC143C']  # Success: Green, Failure: Red
    wedges, texts, autotexts = ax1.pie(success_data, 
                                       labels=['Success (81.5%)', 'Failure (18.5%)'], 
                                       colors=colors, 
                                       autopct='%1.1f%%',
                                       startangle=90,
                                       textprops={'fontsize': 10})
    ax1.set_title('Overall Infrastructure Status\nGOOD (81.5% Success Rate)', fontsize=12, fontweight='bold')
    
    # Chart 2: Success Rate by Category
    categories = ['Multi-turn\nConversation', 'Simple\nFunction', 'Multiple\nFunctions', 
                 'Parallel\nFunctions', 'Parallel\nMultiple', 'Function\nRelevance',
                 'REST API', 'SQL', 'Java', 'JavaScript', 'Executable', 'AST', 'Relevance']
    success_rates = [0.0, 100.0, 100.0, 100.0, 90.0, 100.0, 87.5, 100.0, 100.0, 100.0, 86.7, 90.0, 83.3]
    
    bars = ax2.bar(range(len(categories)), success_rates, 
                   color=['#DC143C' if rate == 0.0 else '#FFD700' if rate < 90.0 else '#2E8B57' for rate in success_rates])
    ax2.set_title('Success Rate by Category\nCRITICAL: Multi-turn Complete Failure (0%)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Success Rate (%)')
    ax2.set_xticks(range(len(categories)))
    ax2.set_xticklabels(categories, rotation=45, ha='right', fontsize=9)
    ax2.set_ylim(0, 110)
    
    # Add success rate labels
    for i, (bar, rate) in enumerate(zip(bars, success_rates)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=8)
    
    # Chart 3: Model Performance Ranking (Top 8)
    model_names = ['claude-4-sonnet-thinking-off', 'claude-4-sonnet-thinking-on-10k', 
                   'claude-3-5-sonnet-20241022', 'gpt-4o-2024-08-06', 
                   'gemini-1.5-pro-002', 'qwen2.5-72b-instruct', 
                   'pixtral-12b-2409', 'llama-3.1-8b-instruct']
    model_scores = [84.2, 84.0, 83.8, 82.5, 80.1, 78.3, 76.5, 73.1]
    
    bars = ax3.barh(range(len(model_names)), model_scores, color='#4682B4')
    ax3.set_title('Model Performance Ranking (Top 8)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Success Rate (%)')
    ax3.set_yticks(range(len(model_names)))
    ax3.set_yticklabels([name.replace('anthropic_', '').replace('openai_', '').replace('google_', '') 
                        for name in model_names], fontsize=9)
    ax3.set_xlim(70, 90)
    
    # Add score labels
    for i, (bar, score) in enumerate(zip(bars, model_scores)):
        width = bar.get_width()
        ax3.text(width + 0.5, bar.get_y() + bar.get_height()/2.,
                f'{score:.1f}%', ha='left', va='center', fontsize=9)
    
    # Chart 4: Critical Issues Summary
    ax4.axis('off')
    
    # Critical Issues text box
    critical_text = """CRITICAL ISSUES

1. Multi-turn Conversation Complete Failure
   • Success Rate: 0.0%
   • Impact: All Multi-turn tests
   • Status: Urgent fix required

2. Initial Analysis Error (Resolved)
   • Cause: Wrong field mapping
   • Result: 100% -> 81.5% corrected
   • Lesson: JSON structure validation essential

GOOD NEWS
• Performance Inversion: 0 cases
• Model Family Bias: None detected
• Infrastructure: Stable (81.5%)"""
    
    ax4.text(0.05, 0.95, critical_text, transform=ax4.transAxes, 
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#FFF8DC", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('E:\\Users\\김현준\\Downloads\\agent_hard_benchmark_2\\gorilla\\berkeley-function-call-leaderboard\\BFCL_Analysis_Summary_Charts.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Priority Issues Heatmap
    create_priority_heatmap()

def create_priority_heatmap():
    """Create priority-based issue distribution heatmap"""
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Priority matrix data
    categories = ['Multi-turn Conv', 'Simple Func', 'Multiple Func', 'Parallel Func', 
                 'Function Relevance', 'REST API', 'SQL', 'Executable', 'AST', 'Relevance']
    priorities = ['P0 Critical', 'P1 High', 'P2 Medium', 'No Issue']
    
    # Priority matrix based on actual analysis results
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
    
    # Create heatmap
    sns.heatmap(priority_matrix, 
                xticklabels=priorities, 
                yticklabels=categories,
                annot=True, 
                cmap=['white', '#FFE4B5', '#FFA500', '#DC143C'],  # White, Beige, Orange, Red
                cbar_kws={'label': 'Issue Count'},
                ax=ax)
    
    ax.set_title('BFCL Benchmark Priority-based Issue Distribution\n(P0: Critical, P1: High, P2: Medium)', 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Priority Level', fontsize=12)
    ax.set_ylabel('Test Category', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('E:\\Users\\김현준\\Downloads\\agent_hard_benchmark_2\\gorilla\\berkeley-function-call-leaderboard\\BFCL_Priority_Issues_Heatmap.png', 
                dpi=300, bbox_inches='tight')
    plt.close()

def create_action_plan_chart():
    """Create action plan timeline chart"""
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    # Action items
    actions = [
        'Fix Multi-turn Conversation Logic',
        'Document JSON Structure',
        'Enhance Error Handling', 
        'Build Monitoring Dashboard',
        'Automate Regression Testing',
        'Performance Trend Analysis System'
    ]
    
    priorities = ['P0', 'P1', 'P1', 'P1', 'P2', 'P2']
    timelines = ['Immediate', '1 Week', '1 Week', '2 Weeks', '1 Month', '2 Months']
    
    # Priority colors
    colors = {'P0': '#DC143C', 'P1': '#FFA500', 'P2': '#32CD32'}
    
    # Bar chart
    bars = ax.barh(range(len(actions)), [1, 7, 7, 14, 30, 60], 
                   color=[colors[p] for p in priorities])
    
    ax.set_title('BFCL Benchmark Improvement Action Plan Timeline', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Expected Duration (Days)', fontsize=12)
    ax.set_yticks(range(len(actions)))
    ax.set_yticklabels([f"{p}: {action}" for p, action in zip(priorities, actions)], fontsize=10)
    
    # Add legend
    legend_elements = [mpatches.Patch(color=colors[p], label=p) for p in ['P0', 'P1', 'P2']]
    ax.legend(handles=legend_elements, title='Priority', loc='lower right')
    
    plt.tight_layout()
    plt.savefig('E:\\Users\\김현준\\Downloads\\agent_hard_benchmark_2\\gorilla\\berkeley-function-call-leaderboard\\BFCL_Action_Plan_Timeline.png', 
                dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    print("Creating BFCL Benchmark analysis result visualizations...")
    
    try:
        create_summary_charts()
        print("SUCCESS: Summary charts created - BFCL_Analysis_Summary_Charts.png")
        
        print("SUCCESS: Priority heatmap created - BFCL_Priority_Issues_Heatmap.png")
        
        create_action_plan_chart()
        print("SUCCESS: Action plan timeline created - BFCL_Action_Plan_Timeline.png")
        
        print("\nAll visualization charts have been successfully created!")
        
    except Exception as e:
        print(f"ERROR during visualization creation: {e}")
        import traceback
        traceback.print_exc()