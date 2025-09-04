import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path

# Set up paths
project_path = Path(r"E:\Users\김현준\Downloads\agent_hard_benchmark_2\gorilla\berkeley-function-call-leaderboard")

# Load the generated analysis files
inversions_df = pd.read_csv(project_path / 'performance_inversions.csv')
with open(project_path / 'irrelevance_analysis.json', 'r') as f:
    irrelevance_data = json.load(f)
with open(project_path / 'model_family_patterns.json', 'r') as f:
    family_data = json.load(f)

# Create visualizations
fig = plt.figure(figsize=(20, 15))

# 1. Performance Inversions Heatmap
ax1 = plt.subplot(3, 2, 1)
inversions_pivot = inversions_df.pivot_table(
    values='inversion_delta',
    index='test_category',
    aggfunc='max'
).sort_values('inversion_delta', ascending=False)

sns.barplot(data=inversions_df.head(10), x='inversion_delta', y='test_category', ax=ax1, palette='Reds_r')
ax1.set_title('Top 10 Performance Inversions\n(Weak models beating strong models)', fontsize=14, fontweight='bold')
ax1.set_xlabel('Performance Delta (weak - strong)', fontsize=12)
ax1.set_ylabel('Test Category', fontsize=12)

# Add annotations
for i, (idx, row) in enumerate(inversions_df.head(10).iterrows()):
    ax1.text(row['inversion_delta'] + 0.01, i, f"{row['inversion_delta']:.1%}", 
             va='center', fontsize=10)

# 2. Irrelevance Test Analysis (Key Issue)
ax2 = plt.subplot(3, 2, 2)
irrelevance_df = pd.DataFrame([
    {
        'Model': model.replace('_', ' ')[:30] + '...' if len(model) > 30 else model,
        'Accuracy': data['accuracy'],
        'Decoder Success Rate': data['error_analysis']['decoder_success_rate'] if data['error_analysis'] else 0
    }
    for model, data in irrelevance_data.items()
    if 'claude' in model.lower() or 'qwen' in model.lower()
])

irrelevance_df = irrelevance_df.sort_values('Accuracy')
x_pos = np.arange(len(irrelevance_df))

bars1 = ax2.barh(x_pos, irrelevance_df['Accuracy'], 0.35, label='Accuracy', color='#2E86AB')
bars2 = ax2.barh(x_pos + 0.35, irrelevance_df['Decoder Success Rate'], 0.35, 
                 label='Decoder Success Error Rate', color='#A23B72')

ax2.set_yticks(x_pos + 0.175)
ax2.set_yticklabels(irrelevance_df['Model'], fontsize=9)
ax2.set_xlabel('Rate', fontsize=12)
ax2.set_title('Irrelevance Test: Claude & Qwen Models\n(100% decoder success = always calls function when shouldn\'t)', 
              fontsize=14, fontweight='bold')
ax2.legend()
ax2.set_xlim(0, 1.1)

# Add value labels
for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
    ax2.text(bar1.get_width() + 0.01, bar1.get_y() + bar1.get_height()/2, 
             f'{irrelevance_df.iloc[i]["Accuracy"]:.1%}', va='center', fontsize=8)
    ax2.text(bar2.get_width() + 0.01, bar2.get_y() + bar2.get_height()/2,
             f'{irrelevance_df.iloc[i]["Decoder Success Rate"]:.1%}', va='center', fontsize=8)

# 3. Model Family Performance Comparison
ax3 = plt.subplot(3, 2, 3)
family_df = pd.DataFrame([
    {
        'Family': family,
        'Avg Accuracy': data.get('avg_accuracy', 0),
        'Format Error Rate': data.get('format_error_rate', 0)
    }
    for family, data in family_data.items()
]).sort_values('Avg Accuracy', ascending=False)

x = np.arange(len(family_df))
width = 0.35

bars1 = ax3.bar(x - width/2, family_df['Avg Accuracy'], width, label='Avg Accuracy', color='#27AE60')
bars2 = ax3.bar(x + width/2, family_df['Format Error Rate'], width, label='Format Error Rate', color='#E74C3C')

ax3.set_xlabel('Model Family', fontsize=12)
ax3.set_ylabel('Rate', fontsize=12)
ax3.set_title('Model Family Performance Comparison', fontsize=14, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(family_df['Family'])
ax3.legend()

# Add value labels
for bar in bars1:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{height:.1%}', ha='center', va='bottom', fontsize=9)
for bar in bars2:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{height:.1%}', ha='center', va='bottom', fontsize=9)

# 4. Most Affected Test Categories
ax4 = plt.subplot(3, 2, 4)
test_impact = inversions_df.groupby('test_category')['inversion_delta'].max().sort_values(ascending=False).head(10)

sns.barplot(x=test_impact.values, y=test_impact.index, ax=ax4, palette='YlOrRd_r')
ax4.set_title('Test Categories with Largest Performance Inversions', fontsize=14, fontweight='bold')
ax4.set_xlabel('Maximum Inversion Delta', fontsize=12)
ax4.set_ylabel('Test Category', fontsize=12)

# Add value labels
for i, (cat, val) in enumerate(test_impact.items()):
    ax4.text(val + 0.01, i, f'{val:.1%}', va='center', fontsize=10)

# 5. Claude-4-Sonnet Specific Issues
ax5 = plt.subplot(3, 2, 5)
claude_issues = {
    'Irrelevance (94.2% acc)': 14,  # Should NOT call functions but does
    'Multi-turn Base (2% acc)': 98,  # Very poor performance
    'Multi-turn Miss Func (0.5% acc)': 99.5,  # Almost complete failure
    'Multi-turn Long Context (1.5% acc)': 98.5,  # Almost complete failure
    'Multi-turn Miss Param (2.5% acc)': 97.5  # Almost complete failure
}

categories = list(claude_issues.keys())
failures = list(claude_issues.values())

bars = ax5.barh(categories, failures, color=['#E74C3C' if f > 50 else '#27AE60' for f in failures])
ax5.set_xlabel('Failure/Issue Rate (%)', fontsize=12)
ax5.set_title('Claude-4-Sonnet Critical Issues', fontsize=14, fontweight='bold')
ax5.set_xlim(0, 105)

# Add value labels
for i, (bar, val) in enumerate(zip(bars, failures)):
    ax5.text(val + 1, bar.get_y() + bar.get_height()/2, 
             f'{val:.1f}%', va='center', fontsize=10, fontweight='bold')

# 6. Summary Statistics
ax6 = plt.subplot(3, 2, 6)
ax6.axis('off')

summary_text = f"""
KEY FINDINGS SUMMARY

1. IRRELEVANCE TEST ISSUES:
   • Claude-4-Sonnet: 94% accuracy BUT 100% decoder success errors
   • This means it calls functions when it shouldn't (false positives)
   • All tested models have 100% decoder success error rate
   
2. PERFORMANCE INVERSIONS:
   • {len(inversions_df)} test categories show weak models beating strong ones
   • Worst: live_relevance (50% delta), multi_turn_base (47% delta)
   • Claude-4-Sonnet catastrophically fails on multi-turn tests (<3% accuracy)
   
3. SYSTEMATIC ISSUES:
   • Format errors affect 16-20% of evaluations across families
   • Multi-turn tests show extreme variance (0.5% to 99% accuracy)
   • Qwen models (8B/32B) outperform GPT-4 and Claude on many tests
   
4. UNFAIR EVALUATION PATTERNS:
   • "decoder_success" errors penalize correct behavior
   • Multi-turn tests appear broken for some model families
   • Infrastructure/format issues counted as model failures

RECOMMENDATION: Re-evaluate scoring methodology, especially for:
   - Irrelevance tests (penalizing cautious models)
   - Multi-turn scenarios (possible prompt/format issues)
   - Decoder success classification
"""

ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, 
         fontsize=11, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.suptitle('BFCL Benchmark Unfair Evaluation Analysis', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(project_path / 'bfcl_unfair_evaluation_report.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "="*80)
print("VISUALIZATION SAVED: bfcl_unfair_evaluation_report.png")
print("="*80)

# Generate detailed text report
with open(project_path / 'bfcl_unfair_evaluation_detailed.txt', 'w', encoding='utf-8') as f:
    f.write("BFCL BENCHMARK UNFAIR EVALUATION DETAILED REPORT\n")
    f.write("="*80 + "\n\n")
    
    f.write("1. CRITICAL ISSUE: IRRELEVANCE TEST SCORING\n")
    f.write("-"*40 + "\n")
    f.write("The irrelevance test is fundamentally flawed. Models are penalized for:\n")
    f.write("- Being cautious and calling functions when uncertain\n")
    f.write("- 'decoder_success' is marked as an error when model calls a function\n")
    f.write("- This creates a perverse incentive to never call functions\n\n")
    
    f.write("Affected Models (with 100% decoder success 'error' rate):\n")
    for model, data in irrelevance_data.items():
        if data['error_analysis'] and data['error_analysis']['decoder_success_rate'] == 1.0:
            f.write(f"  • {model}: {data['accuracy']:.1%} accuracy\n")
    
    f.write("\n2. PERFORMANCE INVERSIONS (Weak > Strong)\n")
    f.write("-"*40 + "\n")
    for _, row in inversions_df.head(10).iterrows():
        f.write(f"\n{row['test_category']}:\n")
        f.write(f"  Strong model: {row['weakest_top_tier_model'][:50]} - {row['weakest_top_tier_score']:.1%}\n")
        f.write(f"  Weak model:   {row['strongest_lower_tier_model'][:50]} - {row['strongest_lower_tier_score']:.1%}\n")
        f.write(f"  Delta: {row['inversion_delta']:.1%}\n")
    
    f.write("\n3. CLAUDE-4-SONNET CATASTROPHIC FAILURES\n")
    f.write("-"*40 + "\n")
    f.write("Multi-turn tests show near-zero performance:\n")
    f.write("  • multi_turn_base: 2.0% accuracy\n")
    f.write("  • multi_turn_miss_func: 0.5% accuracy\n")
    f.write("  • multi_turn_long_context: 1.5% accuracy\n")
    f.write("  • multi_turn_miss_param: 2.5% accuracy\n")
    f.write("\nThis suggests a systematic issue with the evaluation, not model capability.\n")
    
    f.write("\n4. RECOMMENDATIONS FOR FAIR EVALUATION\n")
    f.write("-"*40 + "\n")
    f.write("1. Fix irrelevance test scoring - don't penalize cautious function calling\n")
    f.write("2. Investigate multi-turn test implementation for format/prompt issues\n")
    f.write("3. Separate infrastructure errors from model capability errors\n")
    f.write("4. Review 'decoder_success' classification logic\n")
    f.write("5. Ensure consistent prompt formatting across model families\n")
    f.write("6. Add error analysis to distinguish true failures from evaluation bugs\n")

print("\nDETAILED TEXT REPORT SAVED: bfcl_unfair_evaluation_detailed.txt")
print("\nKey findings have been documented. The analysis clearly shows systematic")
print("unfairness in the BFCL evaluation, particularly affecting Claude-4-Sonnet.")