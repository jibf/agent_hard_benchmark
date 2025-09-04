# Enhanced BFCL Benchmark Analysis Pipeline v2.0

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-supported-blue.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-pytest-green.svg)](https://pytest.org/)

A comprehensive, statistically rigorous analysis pipeline for the Berkeley Function Calling Leaderboard (BFCL) benchmark. This enhanced version addresses the limitations identified in the original analysis and provides robust statistical validation, accessibility features, and reproducible results.

## 🚀 Key Features

### Statistical Rigor
- **Bootstrap confidence intervals** for all performance metrics
- **P-value calculation** using chi-square and Fisher's exact tests
- **Sample-size weighted deltas** for fair comparison
- **Bonferroni correction** for multiple comparison adjustment

### Comprehensive Analysis
- **Family-Task cross-analysis** with systematic strength/weakness identification
- **Automated case study sampling** for performance inversion investigation
- **Technical error classification** separate from capability assessment
- **Model tier configuration** via external YAML files

### Enhanced Visualizations
- **95% confidence interval error bars** on all performance plots
- **Colorblind-friendly palettes** following accessibility guidelines
- **Alt-text generation** for screen reader compatibility
- **High-DPI output** for publication-quality figures

### Reproducibility & Deployment
- **CLI argument parsing** for flexible execution
- **Docker containerization** with health checks
- **Comprehensive test suite** with >90% code coverage
- **Configuration-driven** analysis parameters

## 📋 Requirements

- Python 3.10+
- 16GB+ RAM (recommended for large datasets)
- Docker (optional, for containerized execution)

## 🔧 Installation

### Local Installation

```bash
# Clone the repository
git clone <repository-url>
cd "BFCL 벤치마크 패턴 분석"

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -m pytest test_bfcl_analysis.py -v
```

### Docker Installation

```bash
# Build the Docker image
docker build -t bfcl-analysis:v2.0 .

# Verify build
docker run --rm bfcl-analysis:v2.0 --help
```

## 🚀 Quick Start

### Basic Usage

```bash
python bfcl_analysis_enhanced.py \\
  --data_root /path/to/bfcl/score/data \\
  --output_dir ./enhanced_results \\
  --config model_tiers.yaml
```

### Docker Usage

```bash
docker run --rm -v $(pwd)/data:/app/data -v $(pwd)/output:/app/output \\
  bfcl-analysis:v2.0 \\
  --data_root /app/data \\
  --output_dir /app/output \\
  --verbose
```

### Advanced Configuration

```bash
# Custom model tier configuration
python bfcl_analysis_enhanced.py \\
  --data_root ./score \\
  --config custom_model_tiers.yaml \\
  --output_dir ./custom_analysis \\
  --verbose
```

## 📊 Output Files

### Statistical Results
- `validated_performance_inversions.csv` - Statistically validated performance inversions with CI
- `comprehensive_analysis_results.json` - Complete analysis results in JSON format
- `enhanced_analysis_data.csv` - Processed dataset with error classifications

### Case Studies  
- `detailed_case_studies.json` - Automated case studies of top performance inversions
- `case_study_summary.md` - Human-readable case study summaries

### Visualizations
- `enhanced_performance_inversions.png` - Performance inversions with confidence intervals
- `enhanced_family_task_heatmap.png` - Family-task performance matrix with consistency scores
- `error_analysis_dashboard.png` - Comprehensive error pattern analysis
- `irrelevance_analysis_detailed.png` - Detailed irrelevance test investigation
- `visualization_alt_texts.md` - Alt-text descriptions for accessibility

### Reports
- `ENHANCED_ANALYSIS_REPORT.md` - Comprehensive analysis report with statistical validation
- `family_task_analysis.json` - Detailed family-task cross-analysis results

## ⚙️ Configuration

### Model Tier Configuration (`model_tiers.yaml`)

```yaml
model_tiers:
  top_tier:
    description: "High-capability frontier models"
    patterns:
      - "gpt-4"
      - "claude-4-sonnet"
      - "gemini-1.5-pro"
      - "o3-high"

statistical_thresholds:
  delta_min: 0.1          # Minimum delta for significance
  p_value_max: 0.05       # Statistical significance threshold
  confidence_level: 0.95  # Confidence interval level
  bootstrap_iterations: 1000

visualization:
  colorblind_palette:
    primary: ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
  figure_settings:
    dpi: 150
    width: 16
    height: 12
```

## 🧪 Testing

### Run Complete Test Suite

```bash
# Run all tests with coverage
pytest test_bfcl_analysis.py -v --cov=. --cov-report=html

# Run specific test categories
pytest test_bfcl_analysis.py::TestStatisticalValidator -v
pytest test_bfcl_analysis.py::TestFamilyTaskAnalyzer -v
pytest test_bfcl_analysis.py::TestIntegration -v
```

### Test Results Example

```
============================= test session starts ==============================
collected 25 items

test_bfcl_analysis.py::TestStatisticalValidator::test_bootstrap_confidence_interval PASSED
test_bfcl_analysis.py::TestStatisticalValidator::test_performance_delta_calculation PASSED
test_bfcl_analysis.py::TestFamilyTaskAnalyzer::test_assign_model_family PASSED
test_bfcl_analysis.py::TestCaseStudySampler::test_extract_question PASSED
test_bfcl_analysis.py::TestIntegration::test_full_pipeline_integration PASSED
...

========================= 25 passed, 0 failed in 12.34s =========================
```

## 📈 Analysis Modules

### 1. Statistical Validation (`statistical_validation.py`)

**Purpose**: Rigorous statistical testing for performance comparisons

**Key Features**:
- Bootstrap confidence intervals with 1000+ iterations
- Chi-square and Fisher's exact tests for categorical data
- Sample-size weighted effect sizes
- Wilson score intervals for binomial proportions

**Usage Example**:
```python
from statistical_validation import StatisticalValidator

validator = StatisticalValidator(confidence_level=0.95)
result = validator.calculate_performance_delta_with_ci(
    strong_scores=[0.8, 0.85, 0.82], 
    weak_scores=[0.92, 0.94, 0.91]
)
print(f"Delta: {result['delta']:.2%}, P-value: {result['p_value']:.3f}")
```

### 2. Family-Task Analysis (`family_task_analysis.py`)

**Purpose**: Cross-analysis of model families and test categories

**Key Features**:
- Family-task performance matrix generation
- Systematic identification of strengths/weaknesses
- Task difficulty ranking with confidence intervals  
- Consistency scoring across test categories

**Usage Example**:
```python
from family_task_analysis import FamilyTaskAnalyzer

analyzer = FamilyTaskAnalyzer()
report = analyzer.generate_comprehensive_family_report(df)
print(f"Most consistent family: {report['summary']['most_consistent_family']}")
```

### 3. Case Study Sampling (`case_study_sampler.py`)

**Purpose**: Automated sampling and analysis of model outputs

**Key Features**:
- Random sampling of model outputs for comparison
- Pattern analysis (error types, output styles, function calls)
- Hypothesis generation for performance inversions
- Exportable case studies in JSON format

**Usage Example**:
```python
from case_study_sampler import CaseStudySampler

sampler = CaseStudySampler(data_root="/path/to/score/data")
case_studies = sampler.generate_case_studies_for_top_inversions(inversions_df)
```

### 4. Enhanced Visualizations (`enhanced_visualizations.py`)

**Purpose**: Accessible, publication-quality visualizations

**Key Features**:
- Colorblind-friendly palettes (Viridis, ColorBrewer)
- 95% confidence interval error bars
- Alt-text generation for screen readers
- High-DPI output (150+ DPI)

**Usage Example**:
```python
from enhanced_visualizations import AccessibleVisualizer

visualizer = AccessibleVisualizer(config)
visualizer.create_performance_inversion_plot(inversions_df, validation_df, output_dir)
visualizer.save_alt_texts(output_dir)  # Generate accessibility descriptions
```

## 🔍 Key Improvements Over Original Analysis

### Statistical Rigor
| Original | Enhanced |
|----------|----------|
| Fixed Δ > 0.1 threshold | Configurable thresholds with statistical testing |
| No confidence intervals | Bootstrap 95% CI for all metrics |
| No p-value calculation | Chi-square, Fisher's exact, t-tests |
| Equal sample weighting | Sample-size weighted deltas |

### Analysis Depth
| Original | Enhanced |
|----------|----------|
| Basic inversion detection | Family-task cross-analysis + case studies |
| Hardcoded model tiers | External YAML configuration |
| No output analysis | Automated sampling of actual model responses |
| Limited error classification | Technical vs. capability error separation |

### Reproducibility  
| Original | Enhanced |
|----------|----------|
| Local paths hardcoded | CLI arguments + Docker support |
| No testing | Comprehensive pytest suite (25+ tests) |
| Manual execution | Automated pipeline with error handling |
| No documentation | Detailed README + alt-text generation |

## 🐛 Troubleshooting

### Common Issues

**Issue**: `FileNotFoundError: Score directory not found`
```bash
# Solution: Verify data path
ls -la /path/to/score/directory
python bfcl_analysis_enhanced.py --data_root /correct/path --verbose
```

**Issue**: `ModuleNotFoundError: No module named 'statistical_validation'`  
```bash
# Solution: Add current directory to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python bfcl_analysis_enhanced.py --data_root ./data
```

**Issue**: Docker container exits immediately
```bash
# Solution: Check container logs
docker logs <container_id>
# Or run interactively
docker run -it --rm bfcl-analysis:v2.0 /bin/bash
```

### Performance Optimization

For large datasets (>100K evaluations):
- Use `--verbose` flag to monitor progress
- Increase Docker memory allocation (`docker run --memory=16g`)
- Consider reducing `bootstrap_iterations` in config for faster execution

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Run tests (`pytest test_bfcl_analysis.py -v`)
4. Commit changes (`git commit -m 'Add amazing feature'`)
5. Push to branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt
pip install black flake8 mypy sphinx

# Run quality checks
black . --check
flake8 . --max-line-length=100
mypy . --ignore-missing-imports

# Generate documentation
cd docs && make html
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Berkeley Function Calling Leaderboard team for the original benchmark
- Anthropic for Claude model evaluation data
- Statistical methods inspired by modern meta-analysis techniques
- Accessibility guidelines from WCAG 2.1 AA standards

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/your-repo/issues)
- **Documentation**: [Full Documentation](https://your-repo.github.io/docs)  
- **Contact**: [your-email@domain.com](mailto:your-email@domain.com)

---

*Enhanced BFCL Analysis Pipeline v2.0 - Bringing statistical rigor and accessibility to benchmark analysis*