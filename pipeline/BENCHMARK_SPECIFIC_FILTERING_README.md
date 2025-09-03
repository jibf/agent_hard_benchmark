# Benchmark-Specific Rule-Based Filtering

This system allows you to use either **general comprehensive filtering** (default) or **benchmark-specific filtering** for Step 1 of the pipeline.

## 🎯 Overview

### **Default Behavior (General Filtering)**
- Uses the comprehensive rule-based filtering that works across all benchmarks
- Applies the same discriminativeness criteria to all questions
- Good for general-purpose filtering

### **Benchmark-Specific Filtering**
- Each benchmark can implement its own custom filtering rules
- More targeted and potentially more effective for specific benchmark characteristics
- Allows for benchmark-specific optimizations

## 🚀 Usage

### **1. General Filtering (Default)**
```bash
python main.py --skip-llm-judge
```

### **2. Benchmark-Specific Filtering**
```bash
python main.py --skip-llm-judge --specific-step1
```

### **3. Target Specific Benchmark**
```bash
python main.py --skip-llm-judge --specific-step1 --target_benchmark DrafterBench
```

## 📁 File Structure

```
src/
├── comprehensive_rule_filtering.py          # General filtering (existing)
├── rule_filtering_orchestrator.py           # New orchestrator
├── benchmark_specific_filters/              # New directory
│   ├── __init__.py
│   ├── base_filter.py                       # Base class
│   ├── drafter_bench_filter.py              # DrafterBench rules
│   ├── complex_func_bench_filter.py         # ComplexFuncBench rules
│   ├── bfcl_filter.py                       # BFCL rules
│   ├── nexus_bench_filter.py                # NexusBench rules
│   └── tau_bench_filter.py                  # TAU Bench rules
```

## 🔧 How It Works

### **1. Orchestrator Decision**
The `RuleFilteringOrchestrator` decides which filtering approach to use:
- `use_specific_filters=False` → Use general filtering
- `use_specific_filters=True` → Use benchmark-specific filtering

### **2. Benchmark Detection**
The system automatically detects which benchmark each sample belongs to based on:
- File path patterns
- Task name patterns
- Sample structure

### **3. Filter Application**
- **Known benchmarks**: Use their specific filter
- **Unknown benchmarks**: Fall back to general filtering

## 📝 Implementing Custom Filters

### **1. Create Your Filter Class**
```python
from .base_filter import BaseBenchmarkFilter

class MyBenchmarkFilter(BaseBenchmarkFilter):
    def __init__(self):
        super().__init__("MyBenchmark")
    
    def get_filter_name(self) -> str:
        return "MyBenchmark-Specific Filter"
    
    def filter_samples(self, samples: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        # Implement your custom filtering logic here
        # Return (passed_samples, dropped_samples)
        pass
```

### **2. Register Your Filter**
Add it to the orchestrator in `rule_filtering_orchestrator.py`:
```python
self.benchmark_filters = {
    'MyBenchmark': MyBenchmarkFilter(),
    # ... other filters
}
```

### **3. Update Benchmark Detection**
Modify the `_extract_benchmark_name` method to recognize your benchmark.

## 🎯 Current Implementations

### **DrafterBench Filter** ✅
- **Structure validation**: Checks required fields and message format
- **Score sanity**: Validates 0-100 scale scores
- **Discriminativeness**: Question-level variance analysis

### **Other Benchmarks** 🔄
- Currently use general filtering as fallback
- Ready for custom implementation

## 🔍 Example: DrafterBench vs General

### **General Filter**
- Applies same logic to all benchmarks
- Uses variance > 0.01 threshold
- Good for cross-benchmark consistency

### **DrafterBench Filter**
- Validates DrafterBench-specific structure
- Checks 0-100 score scale
- Uses same discriminativeness logic but with benchmark-specific validation

## 🚀 Benefits

1. **Flexibility**: Choose between general and specific filtering
2. **Customization**: Each benchmark can optimize for its characteristics
3. **Fallback Safety**: Unknown benchmarks use general filtering
4. **Easy Extension**: Simple to add new benchmark-specific filters
5. **Consistent Interface**: All filters follow the same API

## 🧪 Testing

Test the system:
```bash
python test_specific_filtering.py
```

## 🔮 Future Enhancements

1. **More Benchmark Filters**: Implement custom logic for other benchmarks
2. **Hybrid Approaches**: Combine general and specific filtering
3. **Dynamic Thresholds**: Adjust thresholds based on benchmark characteristics
4. **Performance Metrics**: Track filtering effectiveness per benchmark

