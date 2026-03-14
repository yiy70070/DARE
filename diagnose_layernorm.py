import json
import numpy as np
from pathlib import Path

def find_json_files(operator_name, dirs):
    """查找文件"""
    filename = 'layernorm'  # LayerNorm的文件名
    json_files = []
    
    for dir_path in dirs:
        p = Path(dir_path)
        if not p.exists():
            continue
        pattern = f"*{filename}*.json"
        found = list(p.glob(pattern))
        json_files.extend(found)
    
    return json_files

def extract_metrics(data):
    """提取metrics（带过滤）"""
    if data is None or 'detector' not in data:
        return None
    
    detector = data['detector']
    actual_err = detector.get('actual_err', 0)
    predicted_bound = detector.get('predicted_bound', 0)
    
    # 🔧 这里是你当前的过滤逻辑
    if actual_err <= 0:
        return None
    
    if predicted_bound <= 0:
        return None
    
    if actual_err < 1e-10:  # 如果你有这个过滤
        return None
    
    coverage = predicted_bound >= actual_err
    tightness = predicted_bound / actual_err
    
    if not np.isfinite(tightness):
        return None
    
    if tightness > 100:  # 如果你有这个过滤
        return None
    
    return {
        'tightness': tightness,
        'coverage': coverage,
        'time': detector.get('elapsed_sec', 0)
    }

def diagnose_per_config():
    """检查每个配置的LayerNorm数据"""
    
    configs = {
        'Storage-Reduced': [
            'evaluation/experiment',
            'evaluation/experiment_1'
        ],
        'FP32-Uniform': [
            'evaluation/experiment_fp32'
        ],
        'FP16-Uniform': [
            'evaluation/experiment_fp16_all'
        ],
        'FP16-Comp-FP32-Acc': [
            'evaluation/experiment_fp16_compute_fp32_accum',
            'evaluation/experiment_fp16_compute_fp_32_accum'
        ],
        'BF16-Compute': [
            'evaluation/experiment_bf16'
        ]
    }
    
    print("=" * 70)
    print("LayerNorm Data Per Configuration")
    print("=" * 70)
    
    all_tightness = []
    
    for config_name, config_dirs in configs.items():
        print(f"\n{config_name}:")
        
        json_files = find_json_files('layernorm', config_dirs)
        print(f"  Total files found: {len(json_files)}")
        
        if len(json_files) == 0:
            print(f"  ❌ No files found!")
            continue
        
        # 提取metrics
        metrics_list = []
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                metrics = extract_metrics(data)
                if metrics:
                    metrics_list.append(metrics)
            except Exception as e:
                pass
        
        print(f"  Valid after filtering: {len(metrics_list)}")
        
        if len(metrics_list) == 0:
            print(f"  ⚠️  All data filtered out!")
            continue
        
        tightness_values = [m['tightness'] for m in metrics_list]
        coverage_values = [m['coverage'] for m in metrics_list]
        
        median_tight = np.median(tightness_values)
        p95_tight = np.percentile(tightness_values, 95)
        coverage_rate = sum(coverage_values) / len(coverage_values) * 100
        
        print(f"  Coverage: {coverage_rate:.1f}%")
        print(f"  Median tightness: {median_tight:.2f}×")
        print(f"  95th percentile: {p95_tight:.2f}×")
        
        all_tightness.extend(tightness_values)
    
    # 汇总
    print("\n" + "=" * 70)
    print("OVERALL (All Configs)")
    print("=" * 70)
    
    if len(all_tightness) == 0:
        print("❌ No valid data across all configs!")
        print("\nPossible reasons:")
        print("1. Files not found (wrong directory or filename)")
        print("2. All data filtered out (check extract_metrics logic)")
    else:
        print(f"Total valid cases: {len(all_tightness)}")
        print(f"Median tightness: {np.median(all_tightness):.2f}×")
        print(f"95th percentile: {np.percentile(all_tightness, 95):.2f}×")
        print(f"Min: {np.min(all_tightness):.2f}×")
        print(f"Max: {np.max(all_tightness):.2f}×")
        print("\n✅ Data should NOT be NaN!")

if __name__ == "__main__":
    diagnose_per_config()