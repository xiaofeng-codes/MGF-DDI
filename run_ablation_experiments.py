"""
消融实验运行脚本
- 二分类变体: zhang, miner 数据集, fold 0
- 多分类变体: deng 数据集, fold 0
"""
import subprocess
import sys
import os

# 二分类变体列表
BINARY_VARIANTS = [
    "v1_wo_unimol",
    "v2_wo_fuzzy",
    "v3_wo_cl",
    "v4_wo_fusion",
    "v5_wo_edge",
    "v6_wo_interact",
]

# 多分类变体列表
MULTICLASS_VARIANTS = [
    "v1_wo_unimol",
    "v2_wo_fuzzy",
    "v3_wo_cl",
    "v4_wo_fusion",
    "v5_wo_edge",
    "v6_wo_interact",
]

def run_binary_experiments():
    """运行二分类消融实验"""
    print("=" * 60)
    print("开始运行二分类消融实验")
    print("=" * 60)
    
    datasets = ["zhang", "miner"]
    
    for variant in BINARY_VARIANTS:
        for dataset in datasets:

            fold = "0"
            
            script_path = f"ablation_experiments/{variant}/train.py"
            
            print(f"\n>>> 运行 [二分类] {variant} | 数据集: {dataset} | Fold: {fold}")
            print("-" * 50)
            
            subprocess.run([
                sys.executable, 
                script_path,
                "--dataset", dataset,
                "--fold", fold
            ])


def run_multiclass_experiments():
    """运行多分类消融实验"""
    print("=" * 60)
    print("开始运行多分类消融实验")
    print("=" * 60)
    
    dataset = "deng"
    fold = "0"
    
    for variant in MULTICLASS_VARIANTS:
        script_path = f"ablation_experiments/multiclass/{variant}/train.py"
        
        print(f"\n>>> 运行 [多分类] {variant} | 数据集: {dataset} | Fold: {fold}")
        print("-" * 50)
        
        subprocess.run([
            sys.executable,
            script_path,
            "--dataset", dataset,
            "--fold", fold
        ])


def main():
    print("=" * 60)
    print("MVI-DDI 消融实验运行脚本")
    print("=" * 60)
    print("\n配置:")
    print("  - 二分类: zhang, miner (fold 0)")
    print("  - 多分类: deng (fold 0)")
    print()
    
    # 运行二分类实验
    run_binary_experiments()
    
    # 运行多分类实验
    run_multiclass_experiments()
    
    print("\n" + "=" * 60)
    print("所有消融实验完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()

