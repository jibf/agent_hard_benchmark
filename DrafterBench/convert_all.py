#!/usr/bin/env python3

import os
import glob
import subprocess
from pathlib import Path


def main():
    # 获取脚本所在目录
    script_dir = Path(__file__).parent
    latest_results_dir = script_dir / "latest_results"

    # 查找所有 *_All_score.json 文件
    pattern = str(latest_results_dir / "*" / "*_All_score.json")
    score_files = glob.glob(pattern)

    if not score_files:
        print(f"未找到任何 *_All_score.json 文件在 {latest_results_dir}")
        return

    print(f"找到 {len(score_files)} 个结果文件\n")

    for score_file in sorted(score_files):
        score_path = Path(score_file)

        # 从路径中提取文件夹名（即model_name）
        folder_name = score_path.parent.name

        # 将文件夹名中的 "_" 替换为 "/"
        model_path = folder_name.replace("_", "/")

        # 构建命令
        cmd = [
            "python",
            "convert_drafterbench_results.py",
            str(score_path.relative_to(script_dir)),
            "fixed_converted_results",
            "--model-path",
            model_path
        ]

        print(f"处理: {folder_name}")
        print(f"文件: {score_path.name}")
        print(f"模型: {model_path}")
        print(f"命令: {' '.join(cmd)}")
        print("-" * 80)

        # 执行命令
        try:
            result = subprocess.run(
                cmd,
                cwd=script_dir,
                check=True,
                capture_output=True,
                text=True
            )
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(result.stderr)
            print(f"✓ 成功转换: {model_path}\n")
        except subprocess.CalledProcessError as e:
            print(f"✗ 转换失败: {model_path}")
            print(f"错误: {e}")
            if e.stdout:
                print(f"stdout: {e.stdout}")
            if e.stderr:
                print(f"stderr: {e.stderr}")
            print()
            continue


if __name__ == "__main__":
    main()
