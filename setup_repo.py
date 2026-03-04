import os
from pathlib import Path
import json

def main():
    directories = [
        "configs",
        "data/raw",
        "data/processed",
        "notebooks",
        "src/data",
        "src/features",
        "src/mining",
        "src/models",
        "src/evaluation",
        "src/visualization",
        "scripts",
        "outputs/figures",
        "outputs/tables",
        "outputs/models",
        "outputs/reports"
    ]

    # Danh sách các file notebook mẫu theo yêu cầu (đánh số 01 -> 05)
    notebooks = [
        "01_data_exploration.ipynb",
        "02_data_preprocessing.ipynb",
        "03_feature_engineering.ipynb",
        "04_model_training.ipynb",
        "05_model_evaluation.ipynb"
    ]

    base_dir = Path(__file__).parent.absolute()
    
    print("Bắt đầu khởi tạo cấu trúc thư mục...")
    
    # 1. Tạo các thư mục
    for directory in directories:
        dir_path = base_dir / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"  [+] Đã tạo thư mục: {directory}/")

    # 2. Tạo file __init__.py trong thư mục src và các thư mục con để biến chúng thành Python module
    src_dirs = ["src"] + [f"src/{d}" for d in ["data", "features", "mining", "models", "evaluation", "visualization"]]
    for d in src_dirs:
        init_file = base_dir / d / "__init__.py"
        init_file.touch(exist_ok=True)
    print("  [+] Đã tạo các file __init__.py cho cấu trúc thư mục src/")

    # 3. Tạo các file notebook cơ bản
    notebooks_dir = base_dir / "notebooks"
    for nb in notebooks:
        nb_file = notebooks_dir / nb
        if not nb_file.exists():
            # Cấu trúc JSON trống cơ bản chuẩn của Jupyter Notebook
            empty_nb_content = {
                "cells": [],
                "metadata": {},
                "nbformat": 4,
                "nbformat_minor": 5
            }
            with open(nb_file, "w", encoding="utf-8") as f:
                json.dump(empty_nb_content, f)
            print(f"  [+] Đã tạo notebook: notebooks/{nb}")

if __name__ == "__main__":
    main()
    print("\n✅ Quá trình khởi tạo cấu trúc dự án hoàn tất!")
