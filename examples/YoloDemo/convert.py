"""
YOLO 模型转换脚本 - PyTorch (.pt) 转 ONNX

用法:
    python convert.py [--model model.pt] [--imgsz 640]

依赖:
    pip install ultralytics

转换后的 ONNX 文件将保存在原模型的同一目录下。
"""

import argparse
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    print("错误：需要安装 ultralytics 包")
    print("运行：pip install ultralytics")
    exit(1)


def convert_pt_to_onnx(model_path: str, imgsz: int = 640):
    """将 YOLO .pt 模型转换为 ONNX 格式"""

    model_file = Path(model_path)

    if not model_file.exists():
        print(f"错误：模型文件不存在：{model_file}")
        return None

    if model_file.suffix.lower() != ".pt":
        print(f"错误：文件不是 .pt 格式：{model_file}")
        return None

    print(f"加载模型：{model_file}")
    model = YOLO(str(model_file))

    output_dir = model_file.parent
    output_name = model_file.stem + ".onnx"
    output_path = output_dir / output_name

    print(f"转换为 ONNX 格式 (imgsz={imgsz})...")

    try:
        model.export(
            format="onnx",
            imgsz=imgsz,
            simplify=True,
            opset=12,
        )

        onnx_path = output_dir / model_file.stem
        if not onnx_path.suffix:
            onnx_path = onnx_path.with_suffix(".onnx")

        print(f"转换成功：{onnx_path}")
        return str(onnx_path)

    except Exception as e:
        print(f"转换失败：{e}")
        return None


def get_imgsz_interactive() -> int:
    """交互式获取 imgsz 参数"""
    print()
    print("可选输入尺寸:")
    print("  320  - 最快，精度略低")
    print("  640  - 默认，平衡 (推荐)")
    print("  1280 - 最高精度，较慢")
    print()

    while True:
        value = input("请输入 imgsz (默认: 640): ").strip()
        if value == "":
            return 640
        try:
            imgsz = int(value)
            if imgsz > 0:
                return imgsz
            print("请输入大于 0 的整数")
        except ValueError:
            print("请输入有效的整数")


def main():
    parser = argparse.ArgumentParser(description="YOLO .pt 转 ONNX 转换工具")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="要转换的 .pt 模型文件路径 (默认：扫描 ./models 目录)"
    )

    args = parser.parse_args()

    # 交互式获取 imgsz
    imgsz = get_imgsz_interactive()
    print(f"\n使用 imgsz = {imgsz}\n")

    if args.model:
        convert_pt_to_onnx(args.model, imgsz)
    else:
        models_dir = Path("./models")

        if not models_dir.exists():
            print("错误：./models 目录不存在")
            return

        pt_files = list(models_dir.glob("*.pt")) + list(models_dir.glob("*.PT"))

        if not pt_files:
            print("./models 目录中没有找到 .pt 文件")
            print("请将 .pt 模型文件放入 ./models 目录后重试")
            return

        print(f"找到 {len(pt_files)} 个 .pt 文件:")
        for i, pt_file in enumerate(pt_files, 1):
            print(f"  {i}. {pt_file.name}")
        print()

        for pt_file in pt_files:
            print(f"\n--- 转换: {pt_file.name} ---")
            convert_pt_to_onnx(str(pt_file), imgsz)

        print("\n所有转换完成！")

    print()
    input("按回车键退出...")


if __name__ == "__main__":
    main()
