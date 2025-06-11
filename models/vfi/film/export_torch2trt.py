import torch
import torch_tensorrt
from common.interpolator import Interpolator
import argparse
import os
import json


def get_inputs(resolution, half=False):
    return [
        torch.randn(1, 3, resolution[0], resolution[1], dtype=torch.float16 if half else torch.float32, device='cuda'),  # x0
        torch.randn(1, 3, resolution[0], resolution[1], dtype=torch.float16 if half else torch.float32, device='cuda'),  # x1
        torch.tensor([[0.5]], dtype=torch.float16 if half else torch.float32, device='cuda'),        # dt
    ]

def tensorrt_compile(model, half=False, output_dir=None, output_name=None, target_resolution=None):
    try:
        if target_resolution is not None:
            for resolution in target_resolution:
                print(f"Compiling model for resolution {resolution[0]}x{resolution[1]}")
                inputs = get_inputs(resolution, half=half)
                # Compile with TensorRT
                trt_gm = torch_tensorrt.compile(
                    model,
                    inputs=inputs,
                )
                # Save both formats
                ep_file = f"{args.output_dir}/{args.output_name}_{resolution[0]}x{resolution[1]}.ep"
                ts_file = f"{args.output_dir}/{args.output_name}_{resolution[0]}x{resolution[1]}.ts"
                
                print(f"Saving ExportedProgram to: {ep_file}")
                torch_tensorrt.save(trt_gm, ep_file, inputs=inputs)
                
                print(f"Saving TorchScript to: {ts_file}")
                torch_tensorrt.save(trt_gm, ts_file, output_format="torchscript", inputs=inputs)
        else:
            print(f"Compiling model for resolution {target_resolution[0]}x{target_resolution[1]}")
            inputs = get_inputs(target_resolution, half=args.half)
            trt_gm = torch_tensorrt.compile(model, ir="dynamo", inputs=inputs)
        
        print("TensorRT conversion completed successfully!")
        
    except Exception as e:
        print(f"Error during TensorRT compilation: {e}")
        print("This might be due to unsupported operations or insufficient GPU memory")
        raise

def torch2trt(args):
    if args.target_resolution is not None:
        with open(args.target_resolution, 'r') as f:
            target_resolution = json.load(f)
        
        print(f"Number of target resolutions: {len(target_resolution)}")
        for i, resolution in enumerate(target_resolution):
            print(f"Target resolution {i}: {resolution[0]}x{resolution[1]}")
    else:
        target_resolution = [[256, 256]]
        print("No target resolution provided, using default resolution 256x256")
    # Load model
    interpolator = Interpolator()

    if args.model_path.endswith('.pt'):
        # Load state dict
        state_dict = torch.load(args.model_path, map_location='cpu')
        interpolator.load_state_dict(state_dict)
    else:
        raise ValueError(f"Unsupported model format: {args.model_path}")

    interpolator.eval()
    model = interpolator.cuda()

    if args.full_precision:
        model = model.float()
        tensorrt_compile(model, half=False, output_dir=args.output_dir, output_name=args.output_name, target_resolution=target_resolution)
    
    if args.half:
        model = model.half()
        tensorrt_compile(model, half=args.half, output_dir=args.output_dir, output_name=args.output_name, target_resolution=target_resolution)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert PyTorch model to TensorRT')
    parser.add_argument("--model_path", required=True, help="Path to the PyTorch model file (.pth)")
    parser.add_argument("--output_dir", required=True, help="Path to the output directory")
    parser.add_argument("--output_name", required=True, help="Prefix for output TensorRT files")
    parser.add_argument("--half", action="store_true", help="Use half precision (FP16)")
    parser.add_argument("--full_precision", action="store_true", help="Use full precision (FP32)")
    parser.add_argument("--target_resolution", required=True, help="Path to the target resolution file (.json)")
    args = parser.parse_args()

    print(f"Loading model from: {args.model_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Output prefix: {args.output_name}")
    print(f"Target resolution: {args.target_resolution}")
    print(f"Half precision: {args.half}")
    print(f"Full precision: {args.full_precision}")

    os.makedirs(args.output_dir, exist_ok=True)
    torch2trt(args)
