import torch
import torch_tensorrt
from common.interpolator import Interpolator
import argparse
import os
import json


def get_inputs(dynamic_shape=False, half=False):
    if dynamic_shape:
        return [
            torch_tensorrt.Input(
                min_shape=(1, 3, 352, 352),
                opt_shape=(1, 3, 480, 832),
                max_shape=(1, 3, 1152, 1152),
                dtype=torch.float16 if half else torch.float32,
            ),  # x0
            torch_tensorrt.Input(
                min_shape=(1, 3, 352, 352),
                opt_shape=(1, 3, 480, 832),
                max_shape=(1, 3, 1152, 1152),
                dtype=torch.float16 if half else torch.float32,
            ),  # x1
            torch_tensorrt.Input((1, 1), dtype=torch.float16 if half else torch.float32),        # dt
        ]

    else:
        return [
            torch_tensorrt.Input((1, 3, 480, 832), dtype=torch.float16 if half else torch.float32),  # x0
            torch_tensorrt.Input((1, 3, 480, 832), dtype=torch.float16 if half else torch.float32),  # x1
            torch_tensorrt.Input((1, 1), dtype=torch.float16 if half else torch.float32),        # dt
        ]

def torch2trt(args):

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

    inputs = get_inputs(dynamic_shape=args.dynamic_shape, half=args.half)
    model = model.half() if args.half else model.float()

    print("Compiling model with TensorRT...")
    try:
        if args.dynamic_shape:
            # Use torch.export.Dim for proper dynamic shape handling
            from torch.export import Dim
            
            # Define dynamic dimensions with proper constraints
            dim_h = Dim("height", min=352, max=1152)
            dim_w = Dim("width", min=352, max=1152)
            
            # Create example inputs for tracing
            dtype = torch.float16 if args.half else torch.float32
            device = 'cuda'
            
            example_inputs = (
                torch.randn(1, 3, 512, 512, dtype=dtype, device=device),  # x0
                torch.randn(1, 3, 512, 512, dtype=dtype, device=device),  # x1
                torch.tensor([[0.5]], dtype=dtype, device=device),        # dt
            )
            
            # Define dynamic shapes for each input
            dynamic_shapes = {
                'x0': {2: dim_h, 3: dim_w},  # x0: height and width are dynamic
                'x1': {},  # x1: height and width are dynamic
                'batch_dt': {},                   # dt: static shape
            }
            
            # Export with dynamic shapes
            print("Exporting model with dynamic shapes...")
            exported_program = torch.export.export(
                model, 
                example_inputs, 
                dynamic_shapes=dynamic_shapes,
                strict=False
            )

            # Compile with TensorRT
            trt_gm = torch_tensorrt.dynamo.compile(
                exported_program,
                inputs=inputs,
            )
        else:
            # Use regular compilation for fixed shapes
            trt_gm = torch_tensorrt.compile(model, ir="dynamo", inputs=inputs)
        
        # Save both formats
        ep_file = f"{args.output_name}.ep"
        ts_file = f"{args.output_name}.ts"
        
        print(f"Saving ExportedProgram to: {ep_file}")
        torch_tensorrt.save(trt_gm, ep_file, inputs=inputs)
        
        print(f"Saving TorchScript to: {ts_file}")
        torch_tensorrt.save(trt_gm, ts_file, output_format="torchscript", inputs=inputs)
        
        print("TensorRT conversion completed successfully!")
        print(f"Generated files:")
        print(f"  - {ep_file} (ExportedProgram - for Python runtime)")
        print(f"  - {ts_file} (TorchScript - for C++ deployment)")
        
    except Exception as e:
        print(f"Error during TensorRT compilation: {e}")
        print("This might be due to unsupported operations or insufficient GPU memory")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert PyTorch model to TensorRT')
    parser.add_argument("--model_path", required=True, help="Path to the PyTorch model file (.pth)")
    parser.add_argument("--output_name", required=True, help="Prefix for output TensorRT files")
    parser.add_argument("--half", action="store_true", help="Use half precision (FP16)")
    parser.add_argument("--dynamic_shape", action="store_true", help="Use dynamic shape")
    args = parser.parse_args()

    print(f"Loading model from: {args.model_path}")
    print(f"Output prefix: {args.output_name}")
    print(f"Dynamic shape: {'Yes' if args.dynamic_shape else 'No'}")
    print(f"Precision: {'FP16' if args.half else 'FP32'}")
    torch2trt(args)
