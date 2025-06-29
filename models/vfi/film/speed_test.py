import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from contextlib import contextmanager
from typing import List, Tuple, Dict
import os
import argparse
import torch_tensorrt
import json
import glob

from common.interpolator import Interpolator
from common.utils import load_image


def _pad_batch(batch, align=64):
    """Pad the batch to be divisible by align.

    Args:
        batch: Input tensor [B, C, H, W]
        align: Alignment value

    Returns:
        Padded tensor [B, C, H, W] and crop region
    """
    # 1 c h w
    height, width = batch.shape[2:4]
    height_to_pad = (align - height % align) if height % align != 0 else 0
    width_to_pad = (align - width % align) if width % align != 0 else 0

    crop_region = [
        height_to_pad >> 1,
        width_to_pad >> 1,
        height + (height_to_pad >> 1),
        width + (width_to_pad >> 1),
    ]
    batch = F.pad(
        batch,
        (
            width_to_pad >> 1,
            width_to_pad - (width_to_pad >> 1),
            height_to_pad >> 1,
            height_to_pad - (height_to_pad >> 1),
            0,
            0,
            0,
            0,
        ),
        mode="constant",
    )
    return batch, crop_region

@contextmanager
def timer():
    """Context manager for timing code execution."""
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    print(f"Time: {end - start:.4f} seconds")


def load_target_resolutions(json_path: str) -> List[Tuple[int, int]]:
    """Load target resolutions from JSON file."""
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
        return [(res['width'], res['height']) for res in data['resolutions']]
    else:
        print(f"Resolution JSON file not found: {json_path}")
        print("Using default resolutions from available TensorRT models")
        return []


def find_tensorrt_models(trt_models_dir: str, height: int, width: int) -> Dict[str, str]:
    """Find TensorRT models for a specific resolution."""
    resolution_str = f"{height}x{width}"
    models = {}
    
    # Look for .ts and .ep files with the target resolution
    ts_pattern = os.path.join(trt_models_dir, f"film_dynamic_{resolution_str}.ts")
    ep_pattern = os.path.join(trt_models_dir, f"film_dynamic_{resolution_str}.ep")
    
    ts_files = glob.glob(ts_pattern)
    ep_files = glob.glob(ep_pattern)
    
    if ts_files:
        models['ts'] = ts_files[0]
    if ep_files:
        models['ep'] = ep_files[0]
    
    return models


def get_available_resolutions(trt_models_dir: str) -> List[Tuple[int, int]]:
    """Extract available resolutions from TensorRT model filenames."""
    resolutions = set()
    
    # Look for all film_dynamic_*.ts files
    pattern = os.path.join(trt_models_dir, "film_dynamic_*.ts")
    files = glob.glob(pattern)
    
    for file in files:
        basename = os.path.basename(file)
        # Extract resolution from filename like "film_dynamic_928x384.ts"
        if "film_dynamic_" in basename:
            resolution_part = basename.replace("film_dynamic_", "").replace(".ts", "")
            try:
                width, height = map(int, resolution_part.split('x'))
                resolutions.add((width, height))
            except ValueError:
                continue
    
    return sorted(list(resolutions))


def prepare_inputs(img1_path: str, img2_path: str, device: str = 'cuda', half: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare input tensors for inference."""
    img_batch_1, _ = load_image(img1_path)
    img_batch_2, _ = load_image(img2_path)
    
    img_batch_1 = torch.from_numpy(img_batch_1).permute(0, 3, 1, 2)
    img_batch_2 = torch.from_numpy(img_batch_2).permute(0, 3, 1, 2)
    
    if device == 'cuda':
        img_batch_1 = img_batch_1.cuda()
        img_batch_2 = img_batch_2.cuda()
    
    if half:
        img_batch_1 = img_batch_1.half()
        img_batch_2 = img_batch_2.half()
    
    dt = torch.tensor([[0.5]], device=device)
    if half:
        dt = dt.half()
    
    return img_batch_1, img_batch_2, dt


def prepare_inputs_fixed_size(img1_path: str, img2_path: str, target_size: Tuple[int, int] = (256, 256), device: str = 'cuda', half: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare input tensors with fixed size for ONNX inference."""
    img_batch_1, _ = load_image(img1_path)
    img_batch_2, _ = load_image(img2_path)
    
    # Resize to target size
    img_batch_1 = cv2.resize(img_batch_1[0], target_size[::-1])
    img_batch_2 = cv2.resize(img_batch_2[0], target_size[::-1])
    
    # Add batch dimension and convert to tensor
    img_batch_1 = torch.from_numpy(img_batch_1[None, ...]).permute(0, 3, 1, 2)
    img_batch_2 = torch.from_numpy(img_batch_2[None, ...]).permute(0, 3, 1, 2)
    
    if device == 'cuda':
        img_batch_1 = img_batch_1.cuda()
        img_batch_2 = img_batch_2.cuda()
    
    if half:
        img_batch_1 = img_batch_1.half()
        img_batch_2 = img_batch_2.half()
    
    dt = torch.tensor([[0.5]], device=device)
    if half:
        dt = dt.half()
    
    return img_batch_1, img_batch_2, dt

def test_pytorch_inference_compiled(img1_path: str, img2_path: str, compiled_model: nn.Module, warmup: int = 5, runs: int = 20, half: bool = False, target_size: Tuple[int, int] = None):
    """Test compiled PyTorch inference."""
    precision = "FP16" if half else "FP32"
    print("\n" + "="*50)
    print(f"Testing Compiled PyTorch Inference ({precision})")
    print("="*50)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping PyTorch test")
        return None, float('inf')

    if target_size is None:
        target_size = (256, 256)
    
    img1, img2, dt = prepare_inputs_fixed_size(img1_path, img2_path, target_size, device, half)

    # Warmup
    print(f"Warming up ({warmup} iterations)...")
    for _ in range(warmup):
        with torch.no_grad():
            _ = compiled_model(img1, img2, dt)
    
    if device == 'cuda':
        torch.cuda.synchronize()

    # Benchmark
    print(f"Running benchmark ({runs} iterations)...")
    times = []

    for i in range(runs):
        if device == 'cuda':
            torch.cuda.synchronize()

        start = time.perf_counter()
        with torch.no_grad():
            result = compiled_model(img1, img2, dt)

        if device == 'cuda':
            torch.cuda.synchronize()

        end = time.perf_counter()
        times.append(end - start)

        if i % 5 == 0:
            print(f"  Iteration {i+1}/{runs}: {times[-1]:.4f}s")

    avg_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    
    print(f"\nResults:")
    print(f"  Average time: {avg_time:.4f} ± {std_time:.4f} seconds")
    print(f"  Min time: {min_time:.4f} seconds")
    print(f"  Max time: {max_time:.4f} seconds")
    print(f"  FPS: {1/avg_time:.2f}")
    
    return result.shape, avg_time

def test_pytorch_inference(img1_path: str, img2_path: str, model_path: str, warmup: int = 5, runs: int = 20, half: bool = False, target_size: Tuple[int, int] = None):
    """Test pure PyTorch inference."""
    precision = "FP16" if half else "FP32"
    print("\n" + "="*50)
    print(f"Testing Pure PyTorch Inference ({precision})")
    print("="*50)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load model
    interpolator = Interpolator()
    state_dict = torch.load(model_path, map_location='cpu')
    interpolator.load_state_dict(state_dict)
    interpolator.eval()
    
    if device == 'cuda':
        interpolator = interpolator.cuda()
    
    if half and device == 'cuda':
        interpolator = interpolator.half()
    
    # Prepare inputs
    if target_size:
        img1, img2, dt = prepare_inputs_fixed_size(img1_path, img2_path, target_size, device, half)
    else:
        img1, img2, dt = prepare_inputs(img1_path, img2_path, device, half)
    
    # Warmup
    print(f"Warming up ({warmup} iterations)...")
    for _ in range(warmup):
        with torch.no_grad():
            _ = interpolator(img1, img2, dt)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    # Benchmark
    print(f"Running benchmark ({runs} iterations)...")
    times = []
    
    for i in range(runs):
        if device == 'cuda':
            torch.cuda.synchronize()
        
        start = time.perf_counter()
        with torch.no_grad():
            result = interpolator(img1, img2, dt)
        
        if device == 'cuda':
            torch.cuda.synchronize()
        
        end = time.perf_counter()
        times.append(end - start)
        
        if i % 5 == 0:
            print(f"  Iteration {i+1}/{runs}: {times[-1]:.4f}s")
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    
    print(f"\nResults:")
    print(f"  Average time: {avg_time:.4f} ± {std_time:.4f} seconds")
    print(f"  Min time: {min_time:.4f} seconds")
    print(f"  Max time: {max_time:.4f} seconds")
    print(f"  FPS: {1/avg_time:.2f}")
    
    return result.shape, avg_time


def test_jit_inference(img1_path: str, img2_path: str, jit_model: nn.Module, warmup: int = 5, runs: int = 20, half: bool = False, target_size: Tuple[int, int] = None):
    """Test JIT model inference."""
    precision = "FP16" if half else "FP32"
    print("\n" + "="*50)
    print(f"Testing JIT Model Inference ({precision})")
    print("="*50)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Prepare inputs - use target size if provided, otherwise default to 256x256
    if target_size is None:
        target_size = (256, 256)
    img1, img2, dt = prepare_inputs_fixed_size(img1_path, img2_path, target_size, device, half)
    # Warmup
    print(f"Warming up ({warmup} iterations)...")
    for _ in range(warmup):
        with torch.no_grad():
            _ = jit_model(img1, img2, dt)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    # Benchmark
    print(f"Running benchmark ({runs} iterations)...")
    times = []
    
    for i in range(runs):
        if device == 'cuda':
            torch.cuda.synchronize()
        
        start = time.perf_counter()
        with torch.no_grad():
            result = jit_model(img1, img2, dt)
        
        if device == 'cuda':
            torch.cuda.synchronize()
        
        end = time.perf_counter()
        times.append(end - start)
        
        if i % 5 == 0:
            print(f"  Iteration {i+1}/{runs}: {times[-1]:.4f}s")
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    
    print(f"\nResults:")
    print(f"  Average time: {avg_time:.4f} ± {std_time:.4f} seconds")
    print(f"  Min time: {min_time:.4f} seconds")
    print(f"  Max time: {max_time:.4f} seconds")
    print(f"  FPS: {1/avg_time:.2f}")
    
    return result.shape, avg_time


def test_tensorrt_inference(img1_path: str, img2_path: str, trt_path: str, warmup: int = 5, runs: int = 20, half: bool = False, target_size: Tuple[int, int] = None):
    """Test TensorRT inference."""
    precision = "FP16" if half else "FP32"
    print("\n" + "="*50)
    print(f"Testing TensorRT Inference ({precision})")
    print("="*50)
    
    device = 'cuda'
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping TensorRT test")
        return None, float('inf')
    
    try:
        
        # Load TensorRT model
        if trt_path.endswith('.ep'):
            model = torch.export.load(trt_path).module()
        else:  # .ts file
            model = torch.jit.load(trt_path, map_location='cuda')
                    
        if half:
            model = model.half()
        
        # Prepare inputs - use target size if provided, otherwise default to 256x256
        if target_size is None:
            target_size = (256, 256)
        
        img1, img2, dt = prepare_inputs_fixed_size(img1_path, img2_path, target_size, device, half)
        img1, _ = _pad_batch(img1, 64)
        img2, _ = _pad_batch(img2, 64)
        
        # Warmup
        print(f"Warming up ({warmup} iterations)...")
        for _ in range(warmup):
            with torch.no_grad():
                _ = model(img1, img2, dt)
        
        torch.cuda.synchronize()
        
        # Benchmark
        print(f"Running benchmark ({runs} iterations)...")
        times = []
        
        for i in range(runs):
            torch.cuda.synchronize()
            
            start = time.perf_counter()
            with torch.no_grad():
                result = model(img1, img2, dt)
            
            torch.cuda.synchronize()
            
            end = time.perf_counter()
            times.append(end - start)
            
            if i % 5 == 0:
                print(f"  Iteration {i+1}/{runs}: {times[-1]:.4f}s")
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)
        
        print(f"\nResults:")
        print(f"  Average time: {avg_time:.4f} ± {std_time:.4f} seconds")
        print(f"  Min time: {min_time:.4f} seconds")
        print(f"  Max time: {max_time:.4f} seconds")
        print(f"  FPS: {1/avg_time:.2f}")
        
        return result.shape, avg_time
        
    except Exception as e:
        print(f"Error running TensorRT inference: {e}")
        return None, float('inf')


def run_test(test_func, *args, **kwargs):
    """Run a test function with FP16 precision only."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping FP16 test")
        return float('inf')
    
    try:
        shape, time = test_func(*args, **kwargs)
        return time
    except Exception as e:
        print(f"FP16 test failed: {e}")
        return float('inf')


def get_inputs(resolution, half=False):
    return [
        torch.randn(1, 3, resolution[0], resolution[1], dtype=torch.float16 if half else torch.float32, device='cuda'),  # x0
        torch.randn(1, 3, resolution[0], resolution[1], dtype=torch.float16 if half else torch.float32, device='cuda'),  # x1
        torch.tensor([[0.5]], dtype=torch.float16 if half else torch.float32, device='cuda'),        # dt
    ]

def test_resolution_suite(img1_path: str, img2_path: str, target_size: Tuple[int, int], 
                         pytorch_model_path: str, jit_model: nn.Module, trt_models: Dict[str, str], 
                         pytorch_compiled_model: nn.Module,
                         warmup: int = 5, runs: int = 20) -> Dict[str, float]:
    """Test all available methods for a specific resolution in half precision."""
    height, width = target_size
    print(f"\n{'='*80}")
    print(f"TESTING RESOLUTION: {height}x{width} (FP16)")
    print(f"{'='*80}")
    
    results = {}
    
    # Test PyTorch FP16
    print(f"\n🔥 Testing PyTorch FP16 at {height}x{width}")
    time_taken = run_test(
        test_pytorch_inference, img1_path, img2_path, pytorch_model_path, warmup, runs, target_size=target_size, half=True
    )
    results['PyTorch'] = time_taken
    if time_taken != float('inf'):
        print(f"✅ PyTorch FP16: {time_taken:.4f}s ({1/time_taken:.2f} FPS)")
    else:
        print(f"❌ PyTorch FP16: Failed")
    
    # Test JIT FP16
    print(f"\n⚡ Testing JIT FP16 at {height}x{width}")
    time_taken = run_test(
        test_jit_inference, img1_path, img2_path, jit_model, warmup, runs, target_size=target_size, half=True
        )
    results['JIT'] = time_taken
    if time_taken != float('inf'):
        print(f"✅ JIT FP16: {time_taken:.4f}s ({1/time_taken:.2f} FPS)")
    else:
        print(f"❌ JIT FP16: Failed")
    
    # Test TensorRT FP32
    if trt_models:
        # Try .ep first, then .ts
        trt_model_path = trt_models.get('ep') or trt_models.get('ts')
        if trt_model_path:
            print(f"\n🚀 Testing TensorRT FP32 at {height}x{width}")
            print(f"   Using model: {os.path.basename(trt_model_path)}")
            time_taken = run_test(
                test_tensorrt_inference, img1_path, img2_path, trt_model_path, warmup, runs, target_size=target_size, half=False
            )
            results['TensorRT'] = time_taken
            if time_taken != float('inf'):
                print(f"✅ TensorRT FP32: {time_taken:.4f}s ({1/time_taken:.2f} FPS)")
            else:
                print(f"❌ TensorRT FP32: Failed")
        else:
            print(f"⚠️  No TensorRT models found for {width}x{height}")
            results['TensorRT'] = float('inf')
    else:
        print(f"⚠️  No TensorRT models found for {width}x{height}")
        results['TensorRT'] = float('inf')
    
    # Test PyTorch compiled FP16
    print(f"\n🔥 Testing PyTorch compiled FP16 at {height}x{width}")
    time_taken = run_test(
        test_pytorch_inference_compiled, img1_path, img2_path, pytorch_compiled_model, warmup, runs, target_size=target_size, half=True
    )
    results['PyTorch Compiled'] = time_taken
    if time_taken != float('inf'):
        print(f"✅ PyTorch Compiled FP16: {time_taken:.4f}s ({1/time_taken:.2f} FPS)")
    else:
        print(f"❌ PyTorch Compiled FP16: Failed")
    
    # Test ONNX FP16
    # Clean up GPU memory between tests
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return results


def print_resolution_summary(resolution_results: Dict[Tuple[int, int], Dict[str, float]]):
    """Print a comprehensive summary of results across all resolutions."""
    print(f"\n{'='*100}")
    print("COMPREHENSIVE RESOLUTION COMPARISON SUMMARY (FP16)")
    print(f"{'='*100}")
    
    # Find all methods tested
    all_methods = set()
    for results in resolution_results.values():
        all_methods.update(results.keys())
    all_methods = sorted(list(all_methods))
    
    # Create header
    header = f"{'Resolution':<12}"
    for method in all_methods:
        header += f"{method:<12}"
    header += f"{'Best':<12}{'Best FPS':<10}"
    print(header)
    print("-" * len(header))
    
    # Print results for each resolution
    overall_best = {}
    for resolution, results in sorted(resolution_results.items()):
        width, height = resolution
        row = f"{width}x{height:<4}"
        
        best_time = float('inf')
        best_method = "None"
        
        for method in all_methods:
            time_taken = results.get(method, float('inf'))
            if time_taken != float('inf'):
                row += f"{time_taken:<12.4f}"
                if time_taken < best_time:
                    best_time = time_taken
                    best_method = method
            else:
                row += f"{'FAILED':<12}"
        
        if best_time != float('inf'):
            row += f"{best_method:<12}{1/best_time:<10.2f}"
        else:
            row += f"{'None':<12}{'N/A':<10}"
        
        print(row)
        
        # Track overall best for each method
        for method, time_taken in results.items():
            if time_taken != float('inf'):
                if method not in overall_best:
                    overall_best[method] = []
                overall_best[method].append((resolution, time_taken))
    
    # Print method comparison summary
    print(f"\n{'='*100}")
    print("METHOD PERFORMANCE SUMMARY")
    print(f"{'='*100}")
    
    print(f"{'Method':<12}{'Avg Time (s)':<15}{'Avg FPS':<12}{'Best Time (s)':<15}{'Best FPS':<12}{'Success Rate':<12}")
    print("-" * 80)
    
    for method in all_methods:
        if method in overall_best:
            times = [time for _, time in overall_best[method]]
            avg_time = np.mean(times)
            best_time = min(times)
            success_rate = len(times) / len(resolution_results) * 100
            
            print(f"{method:<12}{avg_time:<15.4f}{1/avg_time:<12.2f}{best_time:<15.4f}{1/best_time:<12.2f}{success_rate:<12.1f}%")
        else:
            print(f"{method:<12}{'N/A':<15}{'N/A':<12}{'N/A':<15}{'N/A':<12}{'0.0%':<12}")
    
    # Find best method overall
    if overall_best:
        method_avg_times = {}
        for method, times_list in overall_best.items():
            times = [time for _, time in times_list]
            method_avg_times[method] = np.mean(times)
        
        best_overall_method = min(method_avg_times.items(), key=lambda x: x[1])
        print(f"\n🏆 Best Overall Method: {best_overall_method[0]} (Avg: {best_overall_method[1]:.4f}s, {1/best_overall_method[1]:.2f} FPS)")


def main():
    parser = argparse.ArgumentParser(description='Frame Interpolation Multi-Resolution Speed Test - FP16 Focus')
    
    parser.add_argument('--img1', type=str, default='photos/one.png', help='Path to first image')
    parser.add_argument('--img2', type=str, default='photos/two.png', help='Path to second image')
    parser.add_argument('--pytorch_model', type=str, default='pt_models/film.pt', help='Path to PyTorch model')
    parser.add_argument('--jit_model', type=str, default='pt_models/film_net.pt', help='Path to JIT model (base name)')
    parser.add_argument('--trt_models_dir', type=str, default='trt_models', help='Directory containing TensorRT models')
    parser.add_argument('--warmup', type=int, default=5, help='Number of warmup iterations')
    parser.add_argument('--runs', type=int, default=20, help='Number of benchmark iterations')
    parser.add_argument('--target_resolution_json', type=str, default='target_resolutions.json', help='Path to target resolution JSON file')
    parser.add_argument('--skip_pytorch', action='store_true', help='Skip PyTorch test')
    parser.add_argument('--skip_jit', action='store_true', help='Skip JIT test')
    parser.add_argument('--skip_trt', action='store_true', help='Skip TensorRT test')
    parser.add_argument('--max_resolutions', type=int, default=None, help='Maximum number of resolutions to test (for quick testing)')
    parser.add_argument('--specific_resolutions', type=str, nargs='*', help='Specific resolutions to test (e.g., 640x640 928x384)')
    
    args = parser.parse_args()
    
    print("Frame Interpolation Multi-Resolution Speed Test - FP16 Focus")
    print("="*70)
    print(f"Input images: {args.img1}, {args.img2}")
    print(f"PyTorch model: {args.pytorch_model}")
    print(f"JIT model base: {args.jit_model}")
    print(f"TensorRT models dir: {args.trt_models_dir}")
    print(f"Warmup iterations: {args.warmup}")
    print(f"Benchmark iterations: {args.runs}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠️  CUDA not available - tests will be limited")
        return
    
    # Load target resolutions
    target_resolutions = []
    
    if args.specific_resolutions:
        # Parse specific resolutions from command line
        for res_str in args.specific_resolutions:
            try:
                width, height = map(int, res_str.split('x'))
                target_resolutions.append((width, height))
            except ValueError:
                print(f"⚠️  Invalid resolution format: {res_str}")
        print(f"Using specific resolutions: {target_resolutions}")
    else:
        # Try to load from JSON file first
        target_resolutions = load_target_resolutions(args.target_resolution_json)
        
        # If JSON file doesn't exist or is empty, use available TensorRT models
        if not target_resolutions:
            target_resolutions = get_available_resolutions(args.trt_models_dir)
            print(f"Found {len(target_resolutions)} resolutions from TensorRT models")
    
    if not target_resolutions:
        print("❌ No target resolutions found. Please provide:")
        print("   1. A valid target_resolutions.json file, or")
        print("   2. Use --specific_resolutions flag, or")
        print("   3. Ensure TensorRT models are in the specified directory")
        return
    
    # Limit number of resolutions if specified
    if args.max_resolutions:
        target_resolutions = target_resolutions[:args.max_resolutions]
        print(f"Limited to first {args.max_resolutions} resolutions")
    
    print(f"Testing {len(target_resolutions)} resolutions: {target_resolutions}")
    
    # Skip flags
    skip_methods = []
    if args.skip_pytorch:
        skip_methods.append('PyTorch')
    if args.skip_jit:
        skip_methods.append('JIT')
    if args.skip_trt:
        skip_methods.append('TensorRT')
    
    if skip_methods:
        print(f"Skipping methods: {skip_methods}")
    
    # Run tests for each resolution
    resolution_results = {}

    # Load pytorch model
    pytorch_model = Interpolator()
    state_dict = torch.load(args.pytorch_model, map_location='cpu')
    pytorch_model.load_state_dict(state_dict)
    pytorch_model.eval()
    pytorch_model.to('cuda').half()

    # Get pytorch compiled model
    pytorch_compiled_model = torch.compile(pytorch_model)
    for i, resolution in enumerate(target_resolutions):
        height, width = resolution
        print(f"\n{'='*100}")
        print(f"Compiling model for resolution {i+1}/{len(target_resolutions)}: {height}x{width}")
        print(f"{'='*100}")
        dummy = get_inputs(resolution, half=True)
        pytorch_compiled_model.forward(*dummy)
        print(f"Pytorch compiled model for resolution {height}x{width}")

    base_name = args.jit_model.split('.')[0]
    jit_model = torch.jit.load(f"{base_name}_fp16.pt", map_location='cpu')
    jit_model.eval()
    jit_model.to('cuda').half()

    for i, resolution in enumerate(target_resolutions):
        height, width = resolution
        print(f"\n{'='*100}")
        print(f"RESOLUTION {i+1}/{len(target_resolutions)}: {height}x{width}")
        print(f"{'='*100}")
        
        trt_models = find_tensorrt_models(args.trt_models_dir, height, width)
        if trt_models:
            print(f"Found TensorRT models: {list(trt_models.keys())}")
        else:
            print(f"⚠️  No TensorRT models found for {height}x{width}")
        
        # Run the test suite for this resolution
        try:
            results = test_resolution_suite(
                args.img1, args.img2, resolution,
                args.pytorch_model, jit_model, trt_models, pytorch_compiled_model,
                args.warmup, args.runs
            )
            resolution_results[resolution] = results
            
            # Print quick summary for this resolution
            if results:
                best_method = min(results.items(), key=lambda x: x[1] if x[1] != float('inf') else float('inf'))
                if best_method[1] != float('inf'):
                    print(f"\n🏆 Best for {width}x{height}: {best_method[0]} ({best_method[1]:.4f}s, {1/best_method[1]:.2f} FPS)")
                else:
                    print(f"\n❌ All methods failed for {width}x{height}")
            
        except Exception as e:
            print(f"❌ Error testing resolution {width}x{height}: {e}")
            resolution_results[resolution] = {}
        
        # Clean up memory between resolutions
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Print comprehensive summary
    print_resolution_summary(resolution_results)


if __name__ == '__main__':
    main() 