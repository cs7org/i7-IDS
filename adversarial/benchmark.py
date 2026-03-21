import os
import time
import torch
import psutil
import platform
from pynvml import *
import thop  # Replace ptflops with thop
from pathlib import Path
from loguru import logger
import pandas as pd
import numpy as np


def benchmark_pytorch_model(model_path, input_size=(1, 3, 224, 224), batch_size=32, device="cuda", iterations=10):
    """
    Fully integrated benchmark: time, memory, energy, thermals, FLOPs, power, storage, CPU/GPU info.
    Uses thop instead of ptflops for FLOPs/MACs calculation.
    """

    assert os.path.isfile(model_path), f"Model not found: {model_path}"
    assert torch.cuda.is_available() if device == "cuda" else True

    # ───── Load Model ─────
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.eval().to(device)

    # ───── Parameter and FLOP Count with THOP ─────
    num_params = sum(p.numel() for p in model.parameters())
    num_params_million = round(num_params / 1e6, 3)
    
    # Use thop instead of ptflops
    dummy_input = torch.randn(1, *input_size[1:]).to(device)
    flops, params = thop.profile(model, inputs=(dummy_input,), verbose=False)
    flops_million = round(flops / 1e6, 3)
    params_million = round(params / 1e6, 3)

    # ───── Dummy Input for Benchmarking ─────
    input_shape = (batch_size,) + input_size[1:]
    benchmark_input = torch.randn(input_shape).to(device)

    # ───── Warm-up ─────
    with torch.no_grad():
        for _ in range(3):
            _ = model(benchmark_input)
    
    def calc_fps(model, batch_size, device, in_channel=None, input_hw=None):
        """Returns FPS or None."""

        if in_channel is None:
            in_channel = 1
        if input_hw is None:
            input_hw = (138, 256)
        model.eval()
        dummy = torch.randn(batch_size, in_channel, *input_hw).to(device)
        with torch.no_grad():
            for _ in range(5):
                _ = model(dummy)
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(50):
                _ = model(dummy)
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        return round((50 * batch_size) / elapsed, 1)

    avg_fps = calc_fps(model, batch_size, device, input_size[1], 
                       input_size[2:])
    # ───── CPU & RAM Stats ─────
    process = psutil.Process(os.getpid())
    ram_before = process.memory_info().rss / 1024**2
    total_ram = psutil.virtual_memory().total / 1024**2

    cpu_name = platform.processor()
    cpu_cores_total = psutil.cpu_count(logical=False)
    cpu_threads_total = psutil.cpu_count(logical=True)
    cpu_threads_used = len(process.threads())
    cpu_affinity_count = len(process.cpu_affinity())
    process.cpu_percent(interval=None)
    time.sleep(0.1)

    # ───── GPU Stats ─────
    torch.cuda.reset_peak_memory_stats()
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    gpu_name = nvmlDeviceGetName(handle)
    vram_total = nvmlDeviceGetMemoryInfo(handle).total / 1024**2
    energy_start = nvmlDeviceGetTotalEnergyConsumption(handle)
    temp_start = nvmlDeviceGetTemperature(handle, NVML_TEMPERATURE_GPU)
    power_readings = []

    # ───── Timing + Power Sampling ─────
    start_time = time.time()
    with torch.no_grad():
        for _ in range(iterations):
            power_draw = nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW → W
            power_readings.append(power_draw)
            _ = model(benchmark_input)
    end_time = time.time()

    energy_end = nvmlDeviceGetTotalEnergyConsumption(handle)
    temp_end = nvmlDeviceGetTemperature(handle, NVML_TEMPERATURE_GPU)
    nvmlShutdown()

    # ───── Metrics Calculation ─────
    total_samples = batch_size * iterations
    total_time = end_time - start_time
    ram_after = process.memory_info().rss / 1024**2
    vram_peak = torch.cuda.max_memory_allocated() / 1024**2

    return {
        # General
        "model_path": model_path,
        "batch_size": batch_size,
        "input_size": input_size,
        "model_size_MB": round(os.path.getsize(model_path) / 1024**2, 2),
        "avg_fps": avg_fps,

        # Parameters (from thop)
        "num_params": num_params,
        "num_params_million": num_params_million,
        "flops_million": flops_million,
        "thop_params_million": params_million,  # Additional from thop

        # Timing
        "iterations": iterations,
        "inference_time_per_sample_ms": round((total_time / total_samples) * 1000, 3),
        "throughput_samples_per_sec": round(total_samples / total_time, 2),

        # Memory
        "host_ram_MB": round(ram_after - ram_before, 2),
        "max_host_ram_MB": round(total_ram, 2),
        "gpu_vram_MB": round(vram_peak, 2),
        "max_gpu_vram_MB": round(vram_total, 2),

        # Energy & Thermal
        "energy_mJ": energy_end - energy_start,
        "avg_power_W": round(np.mean(power_readings), 2),
        "start_temp_C": temp_start,
        "end_temp_C": temp_end,

        # GPU
        "gpu_name": gpu_name,
        "device_name": torch.cuda.get_device_name(0) if device == "cuda" else "CPU",

        # CPU
        "cpu_name": cpu_name,
        "cpu_cores_total": cpu_cores_total,
        "cpu_threads_total": cpu_threads_total,
        "cpu_threads_used": cpu_threads_used,
        "cpu_affinity_count": cpu_affinity_count,
        "cpu_usage_percent": round(process.cpu_percent(interval=None), 2),
    }


if __name__ == "__main__":
    input_size = (1, 1, 138, 256) 
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    models = list(Path('results/image_classification').rglob('*best_model_full.pth'))
    models.extend(list(Path('results/autoencoder').rglob('*best_model_full.pth')))

    log_file = 'results/benchmark_results.log'
    logger.add(log_file, level="INFO")

    final_results=[]
    completed_models = set()
    for model_path in models:
        logger.info(f"Benchmarking model: {model_path}")
        if 'autoencoder' in str(model_path):
            model_type = "Autoencoder"
            model_name = 'RDU-Net' if 'rdunet' in str(model_path) else 'U-Net'
        else:
            model_type = "Classifier"
            model_name = 'ResNet18' if 'resnet18' in str(model_path) else 'MobileNetV3'
        if model_name not in completed_models:
            for batch_size in batch_sizes:
                logger.info(f"Running benchmark for {model_name} with batch size {batch_size}")
                try:
                    metrics = benchmark_pytorch_model(model_path, input_size, batch_size, device)
                    # insert model name at first
                    metrics['model_name'] = model_name
                    metrics['model_type'] = model_type
                    # append to final results
                    final_results.append([metrics])
                    logger.info(f"Results for {model_name}: {metrics}")
                    completed_models.add(model_name)
                except Exception as e:
                    logger.error(f"Error benchmarking {model_name}: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    continue
    
    # Save results to a CSV file
    df_data = pd.DataFrame([item for sublist in final_results for item in sublist])
    df_data.to_csv('results/benchmark_results.csv', index=False)
