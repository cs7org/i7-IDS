"""
flops_calculator.py
Requirements: pip install torch torchvision timm thop psutil
"""

import os
import time

import pandas as pd
import psutil
import torch
import torchvision
from thop import profile
from torch import nn

# ── Model data — all metrics start as None ─────────────────────────────────
MODELS_DATA = [
    {
        "Model": "LT-IMFEN",
        "Parameters (M)": 3.8,
        "Accuracy (%)": 76.7,
        "Year": 2024,
        "FLOPs (M)": None,
    },
    {
        "Model": "MnasNet",
        "Parameters (M)": 3.9,
        "Accuracy (%)": 75.2,
        "Year": 2018,
        "FLOPs (M)": None,
    },
    {
        "Model": "EdgeFormer",
        "Parameters (M)": 5.0,
        "Accuracy (%)": 78.6,
        "Year": 2022,
        "FLOPs (M)": None,
    },
    {
        "Model": "EfficientNet-B0",
        "Parameters (M)": 5.3,
        "Accuracy (%)": 77.1,
        "Year": 2019,
        "FLOPs (M)": None,
    },
    {
        "Model": "MobileNetV3-Large",
        "Parameters (M)": 5.4,
        "Accuracy (%)": 75.2,
        "Year": 2019,
        "FLOPs (M)": None,
    },
    {
        "Model": "MobileViT-S",
        "Parameters (M)": 5.6,
        "Accuracy (%)": 78.4,
        "Year": 2021,
        "FLOPs (M)": None,
    },
    {
        "Model": "TinyFormer",
        "Parameters (M)": 6.1,
        "Accuracy (%)": 80.1,
        "Year": 2023,
        "FLOPs (M)": None,
    },
    {
        "Model": "EfficientFormer-L1",
        "Parameters (M)": 12.3,
        "Accuracy (%)": 79.2,
        "Year": 2022,
        "FLOPs (M)": None,
    },
]

IN_CHANNEL = 1
NUM_CLASSES = 9
INPUT_HW = (138, 256)
BATCH_SIZES = [1, 16, 32, 64, 128, 256]
WARMUP_RUNS = 10
TIMED_RUNS = 50

NAME_TO_BACKBONE = {
    "LT-IMFEN": None,  # not in torchvision/timm
    "MnasNet": "mnasnet1_0",
    "EdgeFormer": None,  # not in torchvision/timm
    "EfficientNet-B0": "efficientnet_b0",
    "MobileNetV3-Large": "mobilenet_v3_large",
    "MobileViT-S": "mobilevit_s",
    "TinyFormer": "tiny_vit_5m_224",
    "EfficientFormer-L1": "efficientformer_l1",
}

TIMM_BACKBONES = {"mobilevit_s", "efficientformer_l1", "tiny_vit_5m_224", "tinyformer"}


# ── ImageClfModel ──────────────────────────────────────────────────────────
class ImageClfModel(nn.Module):
    def __init__(
        self, in_channel=1, num_classes=9, backbone="resnet18", pretrained=False
    ):
        super().__init__()
        arch = backbone.lower()
        self.pre_input = None
        # Only add 1->3 conv for EfficientFormer-L1, MobileViT-S, TinyFormer
        needs_1to3 = arch in {"efficientformer_l1", "mobilevit_s", "tiny_vit_5m_224", "tinyformer"}
        if in_channel == 1 and needs_1to3:
            self.pre_input = nn.Conv2d(1, 3, kernel_size=1)
            in_chans_for_backbone = 3
        else:
            in_chans_for_backbone = in_channel

        if arch in TIMM_BACKBONES:
            import timm
            self.backbone = timm.create_model(
                arch,
                pretrained=False,
                in_chans=in_chans_for_backbone,
                num_classes=num_classes,
            )
        else:
            self.backbone = getattr(torchvision.models, backbone)(weights=None)
            if in_channel != 3:
                if "mobilenet_v3" in arch or "efficientnet" in arch:
                    old = self.backbone.features[0][0]
                    self.backbone.features[0][0] = nn.Conv2d(
                        in_channel,
                        old.out_channels,
                        kernel_size=old.kernel_size,
                        stride=old.stride,
                        padding=old.padding,
                        bias=old.bias is not None,
                    )
                elif "mnasnet" in arch:
                    old = self.backbone.layers[0]
                    self.backbone.layers[0] = nn.Conv2d(
                        in_channel,
                        old.out_channels,
                        kernel_size=old.kernel_size,
                        stride=old.stride,
                        padding=old.padding,
                        bias=old.bias is not None,
                    )
            if "mobilenet_v3" in arch or "efficientnet" in arch:
                in_features = self.backbone.classifier[-1].in_features
                self.backbone.classifier[-1] = nn.Linear(in_features, num_classes)
            elif "mnasnet" in arch:
                self.backbone.classifier[1] = nn.Linear(
                    self.backbone.classifier[1].in_features, num_classes
                )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        if self.pre_input is not None:
            x = self.pre_input(x)
        logits = self.backbone(x)
        return logits, self.softmax(logits)


# ── Metrics ────────────────────────────────────────────────────────────────
def calc_flops(model, input_size, device):
    """Returns (MFLOPs, Params_M) or (None, None)."""
    try:
        model.eval()
        dummy = torch.randn(input_size).to(device)
        macs, params = profile(model, inputs=(dummy,), verbose=False)
        mflops = round((macs) / 1e6, 2)  # MFLOPs
        params_m = round(params / 1e6, 3)
        return mflops, params_m
    except Exception as e:
        print(f"    [!] FLOPs error: {e}")
        return None, None


def calc_fps(model, batch_size, device, in_channel=None, input_hw=None):
    """Returns FPS or None."""
    try:
        if in_channel is None:
            in_channel = IN_CHANNEL
        if input_hw is None:
            input_hw = INPUT_HW
        model.eval()
        dummy = torch.randn(batch_size, in_channel, *input_hw).to(device)
        with torch.no_grad():
            for _ in range(WARMUP_RUNS):
                _ = model(dummy)
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(TIMED_RUNS):
                _ = model(dummy)
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        return round((TIMED_RUNS * batch_size) / elapsed, 1)
    except Exception as e:
        print(f"    [!] FPS bs={batch_size} error: {e}")
        return None


def calc_model_size_gb(model):
    """Returns model weight size in GB."""
    try:
        total = sum(p.numel() * p.element_size() for p in model.parameters()) + sum(
            b.numel() * b.element_size() for b in model.buffers()
        )
        return round(total / 1e9, 6)
    except Exception as e:
        print(f"    [!] Model size error: {e}")
        return None


def calc_ram_gb(model, in_channel=None, input_hw=None):
    """Returns RAM delta (GB) after CPU forward pass."""
    try:
        if in_channel is None:
            in_channel = IN_CHANNEL
        if input_hw is None:
            input_hw = INPUT_HW
        process = psutil.Process(os.getpid())
        before = process.memory_info().rss
        dummy = torch.randn(1, in_channel, *input_hw)
        with torch.no_grad():
            _ = model(dummy)
        after = process.memory_info().rss
        return round((after - before) / 1e9, 4)
    except Exception as e:
        print(f"    [!] RAM error: {e}")
        return None


def calc_vram_gb(model, in_channel=None, input_hw=None):
    """Returns VRAM delta (GB) after GPU forward pass. None if no CUDA."""
    if not torch.cuda.is_available():
        return None
    try:
        if in_channel is None:
            in_channel = IN_CHANNEL
        if input_hw is None:
            input_hw = INPUT_HW
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
        model = model.cuda()
        dummy = torch.randn(1, in_channel, *input_hw).cuda()
        with torch.no_grad():
            _ = model(dummy)
        after = torch.cuda.memory_allocated()
        return round((after - before) / 1e9, 4)
    except Exception as e:
        print(f"    [!] VRAM error: {e}")
        return None


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Print GPU info if available
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
        print(f"[*] GPU: {gpu_name} with {gpu_mem} GB VRAM")
        import platform

        # Get full CPU name from /proc/cpuinfo (Linux/HPC)
        cpu_name = "Unknown"
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if "model name" in line:
                        cpu_name = line.strip().split(":")[1].strip()
                        break
        except Exception:
            cpu_name = platform.processor() or "Unknown"

        cpu_freq = psutil.cpu_freq()
        cpu_speed = f"{cpu_freq.max:.2f} MHz" if cpu_freq and cpu_freq.max else "Unknown"

        ram_gb = round(psutil.virtual_memory().total / 1e9, 2)
        print(f"[*] CPU    : {cpu_name}")
        print(f"[*] CPU MHz: {cpu_speed}")
        print(f"[*] RAM    : {ram_gb} GB")

    print(f"[*] Device : {device}")
    print(f"[*] Input  : (B x {IN_CHANNEL} x {INPUT_HW[0]} x {INPUT_HW[1]})")
    print(f"[*] Classes: {NUM_CLASSES}")
    print(f"[*] FPS batch sizes: {BATCH_SIZES}\n")

    df = pd.DataFrame(MODELS_DATA)
    df["Model Size (GB)"] = None
    df["RAM (GB)"] = None
    df["VRAM (GB)"] = None
    for bs in BATCH_SIZES:
        df[f"FPS_bs{bs}"] = None

    for i, row in df.iterrows():
        name = row["Model"]
        backbone = NAME_TO_BACKBONE.get(name)
        print(f"\n── {name} {'─' * (55 - len(name))}")

        if backbone is None:
            print("  not available → all metrics stay None")
            continue

        # EfficientFormer-L1: use (3, 224, 224)
        if name == "EfficientFormer-L1":
            in_channel = 3
            input_hw = (224, 224)
        else:
            in_channel = IN_CHANNEL
            input_hw = INPUT_HW

        # Build model
        try:
            model = ImageClfModel(
                in_channel=in_channel,
                num_classes=NUM_CLASSES,
                backbone=backbone,
                pretrained=False,
            ).to(device)
        except Exception as e:
            print(f"  [!] Model build failed: {e}")
            continue

        # FLOPs & Params
        mflops, params_m = calc_flops(model, (1, in_channel, *input_hw), device)
        if mflops is not None:
            df.at[i, "FLOPs (M)"] = mflops
            df.at[i, "Parameters (M)"] = params_m
        print(f"  FLOPs     : {mflops} MFLOPs    Params: {params_m} M")

        # Model size
        size_gb = calc_model_size_gb(model)
        df.at[i, "Model Size (GB)"] = size_gb
        print(f"  Model Size: {size_gb} GB")

        # RAM
        ram_gb = calc_ram_gb(model.cpu(), in_channel=in_channel, input_hw=input_hw)
        df.at[i, "RAM (GB)"] = ram_gb
        print(f"  RAM delta : {ram_gb} GB")

        # VRAM
        model = model.to(device)
        vram_gb = calc_vram_gb(model, in_channel=in_channel, input_hw=input_hw)
        df.at[i, "VRAM (GB)"] = vram_gb
        print(f"  VRAM delta: {vram_gb} GB")

        # FPS per batch size
        model = model.to(device)
        for bs in BATCH_SIZES:
            fps = calc_fps(model, bs, device, in_channel=in_channel, input_hw=input_hw)
            df.at[i, f"FPS_bs{bs}"] = fps
            print(f"  FPS bs={bs:<4}: {fps}")

    # ── Output ─────────────────────────────────────────────────────────────
    fps_cols = [f"FPS_bs{bs}" for bs in BATCH_SIZES]
    display_cols = [
        "Model",
        "Year",
        "Accuracy (%)",
        "Parameters (M)",
        "FLOPs (M)",
        "Model Size (GB)",
        "RAM (GB)",
        "VRAM (GB)",
    ] + fps_cols

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 250)
    pd.set_option("display.float_format", "{:.4f}".format)

    print("\n\n" + "═" * 150)
    print(df[display_cols].to_string(index=False))
    print("═" * 150)

    print("\n── Markdown ──────────────────────────────────────────────────────")
    print(df[display_cols].to_markdown(index=False))

    df[display_cols].to_csv("backbone_flops.csv", index=False)
    print("\n[✓] Saved → backbone_flops.csv")


if __name__ == "__main__":
    main()

