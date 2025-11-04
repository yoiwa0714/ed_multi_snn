# ED-Multi SNN (English Commented Version)

This `src/en/` directory contains the ED-Multi SNN implementation with English comments.

## Usage

```bash
# Navigate to this directory
cd src/en

# Basic training with MNIST dataset
python ed_multi_lif_snn.py --mnist --epochs 10 --viz

# Training with Fashion-MNIST dataset (with visualization)
python ed_multi_lif_snn.py --fashion --epochs 20 --viz --heatmap

# Multi-layer network training
python ed_multi_lif_snn.py --mnist --hidden 512,256,128 --epochs 30 --viz
```

## Available Files

- `ed_multi_lif_snn.py` - Complete LIF version (recommended)
- `ed_multi_lif_snn_simple.py` - Simple version (for learning)
- `ed_multi_frelu_snn.py` - FReLU version (experimental)
- `modules/` - Common modules

## Detailed Information

- Detailed usage: [../../docs/en/USAGE_EN.md](../../docs/en/USAGE_EN.md)
- Technical specification: [../../docs/en/](../../docs/en/)
- Project overview: [../../README_EN.md](../../README_EN.md)

## Japanese Version

日本語コメント版は`../ja/`ディレクトリをご利用ください。