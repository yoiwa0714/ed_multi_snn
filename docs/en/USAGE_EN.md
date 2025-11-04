# ED-Multi SNN Usage Guide

This guide provides comprehensive usage instructions for the ED-Multi SNN implementation with Spiking Neural Networks.

## Quick Start

### Basic Training Command
```bash
python ed_multi_snn.py --epochs 10 --viz
```

### With Specific Dataset
```bash
python ed_multi_snn.py --fashion --epochs 20 --batch 256 --hidden 512,256,128 --viz
```

## Command Line Options

### Dataset Selection
- `--mnist`: Use MNIST digit recognition dataset (default)
- `--fashion`: Use Fashion-MNIST clothing classification dataset
- `--cifar10`: Use CIFAR-10 image classification dataset
- `--cifar100`: Use CIFAR-100 image classification dataset

### Network Architecture
- `--hidden UNITS`: Hidden layer configuration
  - Single layer: `--hidden 512`
  - Multi-layer: `--hidden 512,256,128,64`
  - Default: `1024`

### Training Parameters
- `--epochs NUM`: Number of training epochs (default: 10)
- `--batch SIZE`: Mini-batch size (default: 128)
- `--lr RATE`: Learning rate (default: 0.01)
- `--u1 VALUE`: Amine diffusion coefficient (default: 0.1)

### SNN Parameters
- `--encoding TYPE`: Spike encoding method
  - `poisson`: Poisson spike encoding
  - `rate`: Rate-based encoding
  - `temporal`: Temporal pattern encoding
  - Default: `poisson`
- `--time_steps NUM`: Number of simulation time steps (default: 100)
- `--threshold VALUE`: LIF neuron firing threshold (default: 1.0)
- `--tau_mem VALUE`: Membrane time constant (default: 20.0)
- `--tau_syn VALUE`: Synaptic time constant (default: 5.0)

### Visualization Options
- `--viz`: Enable real-time learning visualization
- `--heatmap`: Enable neuron firing heatmap visualization
- `--save_plots`: Save visualization plots to disk

### Performance Options
- `--gpu`: Enable GPU acceleration (requires CUDA)
- `--verbose`: Enable detailed profiling output
- `--workers NUM`: Number of data loading workers (default: 4)

## Usage Examples

### Example 1: Basic MNIST Training
```bash
python ed_multi_snn.py --mnist --epochs 30 --viz
```
- Dataset: MNIST digit recognition
- Epochs: 30
- Visualization: Enabled
- Other parameters: Default values

### Example 2: Fashion-MNIST with Multi-layer Network
```bash
python ed_multi_snn.py --fashion --hidden 1024,512,256 --epochs 50 --batch 256 --viz
```
- Dataset: Fashion-MNIST
- Architecture: 3 hidden layers (1024→512→256)
- Training: 50 epochs with batch size 256
- Visualization: Enabled

### Example 3: CIFAR-10 with GPU Acceleration
```bash
python ed_multi_snn.py --cifar10 --hidden 2048,1024,512,256 --epochs 100 --batch 512 --gpu --viz --heatmap
```
- Dataset: CIFAR-10 color images
- Architecture: 4 hidden layers
- Training: 100 epochs with large batch size
- Acceleration: GPU enabled
- Visualization: Learning curves + neuron heatmaps

### Example 4: Custom SNN Parameters
```bash
python ed_multi_snn.py --fashion --encoding temporal --time_steps 200 --threshold 0.8 --tau_mem 15.0 --epochs 40 --viz
```
- Dataset: Fashion-MNIST
- Encoding: Temporal spike patterns
- SNN: 200 time steps, lower threshold, faster membrane dynamics
- Training: 40 epochs with visualization

### Example 5: High-Performance Training
```bash
python ed_multi_snn.py --cifar100 --hidden 4096,2048,1024,512,256 --epochs 200 --batch 1024 --gpu --workers 8 --verbose
```
- Dataset: CIFAR-100 (100 classes)
- Architecture: Deep 5-layer network
- Training: Long training with large batch
- Performance: GPU + multi-worker data loading + profiling

## Advanced Configuration

### Custom Learning Parameters
```bash
python ed_multi_snn.py --lr 0.005 --u1 0.15 --momentum 0.9 --weight_decay 1e-4
```

### Detailed Monitoring
```bash
python ed_multi_snn.py --verbose --save_plots --log_interval 10
```

### Experiment Mode
```bash
python ed_multi_snn.py --experiment_name "deep_fashion_snn" --save_model --tensorboard
```

## Output Interpretation

### Learning Progress Display
- **Epoch Progress**: Shows completion percentage and ETA
- **Loss Values**: Training and validation loss
- **Accuracy**: Classification accuracy percentage
- **SNN Metrics**: Spike rate, membrane potential statistics

### Visualization Components
- **Learning Curves**: Real-time accuracy and loss progression
- **Confusion Matrix**: Classification performance by class
- **Spike Heatmaps**: Neuron firing patterns across layers
- **Weight Histograms**: Weight distribution evolution

### Performance Metrics
- **Processing Speed**: Samples per second
- **Memory Usage**: RAM and GPU memory consumption
- **Convergence**: Learning stability indicators

## Troubleshooting

### Common Issues

#### Memory Errors
```
CUDA out of memory
```
**Solution**: Reduce batch size or hidden layer sizes
```bash
python ed_multi_snn.py --batch 64 --hidden 256,128
```

#### Slow Training
**Problem**: Training takes too long
**Solution**: Enable GPU or reduce network complexity
```bash
python ed_multi_snn.py --gpu --hidden 512,256
```

#### Visualization Issues
**Problem**: Plots not displaying
**Solution**: Check display settings and install required packages
```bash
pip install matplotlib seaborn
export DISPLAY=:0  # For Linux
```

### Performance Optimization

#### For Large Datasets
- Use larger batch sizes (256-1024)
- Enable GPU acceleration
- Increase worker threads
- Consider distributed training

#### For Quick Experiments
- Use smaller networks (--hidden 256,128)
- Reduce epochs (--epochs 5)
- Disable expensive visualizations

#### For Production
- Save models (--save_model)
- Enable logging (--verbose)
- Use configuration files for reproducibility

## Configuration Files

### Example config.yaml
```yaml
dataset: fashion
hidden_layers: [1024, 512, 256]
epochs: 50
batch_size: 256
learning_rate: 0.01
snn_parameters:
  encoding: poisson
  time_steps: 100
  threshold: 1.0
visualization:
  enabled: true
  heatmap: true
  save_plots: true
```

### Loading Configuration
```bash
python ed_multi_snn.py --config config.yaml
```

## Integration with Other Tools

### Jupyter Notebook Usage
```python
from ed_multi_snn import EDMultiSNN
model = EDMultiSNN(hidden=[512, 256], encoding='poisson')
model.train(dataset='fashion', epochs=20, visualization=True)
```

### Hyperparameter Optimization
```bash
# Using Optuna or similar tools
python optimize_hyperparams.py --trials 100 --study_name ed_snn_optimization
```

## Best Practices

1. **Start Simple**: Begin with small networks and short training
2. **Monitor Progress**: Always use --viz for initial experiments
3. **Save Results**: Use --save_model for good configurations
4. **GPU Usage**: Enable GPU for networks with >1000 hidden units
5. **Batch Size**: Start with 128, increase for better GPU utilization
6. **Learning Rate**: Begin with 0.01, adjust based on convergence

## Support and Documentation

- **Technical Specification**: `docs/en/ed_multi_snn.prompt_EN.md`
- **API Documentation**: `docs/en/API_Reference.md`
- **Theory Background**: `docs/en/ED_Method_Overview.md`
- **SNN Fundamentals**: `docs/en/SNN_Overview.md`

For additional help, refer to the comprehensive documentation in the `docs/en/` directory.