# ED Method - A Novel Learning Approach for Spiking Neural Networks (SNN)

[日本語](README.md) | **English**

Implementation of Isamu Kaneko's original Error-Diffusion (ED) method applied to Spiking Neural Networks (SNN).<br>
This implementation provides **three different versions**:

1. ed_multi_lif_snn.py         - Network composed of 100% LIF neurons
2. ed_multi_lif_snn_simple.py  - Basic functionality implementation
3. ed_multi_frelu_snn.py       - Network using FReLU activation function

## About the ED Method

The Error-Diffusion (ED) method is a biologically plausible multi-layer neural network learning algorithm conceived by the late Isamu Kaneko in 1999, which does not use "error backpropagation with chain rule of differentiation".<br>
For detailed information about the ED method, please refer to [ED Method Overview](docs/en/ED_Method_Overview.md).

## Technical Advantages of the ED Method

### 🚀 Acceleration through Parallel Computation

While error backpropagation requires sequential computation from the output layer backwards, the ED method enables **parallel computation between layers** as each layer independently updates weights based on monoamine concentrations. This leads to significant speed improvements, especially in deep networks.

- Independent weight updates based on amine concentrations per layer
- Elimination of sequential computation constraints in backpropagation
- Learning speed improvements in deep networks

### 🛡️ Avoidance of Vanishing Gradient Problem

Since the chain rule of differentiation is not used, **vanishing gradient problems do not occur** even in deep layers. Through local learning via amine diffusion, each layer can directly receive learning signals.

- Local learning without using the chain rule of differentiation
- Stable learning signals regardless of layer depth
- Balance between biological plausibility and practicality

## Overview

This implementation achieves image classification on MNIST and Fashion-MNIST datasets using Isamu Kaneko's biologically plausible "ED method" learning algorithm, without using the biologically less plausible "error backpropagation with chain rule of differentiation".<br>
This implementation is based on a complete spiking neural network where all neurons are LIF neurons.

## Learning Results with ed_multi_lif_snn.py

Examples of learning results using ed_multi_lif_snn.py on MNIST and Fashion-MNIST datasets are shown below.

### MNIST Dataset Learning Example

<img src="viz_results_for_public/lif_mnist_256_lr0.15_e20/realtime_viz_result_20251102_113203.png" alt="MNIST Learning Results" width="60%">

・Maximum Accuracy: 87.60%

・Execution Command

```bash
python ed_multi_lif_snn.py --mnist --train 1000 --test 500 --spike_max_rate 150 --spike_sim_time 50 --spike_dt 1.0 --viz --heatmap --save_fig viz_results_for_public/lif_mnist_256_lr0.15_e20 --epochs 20 --hidden 256 --lr 0.15
```

### Fashion-MNIST Dataset Learning Example

<img src="viz_results_for_public/lif_fashion_256_lr0.15_e20/realtime_viz_result_20251102_113256.png" alt="Fashion-MNIST Learning Results" width="60%">

・Maximum Accuracy: 78.20%

・Execution Command

```bash
python ed_multi_lif_snn.py --fashion --train 1000 --test 500 --spike_max_rate 150 --spike_sim_time 50 --spike_dt 1.0 --viz --heatmap --save_fig viz_results_for_public/lif_fashion_256_lr0.15_e20 --epochs 20 --hidden 256 --lr 0.15
```

## Learning Results with ed_multi_frelu_snn.py

ed_multi_frelu_snn.py is an experimental version of ed_multi_lif_snn.py with the activation function replaced by FReLU.
Examples of learning results using ed_multi_frelu_snn.py on Fashion-MNIST dataset are shown below.

### FReLU Version Fashion-MNIST Dataset Learning Example

<img src="viz_results_for_public/frelu_snn_fashion_hid2048_128_dif1.5_epo20/realtime_viz_result_20251102_062334.png" alt="FReLU Fashion-MNIST Learning Results" width="60%">

・Maximum Accuracy: 76.71%

・Execution Command

```bash
python ed_multi_frelu_snn.py --viz --heatmap --fashion --seed 42 --train 2048 --test 2048 --batch 128 --save_fig viz_results_for_public/frelu_snn_fashion_hid2048_128_dif1.5_epo20 --hidden 2048,128 --epochs 20 --dif 1.5
```

## Experimental Results (Reference)

Detailed records of actual learning trials and reference examples of parameter settings are provided.

### 📊 LIF Version Learning Trial Records

[**lif_snn_learning_test_results.md**](test_results/lif_snn_learning_test_results.md)

- **Complete LIF Version Learning Results**: Multiple trial records on Fashion-MNIST dataset
- **Maximum Achieved Accuracy**: 80.83% (hidden layers: 4096,128,128 neurons, 30 epochs)
- **Parameter Tuning Examples**: Trial and error process for learning rate, amine concentration, and hidden layer structure
- **Execution Details**: Correspondence between command-line arguments and results for each trial

### 🚀 FReLU Version Learning Trial Records

[**frelu_snn_learning_test_results.md**](test_results/frelu_snn_learning_test_results.md)

- **FReLU Version Experimental Results**: Validation of ED method combined with FReLU activation function
- **Performance Comparison**: Analysis comparing LIF and FReLU approaches
- **Parameter Sensitivity**: Investigation of optimal settings for FReLU-based networks

## Implementation Features

### Core Features
- 🧠 **Pure ED Method Implementation**: Complete implementation following Isamu Kaneko's original theory
- 🔬 **Biological Plausibility**: Excitatory/inhibitory neuron pairs with Dale's principle
- ⚡ **SNN Integration**: Full spiking neural network with LIF neuron dynamics
- 🚀 **High Performance**: Optimized with NumPy vectorization and optional GPU support
- 📊 **Real-time Visualization**: Learning progress monitoring with Japanese font support

### Extended Features
- 🎯 **Multi-layer Support**: Dynamic network architecture with multiple hidden layers
- 📦 **Mini-batch Learning**: Efficient batch processing capabilities
- 🔧 **Multiple Encoding**: Poisson, rate, temporal, and population spike encoding
- 🎨 **Heatmap Visualization**: Real-time neuron firing pattern display
- 🔍 **Performance Profiling**: Detailed performance analysis and bottleneck identification

## Quick Start

### Prerequisites
```bash
pip install numpy matplotlib torch torchvision cupy-cuda11x  # Optional: for GPU support
```

### Basic Usage
```bash
# Clone repository
git clone https://github.com/yoiwa0714/ed_multi_snn.git
cd ed_multi_snn

# Use English commented code (src/en/)
cd src/en

# Simple MNIST training
python ed_multi_lif_snn.py --mnist --epochs 10

# Fashion-MNIST with visualization
python ed_multi_lif_snn.py --fashion --viz --heatmap --epochs 20

# Multi-layer network with custom parameters
python ed_multi_lif_snn.py --mnist --hidden 512,256,128 --lr 0.1 --epochs 30
```

> **💡 Language Selection**: 
> - **English commented version**: Use `src/en/` directory
> - **日本語コメント版**: `src/ja/` ディレクトリを使用 ([日本語ガイド](README.md))

For detailed usage instructions, see [USAGE_EN.md](docs/en/USAGE_EN.md).

### Advanced Usage
```bash
# Full feature training with visualization
python ed_multi_lif_snn.py \\
    --fashion \\                    # Use Fashion-MNIST dataset
    --train 2000 --test 1000 \\     # Dataset size
    --hidden 1024,512,256 \\        # Multi-layer architecture
    --epochs 50 \\                  # Training epochs
    --batch 64 \\                   # Batch size
    --lr 0.15 \\                    # Learning rate
    --dif 1.2 \\                    # Amine diffusion rate
    --spike_max_rate 100 \\         # Maximum spike rate
    --spike_sim_time 50 \\          # Simulation time
    --viz --heatmap \\              # Enable visualizations
    --save_fig results/experiment   # Save results
```

## Documentation

### 📚 Core Documentation
- 🇺🇸 [English Technical Documentation](docs/en/)
- 🇯🇵 [Japanese Technical Documentation](docs/ja/)
- 📖 [Complete Specification](ed_multi_snn.prompt_EN.md)
- 🔬 [ED Method Overview](docs/en/ED_Method_Overview.md)

### 🛠️ Development
- 🏗️ [Project Structure](docs/en/Project_Structure.md)
- 🔧 [Module Reference](docs/en/Module_Reference.md)
- 🧪 [Testing Guide](docs/en/Testing_Guide.md)
- 📊 [Performance Analysis](docs/en/Performance_Analysis.md)

## Project Structure

```
ed_multi_snn/
├── README.md                    # Japanese main page
├── README_EN.md                 # English main page
├── ed_multi_snn.prompt.md       # Japanese specification
├── ed_multi_snn.prompt_EN.md    # English specification
├── docs/                        # Documentation
│   ├── ja/                      # Japanese documents
│   └── en/                      # English documents
├── modules/                     # Core modules
│   ├── snn/                     # SNN implementation
│   ├── ed_learning/             # ED method core
│   ├── visualization/           # Visualization tools
│   └── utils/                   # Utilities
├── scripts/                     # Execution scripts
├── test_results/                # Experimental results
└── viz_results_for_public/      # Public visualization results
```

## Research Significance

### Historical Context
The ED method, conceived by Isamu Kaneko in 1999, was ahead of its time in addressing:
- **Biological Plausibility**: Early focus on neurobiologically realistic learning
- **Local Learning**: Independent layer-wise learning without global backpropagation
- **Gradient Vanishing**: Inherent solution to deep network training problems

### Modern Relevance
In the era of Transformers and large language models, the ED method offers:
- **Sustainable Learning**: Energy-efficient local learning algorithms
- **Parallel Processing**: True layer-wise parallel computation capability
- **Biological Inspiration**: Insights for neuromorphic computing and brain-inspired AI

### Future Potential
- 🔋 **Energy Efficiency**: Reduced computational overhead for edge devices
- 🧠 **Neuromorphic Computing**: Direct applicability to brain-inspired hardware
- 🔬 **Computational Neuroscience**: Bridge between artificial and biological neural networks
- 🚀 **Hybrid Architectures**: Integration with modern deep learning approaches

## Contributing

We welcome contributions to this project! Please see our [Contributing Guide](docs/en/Contributing.md) for details on:
- Code style and conventions
- Testing requirements
- Documentation standards
- Pull request process

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

## Citation

If you use this implementation in your research, please cite:

```bibtex
@software{ed_multi_snn_2025,
  title={ED Method Implementation for Spiking Neural Networks},
  author={Yoichi Watanabe},
  year={2025},
  url={https://github.com/yoiwa0714/ed_multi_snn},
  note={Implementation of Isamu Kaneko's Error-Diffusion method for SNNs}
}
```

## Acknowledgments

- **Isamu Kaneko** (1952-2013): Original inventor of the Error-Diffusion learning method
- The computational neuroscience and spiking neural network research communities
- Contributors to the open-source machine learning ecosystem

---

**🌟 Star this repository if you find it useful for your research or projects!**

**📧 For questions or collaboration inquiries, please open an issue on GitHub.**