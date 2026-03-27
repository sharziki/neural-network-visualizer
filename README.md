# Neural Network Visualizer

Build and train neural networks from scratch with no ML libraries. Watch forward propagation, backpropagation, and gradient descent in real-time with interactive visualizations.

## Features

- **From Scratch Implementation**: Pure TypeScript neural network — no TensorFlow, PyTorch, or ML frameworks
- **Interactive Architecture**: Configure hidden layers, neurons per layer, and activation functions
- **Multiple Datasets**: XOR problem, concentric circles, spiral, and Gaussian clusters
- **Real-time Visualization**: Watch neurons activate and weights update during training
- **Decision Boundary**: Live heatmap showing classification regions
- **Training Metrics**: Loss curve, accuracy, epoch counter, and parameter count

## Implemented Components

### Neural Network Core
- Forward propagation with arbitrary depth
- Backpropagation with gradient computation
- Momentum-based gradient descent optimizer
- Xavier/Glorot weight initialization

### Activation Functions
- Sigmoid
- Tanh
- ReLU
- Leaky ReLU

### Datasets
- **XOR**: Classic non-linearly separable problem
- **Circle**: Concentric ring classification
- **Spiral**: Interlocking spiral arms
- **Gaussian**: Overlapping cluster separation

## Tech Stack

- React 19 + TypeScript
- Vite
- Tailwind CSS 4
- Recharts
- Lucide Icons

## Getting Started

```bash
npm install
npm run dev
```

## How It Works

1. **Configure Architecture**: Add/remove hidden layers, adjust neurons per layer
2. **Select Dataset**: Choose a classification problem to solve
3. **Tune Hyperparameters**: Adjust learning rate and momentum
4. **Train**: Click "Start Training" or use "Step" for single epochs
5. **Watch**: Observe the network learn as the decision boundary forms

## Mathematical Background

The network implements:
- **Forward Pass**: `a[l] = σ(W[l] · a[l-1] + b[l])`
- **Loss Function**: Mean Squared Error (MSE)
- **Backprop**: Chain rule gradient computation
- **Update Rule**: `W = W - α(∇W) + β(ΔW_prev)` (momentum)

## License

MIT
