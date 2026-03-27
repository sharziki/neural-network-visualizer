// Neural Network from Scratch - No ML Libraries
// Implements feedforward network with backpropagation

export type ActivationType = 'sigmoid' | 'tanh' | 'relu' | 'leakyRelu' | 'linear';

// Activation functions and their derivatives
const activations = {
  sigmoid: {
    fn: (x: number) => 1 / (1 + Math.exp(-Math.max(-500, Math.min(500, x)))),
    derivative: (x: number) => {
      const s = 1 / (1 + Math.exp(-Math.max(-500, Math.min(500, x))));
      return s * (1 - s);
    },
  },
  tanh: {
    fn: (x: number) => Math.tanh(x),
    derivative: (x: number) => 1 - Math.tanh(x) ** 2,
  },
  relu: {
    fn: (x: number) => Math.max(0, x),
    derivative: (x: number) => (x > 0 ? 1 : 0),
  },
  leakyRelu: {
    fn: (x: number) => (x > 0 ? x : 0.01 * x),
    derivative: (x: number) => (x > 0 ? 1 : 0.01),
  },
  linear: {
    fn: (x: number) => x,
    derivative: (_x: number) => 1,
  },
};

// Network layer structure
export interface Layer {
  weights: number[][]; // weights[i][j] = weight from input i to output j
  biases: number[];
  activation: ActivationType;
}

// Network state for visualization
export interface NetworkState {
  layers: Layer[];
  activations: number[][]; // activation values at each layer
  preActivations: number[][]; // values before activation function
  gradients: number[][][]; // weight gradients
  biasGradients: number[][];
}

// Training history
export interface TrainingHistory {
  epoch: number;
  loss: number;
  accuracy: number;
}

// Dataset types
export type DatasetType = 'xor' | 'circle' | 'spiral' | 'gaussian';

export interface DataPoint {
  input: number[];
  target: number[];
}

// Neural Network class
export class NeuralNetwork {
  layers: Layer[] = [];
  learningRate: number;
  momentum: number;
  velocities: number[][][] = [];
  biasVelocities: number[][] = [];

  constructor(learningRate = 0.1, momentum = 0.9) {
    this.learningRate = learningRate;
    this.momentum = momentum;
  }

  // Initialize network with given architecture
  initialize(architecture: number[], activation: ActivationType = 'sigmoid'): void {
    this.layers = [];
    this.velocities = [];
    this.biasVelocities = [];

    for (let i = 0; i < architecture.length - 1; i++) {
      const inputSize = architecture[i];
      const outputSize = architecture[i + 1];

      // Xavier/Glorot initialization
      const scale = Math.sqrt(2 / (inputSize + outputSize));

      const weights: number[][] = [];
      for (let j = 0; j < inputSize; j++) {
        weights.push([]);
        for (let k = 0; k < outputSize; k++) {
          weights[j].push((Math.random() * 2 - 1) * scale);
        }
      }

      const biases = new Array(outputSize).fill(0).map(() => (Math.random() * 2 - 1) * 0.1);

      this.layers.push({
        weights,
        biases,
        activation: i === architecture.length - 2 ? 'sigmoid' : activation,
      });

      // Initialize velocities for momentum
      this.velocities.push(weights.map((row) => row.map(() => 0)));
      this.biasVelocities.push(biases.map(() => 0));
    }
  }

  // Forward propagation
  forward(input: number[]): NetworkState {
    const activationValues: number[][] = [input];
    const preActivationValues: number[][] = [input];
    let current = input;

    for (const layer of this.layers) {
      const preAct: number[] = [];
      const act: number[] = [];

      for (let j = 0; j < layer.biases.length; j++) {
        let sum = layer.biases[j];
        for (let i = 0; i < current.length; i++) {
          sum += current[i] * layer.weights[i][j];
        }
        preAct.push(sum);
        act.push(activations[layer.activation].fn(sum));
      }

      preActivationValues.push(preAct);
      activationValues.push(act);
      current = act;
    }

    return {
      layers: this.layers,
      activations: activationValues,
      preActivations: preActivationValues,
      gradients: [],
      biasGradients: [],
    };
  }

  // Backpropagation
  backward(state: NetworkState, target: number[]): NetworkState {
    const gradients: number[][][] = [];
    const biasGradients: number[][] = [];
    const deltas: number[][] = [];

    // Output layer error
    const outputLayer = this.layers[this.layers.length - 1];
    const outputAct = state.activations[state.activations.length - 1];
    const outputPreAct = state.preActivations[state.preActivations.length - 1];

    const outputDelta: number[] = [];
    for (let i = 0; i < outputAct.length; i++) {
      const error = outputAct[i] - target[i];
      const derivative = activations[outputLayer.activation].derivative(outputPreAct[i]);
      outputDelta.push(error * derivative);
    }
    deltas.unshift(outputDelta);

    // Hidden layer errors (backpropagate)
    for (let l = this.layers.length - 2; l >= 0; l--) {
      const layer = this.layers[l];
      const nextLayer = this.layers[l + 1];
      const preAct = state.preActivations[l + 1];
      const nextDelta = deltas[0];

      const delta: number[] = [];
      for (let i = 0; i < layer.biases.length; i++) {
        let error = 0;
        for (let j = 0; j < nextDelta.length; j++) {
          error += nextDelta[j] * nextLayer.weights[i][j];
        }
        const derivative = activations[layer.activation].derivative(preAct[i]);
        delta.push(error * derivative);
      }
      deltas.unshift(delta);
    }

    // Compute gradients
    for (let l = 0; l < this.layers.length; l++) {
      const prevActivations = state.activations[l];
      const delta = deltas[l];

      const layerGradients: number[][] = [];
      for (let i = 0; i < prevActivations.length; i++) {
        layerGradients.push([]);
        for (let j = 0; j < delta.length; j++) {
          layerGradients[i].push(prevActivations[i] * delta[j]);
        }
      }
      gradients.push(layerGradients);
      biasGradients.push(delta);
    }

    return {
      ...state,
      gradients,
      biasGradients,
    };
  }

  // Update weights with momentum
  updateWeights(state: NetworkState): void {
    for (let l = 0; l < this.layers.length; l++) {
      const layer = this.layers[l];

      for (let i = 0; i < layer.weights.length; i++) {
        for (let j = 0; j < layer.weights[i].length; j++) {
          this.velocities[l][i][j] =
            this.momentum * this.velocities[l][i][j] - this.learningRate * state.gradients[l][i][j];
          layer.weights[i][j] += this.velocities[l][i][j];
        }
      }

      for (let j = 0; j < layer.biases.length; j++) {
        this.biasVelocities[l][j] =
          this.momentum * this.biasVelocities[l][j] - this.learningRate * state.biasGradients[l][j];
        layer.biases[j] += this.biasVelocities[l][j];
      }
    }
  }

  // Train on single example
  trainStep(input: number[], target: number[]): { loss: number; state: NetworkState } {
    const forwardState = this.forward(input);
    const backwardState = this.backward(forwardState, target);
    this.updateWeights(backwardState);

    // Compute loss (MSE)
    const output = forwardState.activations[forwardState.activations.length - 1];
    let loss = 0;
    for (let i = 0; i < output.length; i++) {
      loss += (output[i] - target[i]) ** 2;
    }
    loss /= output.length;

    return { loss, state: backwardState };
  }

  // Train on dataset
  trainEpoch(data: DataPoint[]): { loss: number; accuracy: number } {
    let totalLoss = 0;
    let correct = 0;

    // Shuffle data
    const shuffled = [...data].sort(() => Math.random() - 0.5);

    for (const point of shuffled) {
      const { loss } = this.trainStep(point.input, point.target);
      totalLoss += loss;

      // Check accuracy
      const prediction = this.predict(point.input);
      const predictedClass = prediction.indexOf(Math.max(...prediction));
      const actualClass = point.target.indexOf(Math.max(...point.target));
      if (predictedClass === actualClass) correct++;
    }

    return {
      loss: totalLoss / data.length,
      accuracy: correct / data.length,
    };
  }

  // Predict (forward only)
  predict(input: number[]): number[] {
    const state = this.forward(input);
    return state.activations[state.activations.length - 1];
  }

  // Get network architecture
  getArchitecture(): number[] {
    if (this.layers.length === 0) return [];
    const arch = [this.layers[0].weights.length];
    for (const layer of this.layers) {
      arch.push(layer.biases.length);
    }
    return arch;
  }

  // Get weights for visualization
  getWeights(): number[][][] {
    return this.layers.map((layer) => layer.weights);
  }

  // Get biases for visualization
  getBiases(): number[][] {
    return this.layers.map((layer) => layer.biases);
  }
}

// Dataset generation functions
export function generateDataset(type: DatasetType, numPoints = 200): DataPoint[] {
  switch (type) {
    case 'xor':
      return generateXOR(numPoints);
    case 'circle':
      return generateCircle(numPoints);
    case 'spiral':
      return generateSpiral(numPoints);
    case 'gaussian':
      return generateGaussian(numPoints);
    default:
      return generateXOR(numPoints);
  }
}

function generateXOR(n: number): DataPoint[] {
  const data: DataPoint[] = [];
  for (let i = 0; i < n; i++) {
    const x = Math.random();
    const y = Math.random();
    const label = (x > 0.5) !== (y > 0.5) ? 1 : 0;
    data.push({
      input: [x, y],
      target: [label === 0 ? 1 : 0, label === 1 ? 1 : 0],
    });
  }
  return data;
}

function generateCircle(n: number): DataPoint[] {
  const data: DataPoint[] = [];
  for (let i = 0; i < n; i++) {
    const r = Math.random() * 0.5;
    const theta = Math.random() * Math.PI * 2;
    const inner = Math.random() > 0.5;
    const radius = inner ? r * 0.5 : 0.3 + r * 0.3;
    const x = 0.5 + radius * Math.cos(theta);
    const y = 0.5 + radius * Math.sin(theta);
    const label = inner ? 1 : 0;
    data.push({
      input: [x, y],
      target: [label === 0 ? 1 : 0, label === 1 ? 1 : 0],
    });
  }
  return data;
}

function generateSpiral(n: number): DataPoint[] {
  const data: DataPoint[] = [];
  const pointsPerClass = Math.floor(n / 2);

  for (let i = 0; i < pointsPerClass; i++) {
    const r = (i / pointsPerClass) * 0.4;
    const theta = (i / pointsPerClass) * Math.PI * 2.5;
    const noise = (Math.random() - 0.5) * 0.1;

    // Class 0 - first spiral
    data.push({
      input: [0.5 + r * Math.cos(theta) + noise, 0.5 + r * Math.sin(theta) + noise],
      target: [1, 0],
    });

    // Class 1 - second spiral (rotated)
    data.push({
      input: [
        0.5 + r * Math.cos(theta + Math.PI) + noise,
        0.5 + r * Math.sin(theta + Math.PI) + noise,
      ],
      target: [0, 1],
    });
  }

  return data;
}

function generateGaussian(n: number): DataPoint[] {
  const data: DataPoint[] = [];

  const gaussianRandom = () => {
    let u = 0,
      v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
  };

  for (let i = 0; i < n; i++) {
    const label = Math.random() > 0.5 ? 1 : 0;
    const centerX = label === 0 ? 0.35 : 0.65;
    const centerY = label === 0 ? 0.35 : 0.65;

    data.push({
      input: [centerX + gaussianRandom() * 0.1, centerY + gaussianRandom() * 0.1],
      target: [label === 0 ? 1 : 0, label === 1 ? 1 : 0],
    });
  }

  return data;
}

// Compute decision boundary
export function computeDecisionBoundary(
  network: NeuralNetwork,
  resolution = 50
): { x: number; y: number; value: number }[] {
  const points: { x: number; y: number; value: number }[] = [];

  for (let i = 0; i < resolution; i++) {
    for (let j = 0; j < resolution; j++) {
      const x = i / (resolution - 1);
      const y = j / (resolution - 1);
      const prediction = network.predict([x, y]);
      points.push({
        x,
        y,
        value: prediction[1] - prediction[0], // Difference shows decision boundary
      });
    }
  }

  return points;
}
