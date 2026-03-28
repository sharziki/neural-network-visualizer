import { useState, useCallback, useRef, useEffect } from 'react';
import { Brain } from 'lucide-react';
import { NetworkVisualization } from './components/NetworkVisualization';
import { DatasetVisualization } from './components/DatasetVisualization';
import { TrainingChart } from './components/TrainingChart';
import { Controls } from './components/Controls';
import { StatsPanel } from './components/StatsPanel';
import {
  NeuralNetwork,
  generateDataset,
  computeDecisionBoundary,
  type ActivationType,
  type DatasetType,
  type TrainingHistory,
  type NetworkState,
  type DataPoint,
} from './lib/neuralNetwork';

function App() {
  // Network configuration
  const [hiddenLayers, setHiddenLayers] = useState<number[]>([8, 8]);
  const [activation, setActivation] = useState<ActivationType>('tanh');

  // Dataset configuration
  const [datasetType, setDatasetType] = useState<DatasetType>('xor');
  const [datasetSize, setDatasetSize] = useState(200);
  const [data, setData] = useState<DataPoint[]>(() => generateDataset('xor', 200));

  // Training parameters
  const [learningRate, setLearningRate] = useState(0.3);
  const [momentum, setMomentum] = useState(0.9);
  const [batchSize, setBatchSize] = useState(1);

  // Training state
  const [isTraining, setIsTraining] = useState(false);
  const [epoch, setEpoch] = useState(0);
  const [currentLoss, setCurrentLoss] = useState(0);
  const [currentAccuracy, setCurrentAccuracy] = useState(0);
  const [history, setHistory] = useState<TrainingHistory[]>([]);
  const [networkState, setNetworkState] = useState<NetworkState | null>(null);
  const [decisionBoundary, setDecisionBoundary] = useState<
    { x: number; y: number; value: number }[]
  >([]);

  // Visualization options
  const [showBoundary, setShowBoundary] = useState(true);

  // Refs
  const networkRef = useRef<NeuralNetwork | null>(null);
  const animationRef = useRef<number | null>(null);
  const isTrainingRef = useRef(false);

  // Architecture includes input (2), hidden layers, and output (2)
  const architecture = [2, ...hiddenLayers, 2];

  // Initialize network
  const initializeNetwork = useCallback(() => {
    const nn = new NeuralNetwork(learningRate, momentum);
    nn.initialize(architecture, activation);
    networkRef.current = nn;

    // Get initial state
    const state = nn.forward([0.5, 0.5]);
    setNetworkState(state);

    // Compute initial decision boundary
    if (showBoundary) {
      setDecisionBoundary(computeDecisionBoundary(nn, 40));
    }
  }, [architecture, activation, learningRate, momentum, showBoundary]);

  // Initialize on mount and when architecture changes
  useEffect(() => {
    initializeNetwork();
  }, []);

  // Regenerate dataset when type or size changes
  useEffect(() => {
    const newData = generateDataset(datasetType, datasetSize);
    setData(newData);
  }, [datasetType, datasetSize]);

  // Training loop
  const trainStep = useCallback(() => {
    if (!networkRef.current || !isTrainingRef.current) return;

    const nn = networkRef.current;
    const { loss, accuracy } = nn.trainEpoch(data);

    setEpoch((prev) => {
      const newEpoch = prev + 1;
      setHistory((prevHistory) => [...prevHistory, { epoch: newEpoch, loss, accuracy }]);
      return newEpoch;
    });
    setCurrentLoss(loss);
    setCurrentAccuracy(accuracy);

    // Update network state for visualization
    const state = nn.forward(data[0].input);
    setNetworkState(state);

    // Update decision boundary periodically
    if (showBoundary) {
      setDecisionBoundary(computeDecisionBoundary(nn, 40));
    }

    // Continue training
    if (isTrainingRef.current) {
      animationRef.current = requestAnimationFrame(trainStep);
    }
  }, [data, showBoundary]);

  // Start/Stop training
  const handleStartStop = useCallback(() => {
    if (isTraining) {
      // Stop training
      isTrainingRef.current = false;
      setIsTraining(false);
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    } else {
      // Start training - reinitialize if needed
      if (!networkRef.current) {
        initializeNetwork();
      }
      isTrainingRef.current = true;
      setIsTraining(true);
      animationRef.current = requestAnimationFrame(trainStep);
    }
  }, [isTraining, initializeNetwork, trainStep]);

  // Single training step
  const handleStep = useCallback(() => {
    if (!networkRef.current) {
      initializeNetwork();
    }

    const nn = networkRef.current!;
    const { loss, accuracy } = nn.trainEpoch(data);

    setEpoch((prev) => {
      const newEpoch = prev + 1;
      setHistory((prevHistory) => [...prevHistory, { epoch: newEpoch, loss, accuracy }]);
      return newEpoch;
    });
    setCurrentLoss(loss);
    setCurrentAccuracy(accuracy);

    // Update network state
    const state = nn.forward(data[0].input);
    setNetworkState(state);

    // Update decision boundary
    if (showBoundary) {
      setDecisionBoundary(computeDecisionBoundary(nn, 40));
    }
  }, [data, initializeNetwork, showBoundary]);

  // Reset network
  const handleReset = useCallback(() => {
    // Stop training if running
    isTrainingRef.current = false;
    setIsTraining(false);
    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current);
    }

    // Reset state
    setEpoch(0);
    setCurrentLoss(0);
    setCurrentAccuracy(0);
    setHistory([]);

    // Regenerate dataset
    const newData = generateDataset(datasetType, datasetSize);
    setData(newData);

    // Reinitialize network
    initializeNetwork();
  }, [datasetType, datasetSize, initializeNetwork]);

  // Handle architecture changes
  const handleHiddenLayersChange = useCallback(
    (layers: number[]) => {
      setHiddenLayers(layers);
      handleReset();
    },
    [handleReset]
  );

  const handleActivationChange = useCallback(
    (act: ActivationType) => {
      setActivation(act);
      handleReset();
    },
    [handleReset]
  );

  const handleDatasetChange = useCallback(
    (ds: DatasetType) => {
      setDatasetType(ds);
      handleReset();
    },
    [handleReset]
  );

  const handleDatasetSizeChange = useCallback(
    (size: number) => {
      setDatasetSize(size);
      handleReset();
    },
    [handleReset]
  );

  const handleLearningRateChange = useCallback(
    (lr: number) => {
      setLearningRate(lr);
      if (networkRef.current) {
        networkRef.current.learningRate = lr;
      }
    },
    []
  );

  const handleMomentumChange = useCallback(
    (m: number) => {
      setMomentum(m);
      if (networkRef.current) {
        networkRef.current.momentum = m;
      }
    },
    []
  );

  return (
    <div className="min-h-screen bg-[hsl(var(--background))]">
      {/* Header */}
      <header className="border-b border-[hsl(var(--border))] bg-[hsl(var(--card))]">
        <div className="max-w-7xl mx-auto px-6 py-6">
      <div className="max-w-[1600px] mx-auto space-y-6">
        {/* Header */}
        <div className="flex items-center gap-6">
          <div className="p-2 rounded-lg bg-[hsl(var(--primary))] bg-opacity-20">
            <Brain className="w-8 h-8 text-[hsl(var(--primary))]" />
          </div>
          <div>
            <h1 className="text-2xl font-bold text-[hsl(var(--foreground))]">
              Neural Network Visualizer
            </h1>
            <p className="text-sm text-[hsl(var(--muted-foreground))]">
              Build and train neural networks from scratch — no ML libraries
            </p>
          </div>
        </div>

        {/* Stats Panel */}
        <StatsPanel
          epoch={epoch}
          loss={currentLoss}
          accuracy={currentAccuracy}
          isTraining={isTraining}
          architecture={architecture}
        />

        {/* Main Content */}
        <div className="flex gap-6">
          {/* Controls */}
          <div className="w-[280px] flex-shrink-0">
            <Controls
              hiddenLayers={hiddenLayers}
              setHiddenLayers={handleHiddenLayersChange}
              activation={activation}
              setActivation={handleActivationChange}
              dataset={datasetType}
              setDataset={handleDatasetChange}
              datasetSize={datasetSize}
              setDatasetSize={handleDatasetSizeChange}
              learningRate={learningRate}
              setLearningRate={handleLearningRateChange}
              momentum={momentum}
              setMomentum={handleMomentumChange}
              batchSize={batchSize}
              setBatchSize={setBatchSize}
              isTraining={isTraining}
              onStartStop={handleStartStop}
              onReset={handleReset}
              onStep={handleStep}
              showBoundary={showBoundary}
              setShowBoundary={setShowBoundary}
            />
          </div>

          {/* Visualizations */}
          <div className="flex-1 space-y-6">
            {/* Top row: Network and Dataset */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <NetworkVisualization
                architecture={architecture}
                networkState={networkState}
                isTraining={isTraining}
              />
              <DatasetVisualization
                data={data}
                decisionBoundary={decisionBoundary}
                showBoundary={showBoundary}
              />
            </div>

            {/* Training Chart */}
            <TrainingChart history={history} />
          </div>
        </div>

        {/* Footer */}
        <div className="text-center text-xs text-[hsl(var(--muted-foreground))] pt-4">
          <p>
            Configure the network architecture, choose a dataset pattern, and watch the network learn
            in real-time.
          </p>
        </div>
        <footer className="mt-8 py-4 text-center text-xs text-[hsl(var(--muted-foreground))] border-t border-[hsl(var(--border))]">
          Made by{' '}
          <a 
            href="https://github.com/sharziki" 
            target="_blank" 
            rel="noopener noreferrer"
            className="text-[hsl(var(--primary))] hover:underline"
          >
            Sharvil Saxena
          </a>
        </footer>

      </div>
    </div>
  );
}

export default App;