import { memo } from 'react';
import { Play, Pause, RotateCcw, Zap, Layers, Database, Gauge, Settings2 } from 'lucide-react';
import type { ActivationType, DatasetType } from '../lib/neuralNetwork';

interface ControlsProps {
  // Architecture
  hiddenLayers: number[];
  setHiddenLayers: (layers: number[]) => void;
  activation: ActivationType;
  setActivation: (activation: ActivationType) => void;

  // Dataset
  dataset: DatasetType;
  setDataset: (dataset: DatasetType) => void;
  datasetSize: number;
  setDatasetSize: (size: number) => void;

  // Training
  learningRate: number;
  setLearningRate: (lr: number) => void;
  momentum: number;
  setMomentum: (m: number) => void;
  batchSize: number;
  setBatchSize: (size: number) => void;

  // Actions
  isTraining: boolean;
  onStartStop: () => void;
  onReset: () => void;
  onStep: () => void;
  showBoundary: boolean;
  setShowBoundary: (show: boolean) => void;
}

export const Controls = memo(function Controls({
  hiddenLayers,
  setHiddenLayers,
  activation,
  setActivation,
  dataset,
  setDataset,
  datasetSize,
  setDatasetSize,
  learningRate,
  setLearningRate,
  momentum,
  setMomentum,
  batchSize: _batchSize,
  setBatchSize: _setBatchSize,
  isTraining,
  onStartStop,
  onReset,
  onStep,
  showBoundary,
  setShowBoundary,
}: ControlsProps) {
  const updateLayer = (index: number, value: number) => {
    const newLayers = [...hiddenLayers];
    newLayers[index] = value;
    setHiddenLayers(newLayers);
  };

  const addLayer = () => {
    if (hiddenLayers.length < 4) {
      setHiddenLayers([...hiddenLayers, 4]);
    }
  };

  const removeLayer = () => {
    if (hiddenLayers.length > 1) {
      setHiddenLayers(hiddenLayers.slice(0, -1));
    }
  };

  return (
    <div className="bg-[hsl(var(--card))] rounded-xl border border-[hsl(var(--border))] p-4 space-y-5">
      {/* Architecture Section */}
      <div>
        <div className="flex items-center gap-2 mb-3">
          <Layers className="w-4 h-4 text-[hsl(var(--primary))]" />
          <h3 className="text-sm font-medium text-[hsl(var(--foreground))]">Architecture</h3>
        </div>

        <div className="space-y-3">
          <div>
            <label className="text-xs text-[hsl(var(--muted-foreground))] mb-1 block">
              Hidden Layers ({hiddenLayers.length})
            </label>
            <div className="flex items-center gap-2">
              {hiddenLayers.map((neurons, idx) => (
                <input
                  key={idx}
                  type="number"
                  min={1}
                  max={16}
                  value={neurons}
                  onChange={(e) => updateLayer(idx, parseInt(e.target.value) || 1)}
                  className="w-12 px-2 py-1 bg-[hsl(var(--secondary))] border border-[hsl(var(--border))] rounded text-sm text-center text-[hsl(var(--foreground))]"
                  disabled={isTraining}
                />
              ))}
              <button
                onClick={addLayer}
                disabled={isTraining || hiddenLayers.length >= 4}
                className="w-8 h-8 flex items-center justify-center bg-[hsl(var(--secondary))] border border-[hsl(var(--border))] rounded text-[hsl(var(--muted-foreground))] hover:text-[hsl(var(--foreground))] disabled:opacity-50"
              >
                +
              </button>
              <button
                onClick={removeLayer}
                disabled={isTraining || hiddenLayers.length <= 1}
                className="w-8 h-8 flex items-center justify-center bg-[hsl(var(--secondary))] border border-[hsl(var(--border))] rounded text-[hsl(var(--muted-foreground))] hover:text-[hsl(var(--foreground))] disabled:opacity-50"
              >
                −
              </button>
            </div>
          </div>

          <div>
            <label className="text-xs text-[hsl(var(--muted-foreground))] mb-1 block">Activation Function</label>
            <select
              value={activation}
              onChange={(e) => setActivation(e.target.value as ActivationType)}
              disabled={isTraining}
              className="w-full px-3 py-2 bg-[hsl(var(--secondary))] border border-[hsl(var(--border))] rounded-lg text-sm text-[hsl(var(--foreground))]"
            >
              <option value="sigmoid">Sigmoid</option>
              <option value="tanh">Tanh</option>
              <option value="relu">ReLU</option>
              <option value="leakyRelu">Leaky ReLU</option>
            </select>
          </div>
        </div>
      </div>

      {/* Dataset Section */}
      <div>
        <div className="flex items-center gap-2 mb-3">
          <Database className="w-4 h-4 text-[hsl(var(--primary))]" />
          <h3 className="text-sm font-medium text-[hsl(var(--foreground))]">Dataset</h3>
        </div>

        <div className="space-y-3">
          <div>
            <label className="text-xs text-[hsl(var(--muted-foreground))] mb-1 block">Pattern</label>
            <select
              value={dataset}
              onChange={(e) => setDataset(e.target.value as DatasetType)}
              disabled={isTraining}
              className="w-full px-3 py-2 bg-[hsl(var(--secondary))] border border-[hsl(var(--border))] rounded-lg text-sm text-[hsl(var(--foreground))]"
            >
              <option value="xor">XOR Problem</option>
              <option value="circle">Concentric Circles</option>
              <option value="spiral">Spiral</option>
              <option value="gaussian">Gaussian Clusters</option>
            </select>
          </div>

          <div>
            <label className="text-xs text-[hsl(var(--muted-foreground))] mb-1 block">
              Data Points: {datasetSize}
            </label>
            <input
              type="range"
              min={50}
              max={500}
              step={50}
              value={datasetSize}
              onChange={(e) => setDatasetSize(parseInt(e.target.value))}
              disabled={isTraining}
              className="w-full accent-[hsl(var(--primary))]"
            />
          </div>
        </div>
      </div>

      {/* Training Parameters */}
      <div>
        <div className="flex items-center gap-2 mb-3">
          <Settings2 className="w-4 h-4 text-[hsl(var(--primary))]" />
          <h3 className="text-sm font-medium text-[hsl(var(--foreground))]">Training</h3>
        </div>

        <div className="space-y-3">
          <div>
            <label className="text-xs text-[hsl(var(--muted-foreground))] mb-1 block">
              Learning Rate: {learningRate.toFixed(3)}
            </label>
            <input
              type="range"
              min={0.001}
              max={1}
              step={0.001}
              value={learningRate}
              onChange={(e) => setLearningRate(parseFloat(e.target.value))}
              disabled={isTraining}
              className="w-full accent-[hsl(var(--primary))]"
            />
          </div>

          <div>
            <label className="text-xs text-[hsl(var(--muted-foreground))] mb-1 block">
              Momentum: {momentum.toFixed(2)}
            </label>
            <input
              type="range"
              min={0}
              max={0.99}
              step={0.01}
              value={momentum}
              onChange={(e) => setMomentum(parseFloat(e.target.value))}
              disabled={isTraining}
              className="w-full accent-[hsl(var(--primary))]"
            />
          </div>
        </div>
      </div>

      {/* Visualization */}
      <div>
        <div className="flex items-center gap-2 mb-3">
          <Gauge className="w-4 h-4 text-[hsl(var(--primary))]" />
          <h3 className="text-sm font-medium text-[hsl(var(--foreground))]">Visualization</h3>
        </div>

        <label className="flex items-center gap-2 cursor-pointer">
          <input
            type="checkbox"
            checked={showBoundary}
            onChange={(e) => setShowBoundary(e.target.checked)}
            className="w-4 h-4 accent-[hsl(var(--primary))]"
          />
          <span className="text-sm text-[hsl(var(--foreground))]">Show Decision Boundary</span>
        </label>
      </div>

      {/* Action Buttons */}
      <div className="space-y-2 pt-2">
        <button
          onClick={onStartStop}
          className={`w-full py-2.5 px-4 rounded-lg font-medium flex items-center justify-center gap-2 transition-colors ${
            isTraining
              ? 'bg-[hsl(var(--destructive))] text-white hover:bg-[hsl(0,84%,55%)]'
              : 'bg-[hsl(var(--primary))] text-white hover:bg-[hsl(263,70%,53%)]'
          }`}
        >
          {isTraining ? (
            <>
              <Pause className="w-4 h-4" />
              Stop Training
            </>
          ) : (
            <>
              <Play className="w-4 h-4" />
              Start Training
            </>
          )}
        </button>

        <div className="flex gap-2">
          <button
            onClick={onStep}
            disabled={isTraining}
            className="flex-1 py-2 px-3 bg-[hsl(var(--secondary))] border border-[hsl(var(--border))] rounded-lg font-medium flex items-center justify-center gap-2 text-[hsl(var(--foreground))] hover:bg-[hsl(217,33%,22%)] disabled:opacity-50 transition-colors"
          >
            <Zap className="w-4 h-4" />
            Step
          </button>
          <button
            onClick={onReset}
            disabled={isTraining}
            className="flex-1 py-2 px-3 bg-[hsl(var(--secondary))] border border-[hsl(var(--border))] rounded-lg font-medium flex items-center justify-center gap-2 text-[hsl(var(--foreground))] hover:bg-[hsl(217,33%,22%)] disabled:opacity-50 transition-colors"
          >
            <RotateCcw className="w-4 h-4" />
            Reset
          </button>
        </div>
      </div>
    </div>
  );
});
