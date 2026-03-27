import { memo, useMemo } from 'react';
import type { NetworkState } from '../lib/neuralNetwork';

interface NetworkVisualizationProps {
  architecture: number[];
  networkState: NetworkState | null;
  isTraining: boolean;
}

export const NetworkVisualization = memo(function NetworkVisualization({
  architecture,
  networkState,
  isTraining,
}: NetworkVisualizationProps) {
  const width = 600;
  const height = 400;
  const padding = 60;

  const { neurons, connections } = useMemo(() => {
    if (architecture.length === 0) {
      return { neurons: [], connections: [] };
    }

    const maxNeurons = Math.max(...architecture);
    const layerWidth = (width - 2 * padding) / (architecture.length - 1);
    const neuronRadius = Math.min(20, (height - 2 * padding) / (maxNeurons * 2.5));

    const neuronPositions: { x: number; y: number; layer: number; index: number }[] = [];
    const connectionList: {
      x1: number;
      y1: number;
      x2: number;
      y2: number;
      weight: number;
      fromLayer: number;
      fromIndex: number;
      toIndex: number;
    }[] = [];

    // Calculate neuron positions
    for (let l = 0; l < architecture.length; l++) {
      const numNeurons = architecture[l];
      const layerHeight = numNeurons * neuronRadius * 2.5;
      const startY = (height - layerHeight) / 2 + neuronRadius;

      for (let n = 0; n < numNeurons; n++) {
        neuronPositions.push({
          x: padding + l * layerWidth,
          y: startY + n * neuronRadius * 2.5,
          layer: l,
          index: n,
        });
      }
    }

    // Calculate connections
    if (networkState && networkState.layers.length > 0) {
      for (let l = 0; l < architecture.length - 1; l++) {
        const weights = networkState.layers[l].weights;
        for (let i = 0; i < architecture[l]; i++) {
          for (let j = 0; j < architecture[l + 1]; j++) {
            const fromNeuron = neuronPositions.find((n) => n.layer === l && n.index === i)!;
            const toNeuron = neuronPositions.find((n) => n.layer === l + 1 && n.index === j)!;
            connectionList.push({
              x1: fromNeuron.x,
              y1: fromNeuron.y,
              x2: toNeuron.x,
              y2: toNeuron.y,
              weight: weights[i][j],
              fromLayer: l,
              fromIndex: i,
              toIndex: j,
            });
          }
        }
      }
    }

    return { neurons: neuronPositions, connections: connectionList };
  }, [architecture, networkState, width, height]);

  const getActivationColor = (layer: number, index: number): string => {
    if (!networkState || networkState.activations.length === 0) {
      if (layer === 0) return 'hsl(var(--neuron-input))';
      if (layer === architecture.length - 1) return 'hsl(var(--neuron-output))';
      return 'hsl(var(--neuron-hidden))';
    }

    const activation = networkState.activations[layer]?.[index] ?? 0;
    const intensity = Math.min(1, Math.max(0, activation));

    if (layer === 0) {
      return `hsl(142, 76%, ${30 + intensity * 30}%)`;
    }
    if (layer === architecture.length - 1) {
      return `hsl(25, 95%, ${35 + intensity * 30}%)`;
    }
    return `hsl(263, 70%, ${35 + intensity * 35}%)`;
  };

  const getWeightColor = (weight: number): string => {
    const absWeight = Math.min(3, Math.abs(weight));
    const opacity = 0.15 + (absWeight / 3) * 0.6;
    if (weight > 0) {
      return `hsla(142, 76%, 46%, ${opacity})`;
    }
    return `hsla(0, 84%, 60%, ${opacity})`;
  };

  const getWeightWidth = (weight: number): number => {
    const absWeight = Math.min(3, Math.abs(weight));
    return 0.5 + (absWeight / 3) * 2.5;
  };

  const neuronRadius = Math.min(20, (height - 2 * padding) / (Math.max(...architecture, 1) * 2.5));

  return (
    <div className="bg-[hsl(var(--card))] rounded-xl border border-[hsl(var(--border))] p-4">
      <h3 className="text-sm font-medium text-[hsl(var(--foreground))] mb-3">Network Architecture</h3>
      <svg
        viewBox={`0 0 ${width} ${height}`}
        className={`w-full network-canvas rounded-lg ${isTraining ? 'training-active' : ''}`}
      >
        {/* Connections */}
        <g>
          {connections.map((conn, idx) => (
            <line
              key={`conn-${idx}`}
              x1={conn.x1}
              y1={conn.y1}
              x2={conn.x2}
              y2={conn.y2}
              stroke={getWeightColor(conn.weight)}
              strokeWidth={getWeightWidth(conn.weight)}
            />
          ))}
        </g>

        {/* Neurons */}
        <g>
          {neurons.map((neuron, idx) => (
            <g key={`neuron-${idx}`}>
              {/* Glow effect */}
              <circle
                cx={neuron.x}
                cy={neuron.y}
                r={neuronRadius + 4}
                fill="none"
                stroke={getActivationColor(neuron.layer, neuron.index)}
                strokeWidth={2}
                opacity={0.3}
              />
              {/* Main neuron */}
              <circle
                cx={neuron.x}
                cy={neuron.y}
                r={neuronRadius}
                fill={getActivationColor(neuron.layer, neuron.index)}
                className={isTraining ? 'animate-pulse-slow' : ''}
              />
              {/* Activation value */}
              {networkState && networkState.activations[neuron.layer] && (
                <text
                  x={neuron.x}
                  y={neuron.y}
                  textAnchor="middle"
                  dominantBaseline="middle"
                  fill="white"
                  fontSize={neuronRadius * 0.6}
                  fontWeight="bold"
                >
                  {networkState.activations[neuron.layer][neuron.index]?.toFixed(2) ?? '0'}
                </text>
              )}
            </g>
          ))}
        </g>

        {/* Layer labels */}
        {architecture.map((_, l) => {
          const x =
            l === 0
              ? padding
              : l === architecture.length - 1
                ? width - padding
                : padding + l * ((width - 2 * padding) / (architecture.length - 1));
          return (
            <text
              key={`label-${l}`}
              x={x}
              y={height - 15}
              textAnchor="middle"
              fill="hsl(var(--muted-foreground))"
              fontSize={12}
            >
              {l === 0 ? 'Input' : l === architecture.length - 1 ? 'Output' : `Hidden ${l}`}
            </text>
          );
        })}
      </svg>

      {/* Legend */}
      <div className="flex items-center justify-center gap-6 mt-3 text-xs text-[hsl(var(--muted-foreground))]">
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-[hsl(var(--neuron-input))]" />
          <span>Input</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-[hsl(var(--neuron-hidden))]" />
          <span>Hidden</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-[hsl(var(--neuron-output))]" />
          <span>Output</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-8 h-0.5 bg-[hsl(var(--weight-positive))]" />
          <span>+Weight</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-8 h-0.5 bg-[hsl(var(--weight-negative))]" />
          <span>-Weight</span>
        </div>
      </div>
    </div>
  );
});
