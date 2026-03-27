import { memo, useMemo } from 'react';
import type { DataPoint } from '../lib/neuralNetwork';

interface DatasetVisualizationProps {
  data: DataPoint[];
  decisionBoundary: { x: number; y: number; value: number }[];
  showBoundary: boolean;
}

export const DatasetVisualization = memo(function DatasetVisualization({
  data,
  decisionBoundary,
  showBoundary,
}: DatasetVisualizationProps) {
  const width = 400;
  const height = 400;
  const padding = 30;

  // Generate heatmap for decision boundary
  const heatmapPixels = useMemo(() => {
    if (!showBoundary || decisionBoundary.length === 0) return [];

    const resolution = Math.sqrt(decisionBoundary.length);
    const cellWidth = (width - 2 * padding) / resolution;
    const cellHeight = (height - 2 * padding) / resolution;

    return decisionBoundary.map((point) => {
      const x = padding + point.x * (width - 2 * padding) - cellWidth / 2;
      const y = padding + (1 - point.y) * (height - 2 * padding) - cellHeight / 2;

      // Map value to color
      const intensity = Math.tanh(point.value * 2); // Normalize to [-1, 1]
      const r = intensity < 0 ? 239 : 34;
      const g = intensity < 0 ? 68 : 197;
      const b = intensity < 0 ? 68 : 94;
      const alpha = 0.15 + Math.abs(intensity) * 0.25;

      return {
        x,
        y,
        width: cellWidth + 1,
        height: cellHeight + 1,
        color: `rgba(${r}, ${g}, ${b}, ${alpha})`,
      };
    });
  }, [decisionBoundary, showBoundary, width, height, padding]);

  const toScreenX = (x: number) => padding + x * (width - 2 * padding);
  const toScreenY = (y: number) => padding + (1 - y) * (height - 2 * padding);

  return (
    <div className="bg-[hsl(var(--card))] rounded-xl border border-[hsl(var(--border))] p-4">
      <h3 className="text-sm font-medium text-[hsl(var(--foreground))] mb-3">Dataset & Decision Boundary</h3>
      <svg viewBox={`0 0 ${width} ${height}`} className="w-full bg-[hsl(222,47%,6%)] rounded-lg">
        {/* Grid lines */}
        <g stroke="hsl(var(--border))" strokeWidth={0.5} opacity={0.5}>
          {[0, 0.25, 0.5, 0.75, 1].map((v) => (
            <g key={v}>
              <line x1={toScreenX(v)} y1={padding} x2={toScreenX(v)} y2={height - padding} />
              <line x1={padding} y1={toScreenY(v)} x2={width - padding} y2={toScreenY(v)} />
            </g>
          ))}
        </g>

        {/* Decision boundary heatmap */}
        {showBoundary &&
          heatmapPixels.map((pixel, idx) => (
            <rect
              key={`heatmap-${idx}`}
              x={pixel.x}
              y={pixel.y}
              width={pixel.width}
              height={pixel.height}
              fill={pixel.color}
            />
          ))}

        {/* Data points */}
        {data.map((point, idx) => {
          const isClass1 = point.target[1] > point.target[0];
          return (
            <g key={`point-${idx}`}>
              <circle
                cx={toScreenX(point.input[0])}
                cy={toScreenY(point.input[1])}
                r={5}
                fill={isClass1 ? 'hsl(142, 76%, 46%)' : 'hsl(0, 84%, 60%)'}
                stroke="white"
                strokeWidth={1}
                opacity={0.9}
              />
            </g>
          );
        })}

        {/* Axis labels */}
        <text
          x={width / 2}
          y={height - 8}
          textAnchor="middle"
          fill="hsl(var(--muted-foreground))"
          fontSize={11}
        >
          X₁
        </text>
        <text
          x={12}
          y={height / 2}
          textAnchor="middle"
          fill="hsl(var(--muted-foreground))"
          fontSize={11}
          transform={`rotate(-90, 12, ${height / 2})`}
        >
          X₂
        </text>
      </svg>

      {/* Legend */}
      <div className="flex items-center justify-center gap-6 mt-3 text-xs text-[hsl(var(--muted-foreground))]">
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-[hsl(0,84%,60%)]" />
          <span>Class 0</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-[hsl(142,76%,46%)]" />
          <span>Class 1</span>
        </div>
        {showBoundary && (
          <div className="flex items-center gap-2">
            <div className="w-8 h-3 rounded" style={{ background: 'linear-gradient(90deg, hsl(0,84%,60%) 0%, hsl(217,33%,17%) 50%, hsl(142,76%,46%) 100%)' }} />
            <span>Boundary</span>
          </div>
        )}
      </div>
    </div>
  );
});
