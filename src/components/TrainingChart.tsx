import { memo } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import type { TrainingHistory } from '../lib/neuralNetwork';

interface TrainingChartProps {
  history: TrainingHistory[];
}

export const TrainingChart = memo(function TrainingChart({ history }: TrainingChartProps) {
  const formattedData = history.map((h) => ({
    epoch: h.epoch,
    loss: Number(h.loss.toFixed(4)),
    accuracy: Number((h.accuracy * 100).toFixed(1)),
  }));

  return (
    <div className="bg-[hsl(var(--card))] rounded-xl border border-[hsl(var(--border))] p-4">
      <h3 className="text-sm font-medium text-[hsl(var(--foreground))] mb-3">Training Progress</h3>
      <div className="h-[200px]">
        {history.length === 0 ? (
          <div className="h-full flex items-center justify-center text-[hsl(var(--muted-foreground))] text-sm">
            Start training to see progress
          </div>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={formattedData}>
              <CartesianGrid
                strokeDasharray="3 3"
                stroke="hsl(217, 33%, 17%)"
                vertical={false}
              />
              <XAxis
                dataKey="epoch"
                stroke="hsl(215, 20%, 55%)"
                tick={{ fill: 'hsl(215, 20%, 55%)', fontSize: 11 }}
                tickLine={{ stroke: 'hsl(217, 33%, 17%)' }}
                label={{ value: 'Epoch', position: 'bottom', offset: -5, fill: 'hsl(215, 20%, 55%)', fontSize: 11 }}
              />
              <YAxis
                yAxisId="left"
                stroke="hsl(215, 20%, 55%)"
                tick={{ fill: 'hsl(215, 20%, 55%)', fontSize: 11 }}
                tickLine={{ stroke: 'hsl(217, 33%, 17%)' }}
                domain={[0, 'auto']}
                label={{ value: 'Loss', angle: -90, position: 'insideLeft', fill: 'hsl(215, 20%, 55%)', fontSize: 11 }}
              />
              <YAxis
                yAxisId="right"
                orientation="right"
                stroke="hsl(215, 20%, 55%)"
                tick={{ fill: 'hsl(215, 20%, 55%)', fontSize: 11 }}
                tickLine={{ stroke: 'hsl(217, 33%, 17%)' }}
                domain={[0, 100]}
                label={{ value: 'Accuracy %', angle: 90, position: 'insideRight', fill: 'hsl(215, 20%, 55%)', fontSize: 11 }}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: 'hsl(222, 47%, 8%)',
                  border: '1px solid hsl(217, 33%, 17%)',
                  borderRadius: '8px',
                  color: 'hsl(210, 40%, 98%)',
                }}
                labelStyle={{ color: 'hsl(215, 20%, 55%)' }}
              />
              <Legend
                wrapperStyle={{ paddingTop: '10px' }}
                formatter={(value) => <span style={{ color: 'hsl(215, 20%, 55%)' }}>{value}</span>}
              />
              <Line
                yAxisId="left"
                type="monotone"
                dataKey="loss"
                stroke="hsl(0, 84%, 60%)"
                strokeWidth={2}
                dot={false}
                name="Loss"
              />
              <Line
                yAxisId="right"
                type="monotone"
                dataKey="accuracy"
                stroke="hsl(142, 76%, 46%)"
                strokeWidth={2}
                dot={false}
                name="Accuracy"
              />
            </LineChart>
          </ResponsiveContainer>
        )}
      </div>
    </div>
  );
});
