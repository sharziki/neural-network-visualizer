import { memo } from 'react';
import { Brain, Target, TrendingDown, Clock } from 'lucide-react';

interface StatsPanelProps {
  epoch: number;
  loss: number;
  accuracy: number;
  isTraining: boolean;
  architecture: number[];
}

interface StatCardProps {
  label: string;
  value: string | number;
  icon: React.ReactNode;
  color: string;
}

function StatCard({ label, value, icon, color }: StatCardProps) {
  return (
    <div className="bg-[hsl(var(--card))] rounded-xl border border-[hsl(var(--border))] p-4">
      <div className="flex items-center gap-3">
        <div className={`p-2 rounded-lg bg-[hsl(var(--secondary))] ${color}`}>{icon}</div>
        <div>
          <p className="text-xs text-[hsl(var(--muted-foreground))] uppercase tracking-wider">{label}</p>
          <p className={`text-xl font-bold ${color}`}>{value}</p>
        </div>
      </div>
    </div>
  );
}

export const StatsPanel = memo(function StatsPanel({
  epoch,
  loss,
  accuracy,
  isTraining,
  architecture,
}: StatsPanelProps) {
  const totalParams = architecture.reduce((sum, neurons, i) => {
    if (i === architecture.length - 1) return sum;
    const nextNeurons = architecture[i + 1];
    return sum + neurons * nextNeurons + nextNeurons; // weights + biases
  }, 0);

  return (
    <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
      <StatCard
        label="Epoch"
        value={isTraining ? `${epoch}` : epoch || '—'}
        icon={<Clock className="w-5 h-5" />}
        color="text-[hsl(var(--primary))]"
      />
      <StatCard
        label="Loss"
        value={loss > 0 ? loss.toFixed(4) : '—'}
        icon={<TrendingDown className="w-5 h-5" />}
        color="text-[hsl(var(--destructive))]"
      />
      <StatCard
        label="Accuracy"
        value={accuracy > 0 ? `${(accuracy * 100).toFixed(1)}%` : '—'}
        icon={<Target className="w-5 h-5" />}
        color="text-[hsl(142,76%,46%)]"
      />
      <StatCard
        label="Parameters"
        value={totalParams.toLocaleString()}
        icon={<Brain className="w-5 h-5" />}
        color="text-[hsl(var(--muted-foreground))]"
      />
    </div>
  );
});
