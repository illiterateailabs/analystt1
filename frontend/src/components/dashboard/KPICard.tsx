import React from 'react';
import { ArrowUp, ArrowDown, Minus, AlertCircle, CheckCircle, Info, TrendingUp, TrendingDown } from 'lucide-react';
import { cn } from '@/lib/utils';

export type KPICardVariant = 'value' | 'trend' | 'progress';
export type KPICardColorScheme = 'neutral' | 'positive' | 'negative' | 'warning';
export type KPICardTrendType = 'up' | 'down' | 'neutral';

interface KPICardProps {
  /**
   * The title of the KPI card.
   */
  title: string;
  /**
   * The main value to display.
   */
  value: string | number;
  /**
   * The unit of the value (e.g., "$", "%", "transactions").
   */
  unit?: string;
  /**
   * Optional description or subtitle for the card.
   */
  description?: string;
  /**
   * Defines the visual variant of the card.
   * 'value': Displays a single large value.
   * 'trend': Displays a value with a trend indicator.
   * 'progress': Displays a value with a progress bar.
   * @default 'value'
   */
  variant?: KPICardVariant;
  /**
   * Defines the color scheme of the card, influencing background and text colors.
   * 'neutral': Default gray/white.
   * 'positive': Greenish tones (e.g., for good performance).
   * 'negative': Reddish tones (e.g., for bad performance/high risk).
   * 'warning': Yellow/orange tones (e.g., for moderate risk).
   * @default 'neutral'
   */
  colorScheme?: KPICardColorScheme;
  /**
   * Optional trend data for 'trend' variant.
   * `type`: 'up', 'down', or 'neutral'.
   * `percentage`: The percentage change (e.g., 5.2 for 5.2%).
   */
  trend?: {
    type: KPICardTrendType;
    percentage: number;
  };
  /**
   * Optional progress data for 'progress' variant.
   * `current`: Current progress value.
   * `max`: Maximum progress value.
   * `label`: Optional label for the progress bar (e.g., "75% complete").
   */
  progress?: {
    current: number;
    max: number;
    label?: string;
  };
  /**
   * Optional icon to display on the card.
   */
  icon?: React.ElementType;
  /**
   * Callback function when the card is clicked (for drill-down).
   */
  onClick?: () => void;
  /**
   * Additional CSS classes for the card container.
   */
  className?: string;
}

const KPICard: React.FC<KPICardProps> = ({
  title,
  value,
  unit,
  description,
  variant = 'value',
  colorScheme = 'neutral',
  trend,
  progress,
  icon: Icon,
  onClick,
  className,
}) => {
  const baseClasses = 'relative p-6 rounded-lg shadow-sm transition-all duration-200';
  const clickableClasses = onClick ? 'cursor-pointer hover:shadow-md' : '';

  const getColorClasses = (scheme: KPICardColorScheme) => {
    switch (scheme) {
      case 'positive':
        return 'bg-green-50 dark:bg-green-950 text-green-800 dark:text-green-200 border border-green-200 dark:border-green-800';
      case 'negative':
        return 'bg-red-50 dark:bg-red-950 text-red-800 dark:text-red-200 border border-red-200 dark:border-red-800';
      case 'warning':
        return 'bg-yellow-50 dark:bg-yellow-950 text-yellow-800 dark:text-yellow-200 border border-yellow-200 dark:border-yellow-800';
      case 'neutral':
      default:
        return 'bg-white dark:bg-gray-900 text-gray-900 dark:text-gray-100 border border-gray-200 dark:border-gray-800';
    }
  };

  const getTrendIcon = (type: KPICardTrendType) => {
    switch (type) {
      case 'up':
        return <ArrowUp className="h-4 w-4" />;
      case 'down':
        return <ArrowDown className="h-4 w-4" />;
      case 'neutral':
      default:
        return <Minus className="h-4 w-4" />;
    }
  };

  const getTrendColor = (type: KPICardTrendType, scheme: KPICardColorScheme) => {
    if (scheme === 'positive') {
      return type === 'up' ? 'text-green-600' : type === 'down' ? 'text-red-600' : 'text-gray-500';
    }
    if (scheme === 'negative') {
      return type === 'up' ? 'text-red-600' : type === 'down' ? 'text-green-600' : 'text-gray-500';
    }
    return type === 'up' ? 'text-green-600' : type === 'down' ? 'text-red-600' : 'text-gray-500';
  };

  const renderContent = () => {
    switch (variant) {
      case 'trend':
        return (
          <>
            <div className="flex items-baseline justify-between">
              <div className="flex items-baseline">
                <span className="text-4xl font-bold">{value}</span>
                {unit && <span className="ml-2 text-lg font-medium text-gray-500 dark:text-gray-400">{unit}</span>}
              </div>
              {trend && (
                <div className={cn('flex items-center text-sm font-medium', getTrendColor(trend.type, colorScheme))}>
                  {getTrendIcon(trend.type)}
                  <span>{trend.percentage.toFixed(1)}%</span>
                </div>
              )}
            </div>
            {description && <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">{description}</p>}
          </>
        );
      case 'progress':
        if (!progress) return null;
        const progressPercentage = Math.round((progress.current / progress.max) * 100);
        return (
          <>
            <div className="flex items-baseline justify-between">
              <div className="flex items-baseline">
                <span className="text-4xl font-bold">{value}</span>
                {unit && <span className="ml-2 text-lg font-medium text-gray-500 dark:text-gray-400">{unit}</span>}
              </div>
              <span className="text-sm font-medium text-gray-600 dark:text-gray-300">
                {progress.label || `${progressPercentage}%`}
              </span>
            </div>
            <div className="mt-4 w-full bg-gray-200 rounded-full h-2.5 dark:bg-gray-700">
              <div
                className={cn(
                  'h-2.5 rounded-full',
                  colorScheme === 'positive' && 'bg-green-600',
                  colorScheme === 'negative' && 'bg-red-600',
                  colorScheme === 'warning' && 'bg-yellow-500',
                  colorScheme === 'neutral' && 'bg-blue-600'
                )}
                style={{ width: `${progressPercentage}%` }}
              ></div>
            </div>
            {description && <p className="mt-3 text-sm text-gray-500 dark:text-gray-400">{description}</p>}
          </>
        );
      case 'value':
      default:
        return (
          <>
            <div className="flex items-baseline">
              <span className="text-4xl font-bold">{value}</span>
              {unit && <span className="ml-2 text-lg font-medium text-gray-500 dark:text-gray-400">{unit}</span>}
            </div>
            {description && <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">{description}</p>}
          </>
        );
    }
  };

  return (
    <div
      className={cn(baseClasses, getColorClasses(colorScheme), clickableClasses, className)}
      onClick={onClick}
      role={onClick ? 'button' : undefined}
      tabIndex={onClick ? 0 : undefined}
    >
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-medium">{title}</h3>
        {Icon && <Icon className={cn('h-6 w-6', colorScheme === 'neutral' ? 'text-gray-500 dark:text-gray-400' : 'text-current opacity-75')} />}
      </div>
      {renderContent()}
    </div>
  );
};

export default KPICard;
