import React, { useState, useEffect, useCallback } from 'react';
import { analyzeTrainingPerformance } from '../services/geminiService';
import { Loader2, X, Sparkles, RefreshCw } from 'lucide-react';

interface Props {
  data: { episode: number; reward: number }[];
  onClose: () => void;
}

export const GeminiAnalysis: React.FC<Props> = ({ data, onClose }) => {
  const [analysis, setAnalysis] = useState<string>('');
  const [loading, setLoading] = useState(true);

  const fetchAnalysis = useCallback(async () => {
      setLoading(true);
      if (data.length === 0) {
        setAnalysis("No training data available yet. Run the simulation for a few episodes!");
        setLoading(false);
        return;
      }
      
      const result = await analyzeTrainingPerformance(data);
      setAnalysis(result);
      setLoading(false);
  }, [data]);

  // Only fetch on mount to prevent hitting rate limits when training runs fast
  useEffect(() => {
    fetchAnalysis();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className="mt-4 bg-slate-900 border border-slate-700 rounded-lg p-4 relative animate-in fade-in slide-in-from-top-2">
      <div className="absolute top-2 right-2 flex gap-2">
        {!loading && (
          <button 
            onClick={fetchAnalysis}
            className="text-slate-500 hover:text-indigo-400 transition-colors"
            title="Refresh Analysis"
          >
            <RefreshCw className="w-4 h-4" />
          </button>
        )}
        <button 
          onClick={onClose}
          className="text-slate-500 hover:text-slate-300"
        >
          <X className="w-4 h-4" />
        </button>
      </div>

      {loading ? (
        <div className="flex items-center gap-3 text-indigo-400 py-4">
          <Loader2 className="w-5 h-5 animate-spin" />
          <span className="text-sm font-medium">Gemini is analyzing your agent's neural pathways...</span>
        </div>
      ) : (
        <div className="flex gap-3">
            <div className="mt-1">
                <Sparkles className="w-5 h-5 text-indigo-400" />
            </div>
            <div>
                <h4 className="text-sm font-bold text-indigo-300 mb-1">Coach's Insight</h4>
                <p className="text-sm text-slate-300 leading-relaxed">
                    {analysis}
                </p>
            </div>
        </div>
      )}
    </div>
  );
};