import React, { useEffect, useState } from 'react';
import { Save, Trash2, Play, RefreshCw, HardDrive, BrainCircuit } from 'lucide-react';
import { PPOAgent } from '../services/ppo';

interface ModelListProps {
    onLoad: (modelName: string, mode: 'test' | 'train') => void;
    onDelete?: (modelName: string) => void;
    currentModel?: string;
    className?: string;
}

export const ModelList: React.FC<ModelListProps> = ({ onLoad, onDelete, currentModel, className }) => {
    const [models, setModels] = useState<string[]>([]);
    const [loading, setLoading] = useState(false);

    const fetchModels = async () => {
        setLoading(true);
        try {
            const list = await PPOAgent.listModels();
            setModels(list);
        } catch (err) {
            console.error("Failed to list models", err);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        fetchModels();
    }, []);

    const handleDelete = async (name: string) => {
        if (confirm(`Are you sure you want to delete model "${name}"?`)) {
            try {

                await PPOAgent.deleteModel(name);
                await fetchModels();
                if (onDelete) onDelete(name);
            } catch (err) {
                console.error("Failed to delete model", err);
            }
        }
    };

    return (
        <div className={`bg-slate-800/50 border border-slate-700 rounded-xl p-4 flex flex-col h-full ${className}`}>
            <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold flex items-center gap-2 text-slate-200">
                    <HardDrive className="w-5 h-5 text-indigo-400" />
                    Saved Models
                </h3>
                <button
                    onClick={fetchModels}
                    className="p-2 hover:bg-slate-700 rounded-lg transition-colors text-slate-400 hover:text-white"
                    title="Refresh List"
                >
                    <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
                </button>
            </div>

            <div className="flex-1 overflow-y-auto space-y-2 pr-2 custom-scrollbar">
                {models.length === 0 ? (
                    <div className="text-center text-slate-500 py-8 text-sm italic">
                        No saved models found.<br />Train an agent and save it!
                    </div>
                ) : (
                    models.map(name => (
                        <div
                            key={name}
                            className={`p-3 rounded-lg border transition-all group flex items-center justify-between
                ${currentModel === name
                                    ? 'bg-indigo-500/20 border-indigo-500/50 shadow-lg shadow-indigo-500/10'
                                    : 'bg-slate-900/50 border-slate-800 hover:border-slate-600 hover:bg-slate-800'}
              `}
                        >
                            <div className="flex flex-col min-w-0">
                                <span className="font-medium text-sm text-slate-200 truncate" title={name}>
                                    {name}
                                </span>
                                <span className="text-[10px] text-slate-500">
                                    PPO Agent
                                </span>
                            </div>

                            <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                                <button
                                    onClick={() => onLoad(name, 'test')}
                                    className="p-1.5 rounded-md hover:bg-blue-500/20 text-slate-400 hover:text-blue-400 transition-colors"
                                    title="Test Model (Inference Only)"
                                >
                                    <Play className="w-4 h-4" />
                                </button>
                                <button
                                    onClick={() => onLoad(name, 'train')}
                                    className="p-1.5 rounded-md hover:bg-emerald-500/20 text-slate-400 hover:text-emerald-400 transition-colors"
                                    title="Resume Training"
                                >
                                    <BrainCircuit className="w-4 h-4" />
                                </button>
                                <button
                                    onClick={() => handleDelete(name)}
                                    className="p-1.5 rounded-md hover:bg-red-500/20 text-slate-400 hover:text-red-400 transition-colors"
                                    title="Delete"
                                >
                                    <Trash2 className="w-4 h-4" />
                                </button>
                            </div>
                        </div>
                    ))
                )}
            </div>
        </div>
    );
};
