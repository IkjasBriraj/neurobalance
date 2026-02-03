import React, { useState, useEffect, useRef, useCallback } from 'react';
import * as tf from '@tensorflow/tfjs';
import {
  Play,
  Pause,
  RotateCcw,
  BrainCircuit,
  Activity,
  Zap,
  MessageSquareQuote,
  Gamepad2,
  Hand,
  Save
} from 'lucide-react';
import { PPOAgent } from './services/ppo';
import { CartPole, State } from './services/cartpole';
import { analyzeTrainingPerformance } from './services/geminiService';
import { SimulationCanvas } from './components/SimulationCanvas';
import { TrainingChart } from './components/TrainingChart';
import { GeminiAnalysis } from './components/GeminiAnalysis';
import { ModelList } from './components/ModelList';

// Constants
const FPS = 60;
const MAX_STEPS_PER_EPISODE = 500;
const TRAIN_EVERY_EPISODES = 1;

const App: React.FC = () => {
  // --- State ---
  const [isPlaying, setIsPlaying] = useState(false);
  const [episode, setEpisode] = useState(0);
  const [step, setStep] = useState(0);
  const [reward, setReward] = useState(0);
  const [highScore, setHighScore] = useState(0);
  const [rewardHistory, setRewardHistory] = useState<{ episode: number; reward: number }[]>([]);
  const [isTraining, setIsTraining] = useState(false);
  const [simSpeed, setSimSpeed] = useState(1);
  const [showAnalysis, setShowAnalysis] = useState(false);
  const [isManualMode, setIsManualMode] = useState(false);
  const [isInteractionEnabled, setIsInteractionEnabled] = useState(false);
  const [currentModel, setCurrentModel] = useState<string | undefined>(undefined);
  const [refreshModelsTrigger, setRefreshModelsTrigger] = useState(0);
  const [trainingEnabled, setTrainingEnabled] = useState(true);

  // --- Refs for Simulation Loop ---
  const requestRef = useRef<number>();
  const cartPoleRef = useRef(new CartPole());
  const agentRef = useRef<PPOAgent | null>(null);
  const episodeRewardRef = useRef(0);
  const stepCountRef = useRef(0);
  const trainingDataRef = useRef<{
    states: number[][];
    actions: number[];
    rewards: number[];
    nextStates: number[][];
    dones: boolean[];
  }>({ states: [], actions: [], rewards: [], nextStates: [], dones: [] });

  // Track pressed keys for manual control
  const keysRef = useRef<{ [key: string]: boolean }>({});

  // --- Initialization ---
  useEffect(() => {
    const initTF = async () => {
      await tf.ready();
      agentRef.current = new PPOAgent(4, 2); // 4 inputs (state), 2 outputs (left/right)
      console.log('TF.js ready, PPO Agent initialized');
    };
    initTF();

    return () => {
      if (requestRef.current) cancelAnimationFrame(requestRef.current);
    };
  }, []);

  // --- Keyboard Listeners ---
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      keysRef.current[e.key] = true;
    };
    const handleKeyUp = (e: KeyboardEvent) => {
      keysRef.current[e.key] = false;
    };

    window.addEventListener('keydown', handleKeyDown);
    window.addEventListener('keyup', handleKeyUp);

    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('keyup', handleKeyUp);
    };
  }, []);

  // --- Simulation Logic ---
  const stepSimulation = useCallback(async () => {
    if (!agentRef.current || isTraining) return;

    const env = cartPoleRef.current;
    const agent = agentRef.current;

    // 1. Get Action
    const stateArr = env.getStateArray();
    let action: number;

    // Human-in-the-loop: Check for manual override
    if (isManualMode && (keysRef.current['ArrowLeft'] || keysRef.current['ArrowRight'])) {
      // If Manual Mode is ON and a key is pressed, use the key
      action = keysRef.current['ArrowLeft'] ? 0 : 1;
    } else {
      // Otherwise use the Agent's brain
      action = agent.getAction(stateArr);
    }

    // 2. Step Physics
    const { done: envDone, reward: stepReward } = env.step(action === 0 ? 'left' : 'right');

    // Force done if max steps reached
    const done = envDone || stepCountRef.current >= MAX_STEPS_PER_EPISODE;

    // 3. Store transition for training ONLY if training is enabled
    // If training is DISABLED (Test Mode), we just run inference and don't collect data/train.
    if (trainingEnabled && !isManualMode) {
      const nextStateArr = env.getStateArray();

      trainingDataRef.current.states.push(stateArr);
      trainingDataRef.current.actions.push(action);
      trainingDataRef.current.rewards.push(stepReward);
      trainingDataRef.current.nextStates.push(nextStateArr);
      trainingDataRef.current.dones.push(done);
    }

    episodeRewardRef.current += stepReward;
    stepCountRef.current += 1;

    // Update UI State (throttled slightly by nature of React, but we do it every frame here for smoothness)
    setStep(stepCountRef.current);
    setReward(episodeRewardRef.current);

    if (done) {
      // Episode Complete
      const finalReward = episodeRewardRef.current;
      setRewardHistory(prev => [...prev, { episode: episode + 1, reward: finalReward }]);
      setHighScore(prev => Math.max(prev, finalReward));
      setEpisode(prev => prev + 1);

      // Reset Environment
      env.reset();
      episodeRewardRef.current = 0;
      stepCountRef.current = 0;

      // Check if we need to train
      if (trainingEnabled && !isManualMode && (episode + 1) % TRAIN_EVERY_EPISODES === 0) {
        cancelAnimationFrame(requestRef.current!);
        setIsTraining(true);
        setIsPlaying(false);

        // Run training in a timeout to allow UI to update to "Training..." state
        setTimeout(async () => {
          const data = trainingDataRef.current;
          await agent.train(data.states, data.actions, data.rewards, data.nextStates, data.dones);

          // Clear buffer
          trainingDataRef.current = { states: [], actions: [], rewards: [], nextStates: [], dones: [] };

          setIsTraining(false);
          setIsPlaying(true); // Resume automatically
        }, 100);
        return; // Exit loop for now
      }
    }

    // Continue Loop
    if (isPlaying && !done) {
      // Adjust speed by calling multiple steps or delaying
      // For simplicity in this demo, we just rely on RAF (approx 60fps)
      // To speed up, we could run the physics loop multiple times per frame.
    }

  }, [episode, isPlaying, isTraining, isManualMode, trainingEnabled]);

  // Animation Loop Handler
  const animate = useCallback(() => {
    if (isPlaying && !isTraining) {
      // Run physics loop N times based on speed
      // Force 1x speed if in manual mode to make it playable
      const speed = isManualMode ? 1 : simSpeed;

      for (let i = 0; i < speed; i++) {
        stepSimulation();
        // If a step caused training pause, break immediately
        if (stepCountRef.current === 0 && episodeRewardRef.current === 0) break;
      }
      requestRef.current = requestAnimationFrame(animate);
    }
  }, [isPlaying, isTraining, simSpeed, stepSimulation, isManualMode]);

  useEffect(() => {
    if (isPlaying && !isTraining) {
      requestRef.current = requestAnimationFrame(animate);
    }
    return () => {
      if (requestRef.current) cancelAnimationFrame(requestRef.current);
    };
  }, [isPlaying, isTraining, animate]);


  // --- Handlers ---
  const togglePlay = () => setIsPlaying(!isPlaying);

  const toggleManualMode = () => {
    setIsManualMode(!isManualMode);
    // Reset speed to 1x if enabling manual mode
    if (!isManualMode) setSimSpeed(1);
  };

  const toggleInteractionMode = () => {
    setIsInteractionEnabled(!isInteractionEnabled);
  };

  const resetSim = () => {
    setIsPlaying(false);
    cartPoleRef.current.reset();
    setEpisode(0);
    setStep(0);
    setReward(0);
    setRewardHistory([]);
    setHighScore(0);
    episodeRewardRef.current = 0;
    stepCountRef.current = 0;
    trainingDataRef.current = { states: [], actions: [], rewards: [], nextStates: [], dones: [] };
    // Reset agent
    if (agentRef.current) {
      agentRef.current.dispose();
      agentRef.current = new PPOAgent(4, 2);
    }
    setCurrentModel(undefined);
  };

  const handleSaveModel = async () => {
    if (!agentRef.current) return;

    // Pause first
    const wasPlaying = isPlaying;
    setIsPlaying(false);

    const name = prompt("Enter a name for this model:", `model-${episode}`);
    if (name) {
      try {
        await agentRef.current.save(name);
        setRefreshModelsTrigger(prev => prev + 1); // Trigger refresh in list
        alert("Model saved successfully!");
      } catch (err) {
        console.error(err);
        alert("Failed to save model.");
      }
    }

    // Optional: Resume if it was playing, but usually safer to stay paused
    // setIsPlaying(wasPlaying);
  };

  const handleLoadModel = async (name: string, mode: 'test' | 'train') => {
    if (!agentRef.current) return;

    setIsPlaying(false);
    try {
      await agentRef.current.load(name);
      setCurrentModel(name);

      if (mode === 'test') {
        setTrainingEnabled(false);
        alert(`Model "${name}" loaded in TEST Mode (Training Disabled).`);
      } else {
        setTrainingEnabled(true);
        alert(`Model "${name}" loaded. Resuming training...`);
      }

      // Reset stats for fresh evaluation of this model?
      // For resume, maybe we keep episode count? 
      // But "Resume from saved checkpoint" usually implies the state *at that time* isn't fully saved (like step count), just weights.
      // So resetting episode stats is cleaner to track "how well it does now".
      setEpisode(0);
      setStep(0);
      setReward(0);
      setRewardHistory([]);
      episodeRewardRef.current = 0;
      stepCountRef.current = 0;
      cartPoleRef.current.reset();

    } catch (err) {
      console.error(err);
      alert(`Failed to load model "${name}".`);
    }
  };

  return (
    <div className="min-h-screen bg-slate-900 text-slate-100 font-sans selection:bg-indigo-500 selection:text-white">
      {/* Header */}
      <header className="border-b border-slate-800 bg-slate-900/50 backdrop-blur-md sticky top-0 z-20">
        <div className="max-w-[1600px] mx-auto px-4 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-indigo-600 rounded-lg shadow-lg shadow-indigo-500/20">
              <BrainCircuit className="w-6 h-6 text-white" />
            </div>
            <h1 className="text-xl font-bold bg-gradient-to-r from-indigo-400 to-cyan-400 bg-clip-text text-transparent">
              NeuroBalance PPO
            </h1>
          </div>
          <div className="flex items-center gap-4 text-sm text-slate-400">
            <span className="flex items-center gap-2">
              <Activity className="w-4 h-4 text-emerald-400" />
              TF.js Backend Active
            </span>
          </div>
        </div>
      </header>

      <main className="max-w-[1600px] mx-auto px-4 py-8 grid grid-cols-1 lg:grid-cols-4 gap-6">

        {/* Leftmost Column: Model List */}
        <div className="lg:col-span-1 h-[600px] lg:h-auto">
          <ModelList
            key={refreshModelsTrigger}
            onLoad={handleLoadModel}
            currentModel={currentModel}
            className="h-full"
          />
        </div>

        {/* Center Column: Simulation & Controls */}
        <div className="lg:col-span-2 space-y-6">
          {/* Canvas Container */}
          <div className="relative rounded-2xl overflow-hidden bg-slate-950 border border-slate-800 shadow-2xl shadow-black/50 aspect-video group">
            <div className="absolute inset-0 z-0">
              <SimulationCanvas
                cartPole={cartPoleRef.current}
                interactionEnabled={isInteractionEnabled}
              />
            </div>

            {/* Overlay Stats */}
            <div className="absolute top-4 left-4 flex flex-col gap-2 z-10 pointer-events-none">
              <div className="bg-slate-900/80 backdrop-blur border border-slate-700 rounded-lg p-3 text-xs font-mono shadow-xl">
                <div className="text-slate-400">EPISODE</div>
                <div className="text-xl font-bold text-white">{episode}</div>
              </div>
              <div className="bg-slate-900/80 backdrop-blur border border-slate-700 rounded-lg p-3 text-xs font-mono shadow-xl">
                <div className="text-slate-400">REWARD</div>
                <div className={`text-xl font-bold ${reward > 100 ? 'text-emerald-400' : 'text-white'}`}>
                  {Math.floor(reward)}
                </div>
              </div>
            </div>

            {/* Manual Mode Indicator */}
            {isManualMode && (
              <div className="absolute top-4 right-4 z-10 animate-pulse pointer-events-none">
                <div className="bg-pink-500/90 backdrop-blur border border-pink-400/50 rounded-lg px-4 py-2 flex items-center gap-3 shadow-xl">
                  <Gamepad2 className="w-5 h-5 text-white" />
                  <div>
                    <div className="text-xs font-bold text-white tracking-wider">MANUAL CONTROL</div>
                    <div className="text-[10px] text-pink-100">Use Arrow Keys</div>
                  </div>
                </div>
              </div>
            )}

            {/* Interaction Mode Indicator */}
            {isInteractionEnabled && !isManualMode && (
              <div className="absolute top-4 right-4 z-10 pointer-events-none">
                <div className="bg-amber-500/90 backdrop-blur border border-amber-400/50 rounded-lg px-4 py-2 flex items-center gap-3 shadow-xl animate-in fade-in slide-in-from-top-4">
                  <Hand className="w-5 h-5 text-white" />
                  <div>
                    <div className="text-xs font-bold text-white tracking-wider">HAND OF GOD</div>
                    <div className="text-[10px] text-amber-100">Drag to Apply Force</div>
                  </div>
                </div>
              </div>
            )}

            {/* Training Indicator */}
            {isTraining && (
              <div className="absolute inset-0 flex items-center justify-center bg-black/60 backdrop-blur-sm z-20 pointer-events-none">
                <div className="flex flex-col items-center gap-4 animate-pulse">
                  <BrainCircuit className="w-16 h-16 text-indigo-400" />
                  <h2 className="text-2xl font-bold text-indigo-400">Training Neural Network...</h2>
                  <p className="text-slate-300">Optimizing Policy & Value Heads</p>
                </div>
              </div>
            )}

            {/* Current Model Indicator */}
            {currentModel && (
              <div className="absolute bottom-4 left-4 z-10 pointer-events-none">
                <div className="bg-indigo-900/80 backdrop-blur border border-indigo-500/50 rounded px-2 py-1 text-[10px] font-mono text-indigo-200">
                  active_model: {currentModel}
                </div>
              </div>
            )}

          </div>

          {/* Controls */}
          <div className="bg-slate-800/50 rounded-xl p-4 border border-slate-700 flex flex-wrap items-center justify-between gap-4">
            <div className="flex items-center gap-2">
              <button
                onClick={togglePlay}
                disabled={isTraining}
                className={`flex items-center gap-2 px-6 py-3 rounded-lg font-semibold transition-all ${isPlaying
                  ? 'bg-amber-500/10 text-amber-500 border border-amber-500/50 hover:bg-amber-500/20'
                  : 'bg-emerald-500 text-white shadow-lg shadow-emerald-500/20 hover:bg-emerald-600'
                  } disabled:opacity-50 disabled:cursor-not-allowed`}
              >
                {isPlaying ? <Pause className="w-5 h-5" /> : <Play className="w-5 h-5" />}
                {isPlaying ? 'Pause' : 'Start'}
              </button>
              <button
                onClick={resetSim}
                disabled={isTraining}
                className="p-3 rounded-lg bg-slate-700 text-slate-300 hover:bg-slate-600 transition-colors disabled:opacity-50"
                title="Reset Environment"
              >
                <RotateCcw className="w-5 h-5" />
              </button>

              <div className="h-8 w-px bg-slate-700 mx-2"></div>

              <button
                onClick={handleSaveModel}
                className="flex items-center gap-2 px-4 py-3 rounded-lg font-semibold bg-indigo-600 text-white hover:bg-indigo-500 transition-colors shadow-lg shadow-indigo-500/20"
                title="Save current model"
              >
                <Save className="w-5 h-5" />
                <span>Save</span>
              </button>
            </div>

            {/* Interaction Controls */}
            <div className="flex items-center gap-2">
              {/* Training Mode Toggle */}
              <div className="flex items-center gap-2 bg-slate-900/50 p-1 rounded-lg border border-slate-700">
                <button
                  onClick={() => setTrainingEnabled(true)}
                  className={`px-3 py-2 rounded-md text-xs font-bold transition-all flex items-center gap-2 ${trainingEnabled
                      ? 'bg-emerald-600 text-white shadow'
                      : 'text-slate-400 hover:text-slate-200'
                    }`}
                  title="Enable Training (Learning)"
                >
                  <BrainCircuit className="w-4 h-4" />
                  <span className="hidden xl:inline">TRAIN</span>
                </button>
                <button
                  onClick={() => setTrainingEnabled(false)}
                  className={`px-3 py-2 rounded-md text-xs font-bold transition-all flex items-center gap-2 ${!trainingEnabled
                      ? 'bg-blue-600 text-white shadow'
                      : 'text-slate-400 hover:text-slate-200'
                    }`}
                  title="Disable Training (Test/Inference)"
                >
                  <Play className="w-4 h-4" />
                  <span className="hidden xl:inline">TEST</span>
                </button>
              </div>

              <button
                onClick={toggleManualMode}
                className={`flex items-center gap-2 px-4 py-3 rounded-lg font-semibold transition-all border ${isManualMode
                  ? 'bg-pink-500 text-white border-pink-400 shadow-lg shadow-pink-500/20'
                  : 'bg-slate-700 text-slate-300 border-slate-600 hover:bg-slate-600'
                  }`}
                title="Control the cart with arrow keys"
              >
                <Gamepad2 className="w-5 h-5" />
              </button>

              <button
                onClick={toggleInteractionMode}
                className={`flex items-center gap-2 px-4 py-3 rounded-lg font-semibold transition-all border ${isInteractionEnabled
                  ? 'bg-amber-500 text-white border-amber-400 shadow-lg shadow-amber-500/20'
                  : 'bg-slate-700 text-slate-300 border-slate-600 hover:bg-slate-600'
                  }`}
                title="Grab and pull the cart"
              >
                <Hand className="w-5 h-5" />
              </button>
            </div>

            <div className="flex items-center gap-4">
              <div className="flex flex-col gap-1">
                <label className="text-xs text-slate-400 font-medium uppercase tracking-wider">Speed</label>
                <div className="flex bg-slate-700 rounded-lg p-1">
                  {[1, 5, 10, 50].map(s => (
                    <button
                      key={s}
                      onClick={() => setSimSpeed(s)}
                      disabled={isManualMode && s > 1} // Disable high speeds in manual mode
                      className={`px-3 py-1 text-xs font-bold rounded ${simSpeed === s
                        ? 'bg-indigo-500 text-white shadow'
                        : isManualMode && s > 1
                          ? 'text-slate-600 cursor-not-allowed'
                          : 'text-slate-400 hover:text-white'
                        }`}
                    >
                      {s}x
                    </button>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Right Column: Analytics */}
        <div className="lg:col-span-1 space-y-6">
          {/* Stats Card */}
          <div className="bg-slate-800/50 border border-slate-700 rounded-xl p-6">
            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <Zap className="w-5 h-5 text-amber-400" />
              Performance Stats
            </h3>
            <div className="grid grid-cols-2 gap-4">
              <div className="bg-slate-900/50 p-4 rounded-lg border border-slate-700">
                <div className="text-slate-400 text-xs uppercase mb-1">High Score</div>
                <div className="text-2xl font-bold text-amber-400">{Math.floor(highScore)}</div>
              </div>
              <div className="bg-slate-900/50 p-4 rounded-lg border border-slate-700">
                <div className="text-slate-400 text-xs uppercase mb-1">Avg Reward (L10)</div>
                <div className="text-2xl font-bold text-indigo-400">
                  {rewardHistory.length > 0
                    ? Math.floor(rewardHistory.slice(-10).reduce((a, b) => a + b.reward, 0) / Math.min(rewardHistory.length, 10))
                    : 0}
                </div>
              </div>
            </div>
          </div>

          {/* Chart */}
          <div className="bg-slate-800/50 border border-slate-700 rounded-xl p-6">
            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <Activity className="w-5 h-5 text-indigo-400" />
              Learning Curve
            </h3>
            <div className="w-full h-64">
              <TrainingChart data={rewardHistory} />
            </div>
          </div>

          {/* Gemini Coach */}
          <div className="bg-gradient-to-br from-indigo-900/20 to-purple-900/20 border border-indigo-500/30 rounded-xl p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold flex items-center gap-2 text-indigo-300">
                <BrainCircuit className="w-5 h-5" />
                AI Coach Analysis
              </h3>
              <button
                onClick={() => setShowAnalysis(true)}
                className="text-xs bg-indigo-600 hover:bg-indigo-500 text-white px-3 py-1.5 rounded-full transition-colors flex items-center gap-1"
              >
                <MessageSquareQuote className="w-3 h-3" />
                Ask Gemini
              </button>
            </div>
            <p className="text-sm text-slate-400 mb-4">
              Get insights on your agent's learning progress directly from Google Gemini.
            </p>
            {showAnalysis && (
              <GeminiAnalysis
                data={rewardHistory}
                onClose={() => setShowAnalysis(false)}
              />
            )}
          </div>
        </div>

      </main>
    </div>
  );
};

export default App;