import * as tf from '@tensorflow/tfjs';

// Hyperparameters
const GAMMA = 0.99;
const GAE_LAMBDA = 0.95;
const CLIP_RATIO = 0.2;
const LEARNING_RATE = 0.0003; // Lower LR is more stable for PPO
const EPOCHS = 10; // 3-10 epochs is standard. 500 causes overfitting/collapse.
const BATCH_SIZE = 64; // Not strictly used in this full-batch implementation but good for ref

export class PPOAgent {
  actor: tf.LayersModel;
  critic: tf.LayersModel;
  actorOptimizer: tf.Optimizer;
  criticOptimizer: tf.Optimizer;

  constructor(inputShape: number, outputShape: number) {
    // Use separate optimizers to avoid variable state collisions (moments) 
    // especially when layer shapes differ but might share internal identifiers or if using same optimizer instance.
    this.actorOptimizer = tf.train.adam(LEARNING_RATE);
    this.criticOptimizer = tf.train.adam(LEARNING_RATE);

    // Actor Network (Policy)
    const actorInput = tf.input({ shape: [inputShape] });
    const actorDense1 = tf.layers.dense({ units: 74, activation: 'relu' }).apply(actorInput);
    const actorDense2 = tf.layers.dense({ units: 74, activation: 'relu' }).apply(actorDense1);
    const actorDense3 = tf.layers.dense({ units: 74, activation: 'relu' }).apply(actorDense2);
    const actorOutput = tf.layers.dense({ units: outputShape, activation: 'softmax' }).apply(actorDense3) as tf.SymbolicTensor;
    this.actor = tf.model({ inputs: actorInput, outputs: actorOutput });

    // Critic Network (Value)
    const criticInput = tf.input({ shape: [inputShape] });
    const criticDense1 = tf.layers.dense({ units: 74, activation: 'relu' }).apply(criticInput);
    const criticDense2 = tf.layers.dense({ units: 74, activation: 'relu' }).apply(criticDense1);
    const criticDense3 = tf.layers.dense({ units: 74, activation: 'relu' }).apply(criticDense2);
    const criticOutput = tf.layers.dense({ units: 1, activation: 'linear' }).apply(criticDense3) as tf.SymbolicTensor;
    this.critic = tf.model({ inputs: criticInput, outputs: criticOutput });
  }

  getAction(state: number[]): number {
    return tf.tidy(() => {
      const stateTensor = tf.tensor2d([state]);
      const probsTensor = this.actor.predict(stateTensor) as tf.Tensor;
      // Sample action from probability distribution
      const probs = probsTensor.dataSync();
      return Math.random() < probs[0] ? 0 : 1;
    });
  }

  dispose() {
    this.actor.dispose();
    this.critic.dispose();
    // Optimizers don't have a public dispose in all versions, 
    // but variables they track are cleaned up when models are disposed ideally.
  }

  // Generalized Advantage Estimation (GAE)
  calculateAdvantages(rewards: number[], values: number[], nextValues: number[], dones: boolean[]) {
    const advantages = new Array(rewards.length).fill(0);
    let lastAdvantage = 0;

    for (let t = rewards.length - 1; t >= 0; t--) {
      const mask = dones[t] ? 0 : 1;
      const nextValue = nextValues[t];
      const delta = rewards[t] + GAMMA * nextValue * mask - values[t];

      lastAdvantage = delta + GAMMA * GAE_LAMBDA * mask * lastAdvantage;
      advantages[t] = lastAdvantage;
    }
    return advantages;
  }

  async train(states: number[][], actions: number[], rewards: number[], nextStates: number[][], dones: boolean[]) {
    // Prepare Data
    const stateTensor = tf.tensor2d(states);
    const nextStateTensor = tf.tensor2d(nextStates);

    // Get Values for current and next states
    // Note: predict() inside async context is fine, but for heavy training loops 
    // we want to be careful with memory. Here we dispose manually.
    const valuesTensor = this.critic.predict(stateTensor) as tf.Tensor;
    const nextValuesTensor = this.critic.predict(nextStateTensor) as tf.Tensor;

    const values = await valuesTensor.data();
    const nextValues = await nextValuesTensor.data();

    // Compute Advantages & Targets
    const advantagesArr = this.calculateAdvantages(
      rewards,
      Array.from(values),
      Array.from(nextValues),
      dones
    );

    // Compute Returns: Return = Advantage + Value
    const returnsArr = advantagesArr.map((adv, i) => adv + values[i]);

    const advantageTensor = tf.tensor1d(advantagesArr);
    const targetValueTensor = tf.tensor1d(returnsArr);
    const actionTensor = tf.tensor1d(actions, 'int32');

    // Make one-hot actions for probability selection
    const oneHotActions = tf.oneHot(actionTensor, 2);

    // Old Probabilities (for PPO ratio) - computed once before updates
    // We detach this from the graph (it's just a constant tensor for the loop)
    const oldProbsTensor = this.actor.predict(stateTensor) as tf.Tensor;
    // We only care about the probability of the action that was actually taken
    const oldActionProbs = tf.mul(oneHotActions, oldProbsTensor).sum(1);
    // Keep oldActionProbs as a tensor, but we must ensure it's not part of the trainable graph in the loop?
    // In TF.js, tensors are just data unless created inside a tape-aware function. 
    // Since this is outside minimize(), it's a constant.

    // Optimization Loop
    for (let i = 0; i < EPOCHS; i++) {
      // Optimize Actor
      this.actorOptimizer.minimize(() => {
        const newProbsTensor = this.actor.predict(stateTensor) as tf.Tensor;
        const newActionProbs = tf.mul(oneHotActions, newProbsTensor).sum(1);

        const ratio = tf.div(newActionProbs, oldActionProbs);
        const clippedRatio = tf.clipByValue(ratio, 1 - CLIP_RATIO, 1 + CLIP_RATIO);

        const surr1 = tf.mul(ratio, advantageTensor);
        const surr2 = tf.mul(clippedRatio, advantageTensor);

        // Maximize objective = minimize negative
        // PPO Objective: E[ min(r*A, clip(r, 1-e, 1+e)*A) ]
        const loss = tf.mean(tf.minimum(surr1, surr2)).mul(-1);
        return loss as tf.Scalar;
      });

      // Optimize Critic
      this.criticOptimizer.minimize(() => {
        const valuePreds = this.critic.predict(stateTensor) as tf.Tensor;
        // Value Loss: MSE between predicted value and actual returns
        return tf.mean(tf.squaredDifference(valuePreds.reshape([-1]), targetValueTensor)) as tf.Scalar;
      });
    }

    // Cleanup
    tf.dispose([
      stateTensor, nextStateTensor, valuesTensor, nextValuesTensor,
      advantageTensor, targetValueTensor, actionTensor,
      oneHotActions, oldProbsTensor, oldActionProbs
    ]);
  }

  // --- Persistence ---

  async save(modelName: string): Promise<void> {
    // Save Actor and Critic separately
    // We use a custom scheme 'indexeddb://' which TF.js supports 
    await this.actor.save(`indexeddb://actor-${modelName}`);
    await this.critic.save(`indexeddb://critic-${modelName}`);
    console.log(`Model ${modelName} saved to IndexedDB`);
  }

  async load(modelName: string): Promise<void> {
    try {
      const loadedActor = await tf.loadLayersModel(`indexeddb://actor-${modelName}`);
      const loadedCritic = await tf.loadLayersModel(`indexeddb://critic-${modelName}`);

      // Dispose old models
      this.actor.dispose();
      this.critic.dispose();

      // Assign new models
      this.actor = loadedActor;
      this.critic = loadedCritic;

      // Re-compile/Optimizer setup is tricky because loading doesn't always restore optimizers perfectly 
      // or we might want to continue training with fresh optimizers but same weights.
      // For this app, we'll re-initialize optimizers to be safe or just keep using the existing ones (which are separate objects).
      // Actually, if we want to continue training, we should probably keep the optimizer configuration.
      // But since we just swapped the *model* objects, the old optimizers are now disconnected from these new models (?)
      // TF.js Optimizers are not strictly bound to the model instance until minimize() is called with that model's weights.
      // So we might be okay just keeping the old optimizer instances. 
      // HOWEVER, `minimize` takes a function.

      console.log(`Model ${modelName} loaded from IndexedDB`);

    } catch (err) {
      console.error(`Failed to load model ${modelName}:`, err);
      throw err;
    }
  }

  static async listModels(): Promise<string[]> {
    const models = await tf.io.listModels();
    // Filter keys that start with "indexeddb://actor-" to get unique model names
    const keys = Object.keys(models).filter(k => k.startsWith('indexeddb://actor-'));
    return keys.map(k => k.replace('indexeddb://actor-', ''));
  }

  static async deleteModel(modelName: string): Promise<void> {
    await tf.io.removeModel(`indexeddb://actor-${modelName}`);
    await tf.io.removeModel(`indexeddb://critic-${modelName}`);
  }
}