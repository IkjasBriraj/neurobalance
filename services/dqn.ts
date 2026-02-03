import * as tf from '@tensorflow/tfjs';

// Hyperparameters
const GAMMA = 0.99;
const LEARNING_RATE = 0.0001; // Lower LR for stability
const BATCH_SIZE = 64;
const BUFFER_SIZE = 50000;
const EPSILON_START = 1.0;
const EPSILON_END = 0.01;
const TARGET_UPDATE_FREQ = 10; // Update target model every N train calls

// Replay Buffer
class ReplayBuffer {
    buffer: {
        state: number[];
        action: number;
        reward: number;
        nextState: number[];
        done: boolean;
    }[];
    maxSize: number;

    constructor(maxSize: number) {
        this.buffer = [];
        this.maxSize = maxSize;
    }

    add(state: number[], action: number, reward: number, nextState: number[], done: boolean) {
        if (this.buffer.length >= this.maxSize) {
            this.buffer.shift();
        }
        this.buffer.push({ state, action, reward, nextState, done });
    }

    sample(batchSize: number) {
        const size = this.buffer.length;
        if (size === 0) return null;

        const samples = [];
        const indices = new Set<number>();

        // Random sampling without replacement (approx) or replacement
        while (samples.length < Math.min(size, batchSize)) {
            const index = Math.floor(Math.random() * size);
            if (!indices.has(index)) {
                indices.add(index);
                samples.push(this.buffer[index]);
            }
        }

        return {
            states: samples.map(s => s.state),
            actions: samples.map(s => s.action),
            rewards: samples.map(s => s.reward),
            nextStates: samples.map(s => s.nextState),
            dones: samples.map(s => s.done)
        };
    }

    size() {
        return this.buffer.length;
    }
}

export class DQNAgent {
    model: tf.LayersModel;
    targetModel: tf.LayersModel;
    optimizer: tf.Optimizer;
    replayBuffer: ReplayBuffer;
    epsilon: number;
    inputShape: number;
    outputShape: number;
    trainCalls: number = 0;

    constructor(inputShape: number, outputShape: number) {
        this.inputShape = inputShape;
        this.outputShape = outputShape;
        this.optimizer = tf.train.adam(LEARNING_RATE);
        this.replayBuffer = new ReplayBuffer(BUFFER_SIZE);
        this.epsilon = EPSILON_START;

        // Initialize Main Network
        this.model = this.createModel();
        // Initialize Target Network
        this.targetModel = this.createModel();
        this.updateTargetModel();
    }

    createModel(): tf.LayersModel {
        const input = tf.input({ shape: [this.inputShape] });
        const dense1 = tf.layers.dense({ units: 64, activation: 'relu' }).apply(input);
        const dense2 = tf.layers.dense({ units: 64, activation: 'relu' }).apply(dense1);
        const dense3 = tf.layers.dense({ units: 64, activation: 'relu' }).apply(dense2);
        const output = tf.layers.dense({ units: this.outputShape, activation: 'linear' }).apply(dense3) as tf.SymbolicTensor;
        return tf.model({ inputs: input, outputs: output });
    }

    updateTargetModel() {
        this.targetModel.setWeights(this.model.getWeights());
    }

    getAction(state: number[]): number {
        // Epsilon-greedy
        if (Math.random() < this.epsilon) {
            return Math.floor(Math.random() * this.outputShape);
        }

        return tf.tidy(() => {
            const stateTensor = tf.tensor2d([state]);
            const qValues = this.model.predict(stateTensor) as tf.Tensor;
            return qValues.argMax(1).dataSync()[0];
        });
    }

    // Decay epsilon
    updateEpsilon() {
        if (this.epsilon > EPSILON_END) {
            // Decay slower to allow more exploration
            this.epsilon = Math.max(EPSILON_END, this.epsilon * 0.995);
        }
    }

    dispose() {
        this.model.dispose();
        this.targetModel.dispose();
    }

    async train(states: number[][], actions: number[], rewards: number[], nextStates: number[][], dones: boolean[]) {
        // 1. Add new experiences to Replay Buffer
        for (let i = 0; i < states.length; i++) {
            this.replayBuffer.add(states[i], actions[i], rewards[i], nextStates[i], dones[i]);
        }

        // 2. Train on mini-batches
        // Conservative training to prevent overfitting on small/bad episodes
        const TRAIN_STEPS = 5;

        if (this.replayBuffer.size() < BATCH_SIZE) return;

        for (let i = 0; i < TRAIN_STEPS; i++) {
            const batch = this.replayBuffer.sample(BATCH_SIZE);
            if (!batch) break;

            tf.tidy(() => {
                const stateTensor = tf.tensor2d(batch.states);
                const actionTensor = tf.tensor1d(batch.actions, 'int32');
                const rewardTensor = tf.tensor1d(batch.rewards);
                const nextStateTensor = tf.tensor2d(batch.nextStates);
                const doneTensor = tf.tensor1d(batch.dones.map(d => d ? 1 : 0)); // 1 if done, 0 if not

                // Calculate Target Q-Values using Double DQN
                // Q_target = r + gamma * Q_target_net(s', argmax(Q_online_net(s')))

                // 1. Select best action using Online Model
                const nextActionsProbs = this.model.predict(nextStateTensor) as tf.Tensor;
                const bestNextActions = nextActionsProbs.argMax(1);
                const oneHotBestNext = tf.oneHot(bestNextActions, this.outputShape);

                // 2. Get value of that action from Target Model
                const nextQValues = this.targetModel.predict(nextStateTensor) as tf.Tensor;
                const nextQValuesEx = nextQValues.mul(oneHotBestNext).sum(1);

                const targetQ = rewardTensor.add(nextQValuesEx.mul(GAMMA).mul(tf.scalar(1).sub(doneTensor)));

                // Train the model
                const loss = this.optimizer.minimize(() => {
                    const qValues = this.model.predict(stateTensor) as tf.Tensor;

                    // Get Q-value for the taken action
                    const oneHotActions = tf.oneHot(actionTensor, this.outputShape);
                    const qAction = qValues.mul(oneHotActions).sum(1);

                    // Loss: Huber Loss
                    return tf.losses.huberLoss(targetQ, qAction) as tf.Scalar;
                }, true); // returnCost = true

                if (i === TRAIN_STEPS - 1 && loss) {
                    const lossVal = loss.dataSync()[0];
                    console.log(`Episode Training Loss: ${lossVal.toFixed(5)}`);
                    loss.dispose();
                } else if (loss) {
                    loss.dispose();
                }
            });
        }

        // 3. Update Epsilon
        this.updateEpsilon();

        // 4. Update Target Network periodically
        this.trainCalls++;
        if (this.trainCalls % TARGET_UPDATE_FREQ === 0) {
            this.updateTargetModel();
            console.log(`Updated Target Network (Epsilon: ${this.epsilon.toFixed(3)})`);
        }
    }

    // --- Persistence ---

    async save(modelName: string): Promise<void> {
        await this.model.save(`indexeddb://dqn-${modelName}`);
        console.log(`DQN Model ${modelName} saved to IndexedDB`);
    }

    async load(modelName: string): Promise<void> {
        try {
            const loadedModel = await tf.loadLayersModel(`indexeddb://dqn-${modelName}`);

            this.model.dispose();
            this.targetModel.dispose();

            this.model = loadedModel;
            this.targetModel = this.createModel();
            this.updateTargetModel(); // Copy weights to target

            // Ideally we should compile the model with optimizer, but we handle optimizer separately in this custom loop
            // Just ensure optimizer handles the new weights if needed. 
            // In TF.js, optimizer.minimize() gets fresh gradients from the function provided.

            console.log(`DQN Model ${modelName} loaded from IndexedDB`);

        } catch (err) {
            console.error(`Failed to load model ${modelName}:`, err);
            throw err;
        }
    }

    static async listModels(): Promise<string[]> {
        const models = await tf.io.listModels();
        const keys = Object.keys(models).filter(k => k.startsWith('indexeddb://dqn-'));
        return keys.map(k => k.replace('indexeddb://dqn-', ''));
    }

    static async deleteModel(modelName: string): Promise<void> {
        await tf.io.removeModel(`indexeddb://dqn-${modelName}`);
    }
}
