import { GoogleGenAI } from "@google/genai";

export const analyzeTrainingPerformance = async (rewardHistory: { episode: number; reward: number }[]): Promise<string> => {
  try {
    const apiKey = process.env.API_KEY;
    if (!apiKey) {
      throw new Error("API Key not found");
    }

    const ai = new GoogleGenAI({ apiKey });

    // Sample the last 50 episodes to keep prompt size manageable
    const recentHistory = rewardHistory.slice(-50);
    const dataStr = JSON.stringify(recentHistory);

    // Max steps = 500, Reward per step = 5. Max potential reward approx 2500.
    const prompt = `
      I am training a Reinforcement Learning agent (PPO) to balance a CartPole.
      Here is the reward history for the recent episodes (Episode vs Reward):
      ${dataStr}

      The reward function is: +5 for every step balanced (angle 70-139 deg), -10 for failure.
      Max steps per episode is 500, so max score is around 2500.
      
      Please analyze this data briefly in 2-3 sentences. 
      Is the agent learning? Is it unstable? What would you recommend checking (e.g., learning rate, batch size) if it's stuck?
      Be encouraging but technical.
    `;

    const response = await ai.models.generateContent({
      model: 'gemini-3-flash-preview',
      contents: prompt,
    });

    return response.text || "No analysis available.";
  } catch (error: any) {
    console.error("Gemini analysis failed:", error);
    if (error?.status === 429 || error?.message?.includes('429') || error?.message?.includes('RESOURCE_EXHAUSTED')) {
      return "Gemini quota exceeded. Please wait a moment before trying again.";
    }
    return "Could not retrieve analysis at this time. Please check your API key.";
  }
};