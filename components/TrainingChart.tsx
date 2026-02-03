import React from 'react';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

interface Props {
  data: { episode: number; reward: number }[];
}

export const TrainingChart: React.FC<Props> = ({ data }) => {
  return (
    <ResponsiveContainer width="100%" height="100%">
      <AreaChart
        data={data}
        margin={{
          top: 10,
          right: 30,
          left: 0,
          bottom: 0,
        }}
      >
        <defs>
          <linearGradient id="colorReward" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor="#818cf8" stopOpacity={0.8}/>
            <stop offset="95%" stopColor="#818cf8" stopOpacity={0}/>
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
        <XAxis dataKey="episode" stroke="#94a3b8" fontSize={12} tickLine={false} />
        <YAxis stroke="#94a3b8" fontSize={12} tickLine={false} />
        <Tooltip 
          contentStyle={{ backgroundColor: '#1e293b', borderColor: '#475569', color: '#f8fafc' }}
          itemStyle={{ color: '#818cf8' }}
        />
        <Area 
          type="monotone" 
          dataKey="reward" 
          stroke="#818cf8" 
          fillOpacity={1} 
          fill="url(#colorReward)" 
          isAnimationActive={false} // Performance
        />
      </AreaChart>
    </ResponsiveContainer>
  );
};