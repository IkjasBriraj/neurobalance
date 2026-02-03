import React, { useRef, useEffect, useState } from 'react';
import { CartPole } from '../services/cartpole';

interface Props {
  cartPole: CartPole;
  interactionEnabled: boolean;
}

export const SimulationCanvas: React.FC<Props> = ({ cartPole, interactionEnabled }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const isDraggingRef = useRef(false);
  const lastXRef = useRef(0);
  
  // -- Interaction Handlers --

  const handleStart = (clientX: number) => {
    if (!interactionEnabled) return;
    isDraggingRef.current = true;
    lastXRef.current = clientX;
  };

  const handleMove = (clientX: number) => {
    if (!interactionEnabled || !isDraggingRef.current) return;
    
    const deltaX = clientX - lastXRef.current;
    
    // Sensitivity multiplier. 
    // Moving 10px in one event frame is fast -> 10 * 1.5 = 15 Force (1.5x Engine Force)
    const FORCE_MULTIPLIER = 1.5; 
    
    const force = deltaX * FORCE_MULTIPLIER;
    cartPole.applyForce(force);
    
    lastXRef.current = clientX;
  };

  const handleEnd = () => {
    isDraggingRef.current = false;
  };

  // Mouse
  const onMouseDown = (e: React.MouseEvent) => handleStart(e.clientX);
  const onMouseMove = (e: React.MouseEvent) => handleMove(e.clientX);
  const onMouseUp = () => handleEnd();
  const onMouseLeave = () => handleEnd();

  // Touch
  const onTouchStart = (e: React.TouchEvent) => handleStart(e.touches[0].clientX);
  const onTouchMove = (e: React.TouchEvent) => handleMove(e.touches[0].clientX);
  const onTouchEnd = () => handleEnd();


  // -- Render Loop --
  useEffect(() => {
    let animationFrameId: number;

    const render = () => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      const { width, height } = canvas;
      const scale = 100; // pixels per meter
      
      // Clear
      ctx.fillStyle = '#0f172a'; // slate-900
      ctx.fillRect(0, 0, width, height);

      // Track
      const trackY = height * 0.7;
      ctx.beginPath();
      ctx.moveTo(0, trackY);
      ctx.lineTo(width, trackY);
      ctx.strokeStyle = '#334155'; // slate-700
      ctx.lineWidth = 2;
      ctx.stroke();

      // State
      const { x, theta } = cartPole.state;
      const cartX = width / 2 + x * scale;
      const cartY = trackY;
      const cartWidth = 50;
      const cartHeight = 30;

      // Cart
      ctx.fillStyle = '#6366f1'; // indigo-500
      ctx.shadowColor = '#4f46e5';
      ctx.shadowBlur = 15;
      ctx.fillRect(cartX - cartWidth / 2, cartY - cartHeight / 2, cartWidth, cartHeight);
      ctx.shadowBlur = 0;

      // Pole
      const poleLength = 120; // visual length
      const poleEndX = cartX + Math.sin(theta) * poleLength;
      const poleEndY = cartY - Math.cos(theta) * poleLength;

      ctx.beginPath();
      ctx.moveTo(cartX, cartY);
      ctx.lineTo(poleEndX, poleEndY);
      ctx.lineWidth = 8;
      ctx.lineCap = 'round';
      ctx.strokeStyle = '#e2e8f0'; // slate-200
      ctx.stroke();

      // Pivot Point
      ctx.beginPath();
      ctx.arc(cartX, cartY, 5, 0, 2 * Math.PI);
      ctx.fillStyle = '#fbbf24'; // amber-400
      ctx.fill();
      
      // Interaction Indicator
      if (interactionEnabled) {
          ctx.font = '12px sans-serif';
          ctx.fillStyle = 'rgba(255, 255, 255, 0.5)';
          ctx.textAlign = 'center';
          ctx.fillText("DRAG TO APPLY FORCE", width / 2, height - 20);
      }

      animationFrameId = requestAnimationFrame(render);
    };

    render();

    return () => cancelAnimationFrame(animationFrameId);
  }, [cartPole, interactionEnabled]);

  return (
    <canvas 
        ref={canvasRef} 
        width={800} 
        height={450} 
        className={`w-full h-full object-cover touch-none ${interactionEnabled ? 'cursor-grab active:cursor-grabbing' : 'cursor-default'}`}
        onMouseDown={onMouseDown}
        onMouseMove={onMouseMove}
        onMouseUp={onMouseUp}
        onMouseLeave={onMouseLeave}
        onTouchStart={onTouchStart}
        onTouchMove={onTouchMove}
        onTouchEnd={onTouchEnd}
    />
  );
};