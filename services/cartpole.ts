// Physics constants mirroring standard CartPole-v1
const GRAVITY = 9.8;
const MASSCART = 1.0;
const MASSPOLE = 0.1;
const TOTAL_MASS = MASSCART + MASSPOLE;
const LENGTH = 0.5; // actually half the pole's length
const POLEMASS_LENGTH = MASSPOLE * LENGTH;
const FORCE_MAG = 10.0;
const TAU = 0.02; // seconds between state updates

const X_THRESHOLD = 2.4;

export interface State {
  x: number;
  xDot: number;
  theta: number;
  thetaDot: number;
}

export class CartPole {
  state: State;
  private externalForce: number = 0;

  constructor() {
    this.state = { x: 0, xDot: 0, theta: 0, thetaDot: 0 };
    this.reset();
  }

  reset() {
    // Start with small random perturbation
    this.state = {
      x: Math.random() * 0.1 - 0.05,
      xDot: Math.random() * 0.1 - 0.05,
      theta: Math.random() * 0.1 - 0.05,
      thetaDot: Math.random() * 0.1 - 0.05
    };
    this.externalForce = 0;
    return this.state;
  }

  // Allow external interaction (e.g. mouse drag) to push the cart
  applyForce(force: number) {
    this.externalForce += force;
  }

  getStateArray(): number[] {
    return [this.state.x, this.state.xDot, this.state.theta, this.state.thetaDot];
  }

  step(action: 'left' | 'right'): { state: State; reward: number; done: boolean } {
    // Combine engine force with any external user-applied force
    const engineForce = action === 'right' ? FORCE_MAG : -FORCE_MAG;
    const force = engineForce + this.externalForce;

    // Reset external force after applying it (impulse behavior)
    this.externalForce = 0;

    const { x, xDot, theta, thetaDot } = this.state;

    const costheta = Math.cos(theta);
    const sintheta = Math.sin(theta);

    const temp = (force + POLEMASS_LENGTH * thetaDot * thetaDot * sintheta) / TOTAL_MASS;
    const thetaAcc = (GRAVITY * sintheta - costheta * temp) / (LENGTH * (4.0 / 3.0 - MASSPOLE * costheta * costheta / TOTAL_MASS));
    const xAcc = temp - POLEMASS_LENGTH * thetaAcc * costheta / TOTAL_MASS;

    // Euler integration
    const nextx = x + TAU * xDot;
    const nextxDot = xDot + TAU * xAcc;
    const nextTheta = theta + TAU * thetaDot;
    const nextThetaDot = thetaDot + TAU * thetaAcc;

    this.state = {
      x: nextx,
      xDot: nextxDot,
      theta: nextTheta,
      thetaDot: nextThetaDot
    };

    // Convert radians to degrees, assuming 0 rad is 90 degrees (upright)
    // Positive theta tilts Right (> 90), Negative theta tilts Left (< 90)
    const degrees = 90 + (nextTheta * 180 / Math.PI);

    // Custom Reward Logic per user request:
    // Safe Range: 70 to 139 degrees -> +5 reward
    // Failure/Penalty: Outside this range (e.g. 60, 67, 140) -> -10 reward

    const isAngleSafe = degrees >= 70 && degrees <= 139;
    const isPositionSafe = nextx > -X_THRESHOLD && nextx < X_THRESHOLD;

    // Episode ends if angle is unsafe (fell over) or cart is out of bounds
    const done = !isAngleSafe || !isPositionSafe;

    let reward = 0;
    if (!isAngleSafe || !isPositionSafe) {
      reward = -10;
    } else {
      // Shaped Reward Logic "Better":
      // Encourages the agent to be perfectly upright (90 deg) and in the center (x=0).
      // Provides a gradient so the agent knows if it's improving even within the safe zone.

      const angleLimit = 139 - 90; // approx 49 degrees deviation allowed
      const angleError = Math.abs(degrees - 90) / angleLimit; // 0.0 (perfect) to 1.0 (about to fail)

      const positionError = Math.abs(nextx) / X_THRESHOLD; // 0.0 (center) to 1.0 (edge)

      // Base reward for survival: 1.0
      // Bonus for Uprightness: up to 2.0
      // Bonus for Centering: up to 2.0
      // Total Max Reward: ~5.0 (Matches previous max)

      reward = 1.0 + (2.0 * (1.0 - angleError)) + (2.0 * (1.0 - positionError));

      // Explicit Penalty for being too far from center (> 1.5 units) to discourage edge-riding
      if (Math.abs(nextx) > 1.5) {
        reward -= 2.0;
      }

      // Small penalty for high angular velocity to reduce shaking
      // reward -= Math.abs(nextThetaDot) * 0.1;
    }

    return { state: this.state, reward, done };
  }
}