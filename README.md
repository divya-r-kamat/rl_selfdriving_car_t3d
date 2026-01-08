# Self Driving Car using TD3 (Twin Delayed DDPG).

A reinforcement learning project that trains a car to navigate autonomously using the **TD3 algorithm**.

---

## What is TD3?

**TD3 (Twin Delayed Deep Deterministic Policy Gradient)** is a state-of-the-art reinforcement learning algorithm for continuous control tasks.

### Simple Explanation

Think of TD3 as having two components:

1. **Actor** - The "driver" that decides what steering angle to use
2. **Critics** - Two "teachers" that evaluate how good the steering was

The actor learns to drive by trying actions, and the critics give feedback on whether those actions were good or bad.

### Why "Twin Delayed"?

- **Twin** = Uses TWO critic networks instead of one (more reliable feedback)
- **Delayed** = Updates the actor less frequently (more stable learning)

---

## 🔑 Key Features

### Continuous Control
Unlike methods that pick from fixed actions (turn left/straight/right), TD3 can output **any steering angle** between -30° and +30°. This creates smooth, natural driving behavior.

### How It Works

```
1. Car observes environment (sensors detect obstacles)
2. Actor network outputs steering angle
3. Car executes action and gets reward
4. Critics evaluate if the action was good
5. Actor improves based on critics' feedback
6. Repeat until car learns to drive!
```

### What the Car Learns

- ✅ Avoid obstacles (walls, barriers)
- ✅ Navigate toward targets
- ✅ Drive smoothly without jerky movements
- ✅ Find efficient paths

---

## 🏗️ Architecture Overview

### State (What the car sees)
- 7 distance sensors (detecting obstacles)
- Angle to target
- Distance to target

### Action (What the car does)
- Single continuous value: steering angle from -30° to +30°

### Reward (Feedback)
- **+100** for reaching target
- **-100** for crashing
- **+20** bonus for staying on road
- **-0.1** time penalty (encourages efficiency)

---

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install torch PyQt6 numpy

# Run the application
python td3_car_navigation.py
```

### Usage

1. **Click on map** to place the car
2. **Click again** to place target(s)
3. **Right-click** when done
4. **Press SPACE** to start training
5. Watch the car learn!

---

## 📊 Training Process

### Phase 1: Random Exploration (0-10k steps)
- Car takes random actions
- Learns what obstacles look like
- Fills memory with experiences

### Phase 2: Learning (10k-50k steps)  
- Actor starts making decisions
- Learns to avoid crashes
- Begins reaching targets occasionally

### Phase 3: Mastery (50k+ steps)
- Consistent target reaching
- Smooth, efficient paths
- Few crashes

**Expected training time:** 30-60 minutes on a modern CPU

---

## 🎮 Controls

- **SPACE** - Start/Pause training
- **↺ RESET** - Clear and restart
- **📂 LOAD MAP** - Use custom map
- **💾 SAVE MODEL** - Save trained model
- **📥 LOAD MODEL** - Load pre-trained model

---

## 📈 Results

After training, you should see:
- Success rate: 80-90%
- Smooth steering (no jerky movements)
- Efficient paths to targets
- Consistent behavior

---

## ⚙️ Key Hyperparameters

```python
BATCH_SIZE = 512        # Training batch size
DISCOUNT = 0.98         # Future reward importance
POLICY_FREQ = 2         # Update actor every 2 steps
START_TIMESTEPS = 10000 # Random exploration period
EXPL_NOISE = 0.1        # Exploration noise
```

---

## 🎓 How TD3 Improves Upon DDPG

TD3 fixes three main problems in DDPG:

1. **Twin Critics** → Prevents overestimating action values
2. **Delayed Updates** → Makes training more stable
3. **Target Smoothing** → Reduces variance in learning

These improvements make TD3 one of the best algorithms for continuous control!

---




Made with ❤️ using PyTorch and PyQt6

</div>
