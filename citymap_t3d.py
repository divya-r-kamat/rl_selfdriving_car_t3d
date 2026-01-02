"""
===============================================================================
TD3 Self-Driving Car 
===============================================================================
"""

import os
import sys
import math
import numpy as np
import random
import time
from collections import deque

# --- PYTORCH ---
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# --- PYQT ---
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QGraphicsScene, 
                             QGraphicsView, QGraphicsItem, QFrame, QFileDialog,
                             QTextEdit, QGridLayout)
from PyQt6.QtGui import (QImage, QPixmap, QColor, QPen, QBrush, QPainter, 
                         QPolygonF, QFont, QPainterPath)
from PyQt6.QtCore import Qt, QTimer, QPointF, QRectF

# ==========================================
# CONFIGURATION & THEME
# ==========================================
C_BG_DARK   = QColor("#2E3440") 
C_PANEL     = QColor("#3B4252")
C_INFO_BG   = QColor("#4C566A") 
C_ACCENT    = QColor("#88C0D0") 
C_TEXT      = QColor("#ECEFF4") 
C_SUCCESS   = QColor("#A3BE8C") 
C_FAILURE   = QColor("#BF616A") 
C_SENSOR_ON = QColor("#A3BE8C")
C_SENSOR_OFF= QColor("#BF616A")

# ==========================================
# PHYSICS PARAMETERS
# ==========================================
CAR_WIDTH = 14 
CAR_HEIGHT = 8   
SENSOR_DIST = 16
SPEED = 5
MAX_TURN_ANGLE = 30

# ==========================================
# TD3 HYPERPARAMETERS (from provided code)
# ==========================================
START_TIMESTEPS = 10000  # Random actions before using policy
EVAL_FREQ = 5000         # Evaluation frequency
EXPL_NOISE = 0.1         # Exploration noise
BATCH_SIZE = 512         # Training batch size
DISCOUNT = 0.98          # Gamma
TAU = 0.005              # Target network update rate
POLICY_NOISE = 0.2       # Target policy smoothing
NOISE_CLIP = 0.5         # Noise clipping
POLICY_FREQ = 2          # Delayed policy updates

# Target Colors
TARGET_COLORS = [
    QColor(0, 255, 255), QColor(255, 100, 255), QColor(0, 255, 100),
    QColor(255, 150, 0), QColor(100, 150, 255), QColor(255, 50, 150),
    QColor(150, 255, 50), QColor(255, 255, 0),
]

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# REPLAY BUFFER (Exact from TD3 code)
# ==========================================
class ReplayBuffer(object):
    def __init__(self, max_size=1e6):
        self.storage = []
        self.max_size = int(max_size)
        self.ptr = 0

    def add(self, transition):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = transition
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(transition)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = [], [], [], [], []
        for i in ind:
            state, next_state, action, reward, done = self.storage[i]
            batch_states.append(np.array(state))
            batch_next_states.append(np.array(next_state))
            batch_actions.append(np.array(action))
            batch_rewards.append(np.array(reward))
            batch_dones.append(np.array(done))
        return (np.array(batch_states), np.array(batch_next_states), 
                np.array(batch_actions), np.array(batch_rewards).reshape(-1, 1), 
                np.array(batch_dones).reshape(-1, 1))

# ==========================================
# ACTOR NETWORK (Exact from TD3 code)
# ==========================================
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.layer_1 = nn.Linear(state_dim, 400)
        self.layer_2 = nn.Linear(400, 300)
        self.layer_3 = nn.Linear(300, action_dim)
        self.max_action = max_action

    def forward(self, x):
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = self.max_action * torch.tanh(self.layer_3(x))
        return x

# ==========================================
# CRITIC NETWORK (Exact from TD3 code)
# ==========================================
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # First Critic
        self.layer_1 = nn.Linear(state_dim + action_dim, 400)
        self.layer_2 = nn.Linear(400, 300)
        self.layer_3 = nn.Linear(300, 1)
        # Second Critic
        self.layer_4 = nn.Linear(state_dim + action_dim, 400)
        self.layer_5 = nn.Linear(400, 300)
        self.layer_6 = nn.Linear(300, 1)

    def forward(self, x, u):
        xu = torch.cat([x, u], 1)
        # First Critic
        x1 = F.relu(self.layer_1(xu))
        x1 = F.relu(self.layer_2(x1))
        x1 = self.layer_3(x1)
        # Second Critic
        x2 = F.relu(self.layer_4(xu))
        x2 = F.relu(self.layer_5(x2))
        x2 = self.layer_6(x2)
        return x1, x2

    def Q1(self, x, u):
        xu = torch.cat([x, u], 1)
        x1 = F.relu(self.layer_1(xu))
        x1 = F.relu(self.layer_2(x1))
        x1 = self.layer_3(x1)
        return x1

# ==========================================
# TD3 ALGORITHM (Exact from TD3 code)
# ==========================================
class TD3(object):
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())
        self.max_action = max_action

    def select_action(self, state):
        state = torch.Tensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, iterations, batch_size=100, discount=0.99, 
              tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):
        
        for it in range(iterations):
            # Sample batch
            batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = replay_buffer.sample(batch_size)
            state = torch.Tensor(batch_states).to(device)
            next_state = torch.Tensor(batch_next_states).to(device)
            action = torch.Tensor(batch_actions).to(device)
            reward = torch.Tensor(batch_rewards).to(device)
            done = torch.Tensor(batch_dones).to(device)

            # Target policy smoothing
            next_action = self.actor_target(next_state)
            noise = torch.Tensor(batch_actions).data.normal_(0, policy_noise).to(device)
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action = (next_action + noise).clamp(-self.max_action, self.max_action)

            # Compute target Q-value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + ((1 - done) * discount * target_Q).detach()

            # Get current Q estimates
            current_Q1, current_Q2 = self.critic(state, action)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            # Optimize critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Delayed policy updates
            if it % policy_freq == 0:
                actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
                
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Update target networks
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
        torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))

    def load(self, filename, directory):
        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
        self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

# ==========================================
# CAR BRAIN
# ==========================================
class CarBrain:
    def __init__(self, map_image: QImage):
        self.map = map_image
        self.w, self.h = map_image.width(), map_image.height()
        
        # State and action dimensions
        self.state_dim = 9
        self.action_dim = 1
        self.max_action = 1.0
        
        # TD3
        self.policy = TD3(self.state_dim, self.action_dim, self.max_action)
        self.replay_buffer = ReplayBuffer()
        
        # Training variables
        self.total_timesteps = 0
        self.episode_num = 0
        self.episode_timesteps = 0
        self.episode_reward = 0
        self.last_reward = 0
        self.last_action = [0.0]
        self.explore_noise = EXPL_NOISE
        self.consecutive_crashes = 0
        
        # Positions
        self.start_pos = QPointF(100, 100)
        self.car_pos = QPointF(100, 100)
        self.car_angle = 0
        self.targets = []
        self.current_target_idx = 0
        self.targets_reached = 0
        
        self.alive = True
        self.score = 0
        self.sensor_coords = []
        self.prev_dist = None

    def set_start_pos(self, point):
        self.start_pos = point
        self.car_pos = point

    def add_target(self, point):
        self.targets.append(QPointF(point.x(), point.y()))

    def reset(self):
        self.alive = True
        self.score = 0
        self.car_pos = QPointF(self.start_pos.x(), self.start_pos.y())
        self.car_angle = random.randint(0, 360)
        self.current_target_idx = 0
        self.targets_reached = 0
        
        if len(self.targets) > 0:
            self.target_pos = self.targets[0]
        
        state, dist = self.get_state()
        self.prev_dist = dist
        self.episode_timesteps = 0
        self.episode_reward = 0
        return state

    def switch_to_next_target(self):
        if self.current_target_idx < len(self.targets) - 1:
            self.current_target_idx += 1
            self.target_pos = self.targets[self.current_target_idx]
            self.targets_reached += 1
            return True
        return False

    def get_state(self):
        sensor_vals = []
        self.sensor_coords = []
        angles = [-45, -30, -15, 0, 15, 30, 45]
        
        for a in angles:
            rad = math.radians(self.car_angle + a)
            sx = self.car_pos.x() + math.cos(rad) * SENSOR_DIST
            sy = self.car_pos.y() + math.sin(rad) * SENSOR_DIST
            self.sensor_coords.append(QPointF(sx, sy))
            
            val = 0.0
            if 0 <= sx < self.w and 0 <= sy < self.h:
                c = QColor(self.map.pixel(int(sx), int(sy)))
                brightness = (c.red() + c.green() + c.blue()) / 3.0
                val = brightness / 255.0
            sensor_vals.append(val)
        
        dx = self.target_pos.x() - self.car_pos.x()
        dy = self.target_pos.y() - self.car_pos.y()
        dist = math.sqrt(dx*dx + dy*dy)
        
        rad_to_target = math.atan2(dy, dx)
        angle_to_target = math.degrees(rad_to_target)
        
        angle_diff = (angle_to_target - self.car_angle) % 360
        if angle_diff > 180:
            angle_diff -= 360
        
        norm_dist = min(dist / 800.0, 1.0)
        norm_angle = angle_diff / 180.0
        
        state = sensor_vals + [norm_angle, norm_dist]
        return np.array(state, dtype=np.float32), dist

    def step(self, action):
        turn = action * MAX_TURN_ANGLE
        self.car_angle += turn
        
        rad = math.radians(self.car_angle)
        new_x = self.car_pos.x() + math.cos(rad) * SPEED
        new_y = self.car_pos.y() + math.sin(rad) * SPEED
        self.car_pos = QPointF(new_x, new_y)
        
        next_state, dist = self.get_state()
        reward = -0.1
        done = False
        
        car_center_val = self.check_pixel(self.car_pos.x(), self.car_pos.y())
        
        if car_center_val < 0.4:
            reward = -100
            done = True
            self.alive = False
        elif dist < 20:
            reward = 100
            has_next = self.switch_to_next_target()
            if has_next:
                done = False
                _, new_dist = self.get_state()
                self.prev_dist = new_dist
            else:
                done = True
        else:
            reward += next_state[3] * 20
            if self.prev_dist is not None:
                if dist < self.prev_dist:
                    reward += 10
                else:
                    reward -= 5
        
        self.prev_dist = dist
        self.score += reward
        self.episode_reward += reward
        self.episode_timesteps += 1
        self.total_timesteps += 1
        
        return next_state, reward, done

    def check_pixel(self, x, y):
        if 0 <= x < self.w and 0 <= y < self.h:
            c = QColor(self.map.pixel(int(x), int(y)))
            return ((c.red() + c.green() + c.blue()) / 3.0) / 255.0
        return 0.0

# ==========================================
# REWARD CHART
# ==========================================
class RewardChart(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumHeight(150)
        self.setStyleSheet(f"background-color: {C_PANEL.name()}; border-radius: 5px;")
        self.scores = []
        self.max_points = 50

    def update_chart(self, new_score):
        self.scores.append(new_score)
        if len(self.scores) > self.max_points:
            self.scores.pop(0)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        w = self.width()
        h = self.height()
        painter.fillRect(0, 0, w, h, C_PANEL)
        
        if len(self.scores) < 2:
            return

        min_val = min(self.scores)
        max_val = max(self.scores)
        if max_val == min_val: max_val += 1
        
        points = []
        step_x = w / (self.max_points - 1)
        
        # Draw raw line (cyan/blue)
        for i, score in enumerate(self.scores):
            x = i * step_x
            ratio = (score - min_val) / (max_val - min_val)
            y = h - (ratio * (h * 0.8) + (h * 0.1))
            points.append(QPointF(x, y))

        path = QPainterPath()
        path.moveTo(points[0])
        for p in points[1:]:
            path.lineTo(p)
            
        pen = QPen(QColor(136, 192, 208), 2)  # Cyan/blue for raw
        painter.setPen(pen)
        painter.drawPath(path)
        
        # Draw average line (yellow/gold)
        if len(self.scores) >= 2:
            avg_points = []
            window_size = 10
            
            for i in range(len(self.scores)):
                start_idx = max(0, i - window_size + 1)
                avg_score = sum(self.scores[start_idx:i+1]) / (i - start_idx + 1)
                
                x = i * step_x
                ratio = (avg_score - min_val) / (max_val - min_val)
                y = h - (ratio * (h * 0.8) + (h * 0.1))
                avg_points.append(QPointF(x, y))
            
            if len(avg_points) > 1:
                avg_path = QPainterPath()
                avg_path.moveTo(avg_points[0])
                for p in avg_points[1:]:
                    avg_path.lineTo(p)
                
                avg_pen = QPen(QColor(255, 215, 0), 3)  # Gold for average
                painter.setPen(avg_pen)
                painter.drawPath(avg_path)
        
        # Draw legend
        legend_x = 10
        legend_y = 15
        
        # Raw line legend
        painter.setPen(QPen(QColor(136, 192, 208), 2))
        painter.drawLine(legend_x, legend_y, legend_x + 20, legend_y)
        painter.setPen(QPen(QColor(200, 200, 200)))
        painter.setFont(QFont("Segoe UI", 9))
        painter.drawText(legend_x + 25, legend_y + 4, "Raw")
        
        # Avg line legend
        painter.setPen(QPen(QColor(255, 215, 0), 3))
        painter.drawLine(legend_x + 60, legend_y, legend_x + 80, legend_y)
        painter.setPen(QPen(QColor(200, 200, 200)))
        painter.drawText(legend_x + 85, legend_y + 4, "Avg (10)")

# ==========================================
# VISUAL ITEMS
# ==========================================
class SensorItem(QGraphicsItem):
    def __init__(self):
        super().__init__()
        self.setZValue(90)
        self.is_detecting = True
        
    def set_detecting(self, detecting):
        self.is_detecting = detecting
        self.update()
    
    def boundingRect(self):
        return QRectF(-3, -3, 6, 6)
    
    def paint(self, painter, option, widget):
        color = C_SENSOR_ON if self.is_detecting else C_SENSOR_OFF
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(color))
        painter.drawEllipse(QPointF(0, 0), 2, 2)

class CarItem(QGraphicsItem):
    def __init__(self):
        super().__init__()
        self.setZValue(100)

    def boundingRect(self):
        return QRectF(-CAR_WIDTH/2, -CAR_HEIGHT/2, CAR_WIDTH, CAR_HEIGHT)

    def paint(self, painter, option, widget):
        painter.setBrush(QBrush(C_ACCENT))
        painter.setPen(QPen(Qt.GlobalColor.white, 1))
        painter.drawRoundedRect(self.boundingRect(), 2, 2)
        painter.setBrush(Qt.GlobalColor.white)
        painter.drawRect(int(CAR_WIDTH/2)-2, -3, 2, 6)

class TargetItem(QGraphicsItem):
    def __init__(self, color=None, is_active=True, number=1):
        super().__init__()
        self.setZValue(50)
        self.pulse = 0
        self.growing = True
        self.color = color if color else QColor(0, 255, 255)
        self.is_active = is_active
        self.number = number

    def set_active(self, active):
        self.is_active = active
        self.update()

    def boundingRect(self):
        return QRectF(-20, -20, 40, 40)

    def paint(self, painter, option, widget):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        if self.is_active:
            if self.growing:
                self.pulse += 0.5
                if self.pulse > 10: self.growing = False
            else:
                self.pulse -= 0.5
                if self.pulse < 0: self.growing = True
            
            r = 10 + self.pulse
            outer_color = QColor(self.color)
            outer_color.setAlpha(100)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(outer_color))
            painter.drawEllipse(QPointF(0, 0), r, r)
            
            painter.setBrush(QBrush(self.color))
            painter.setPen(QPen(Qt.GlobalColor.white, 2))
            painter.drawEllipse(QPointF(0, 0), 8, 8)
        else:
            dimmed_color = QColor(self.color)
            dimmed_color.setAlpha(120)
            painter.setPen(QPen(Qt.GlobalColor.white, 1))
            painter.setBrush(QBrush(dimmed_color))
            painter.drawEllipse(QPointF(0, 0), 6, 6)
        
        painter.setPen(QPen(Qt.GlobalColor.white))
        painter.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        painter.drawText(QRectF(-10, -10, 20, 20), Qt.AlignmentFlag.AlignCenter, str(self.number))

# ==========================================
# MAIN APPLICATION
# ==========================================
class TD3NavApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TD3 Self-Driving Car - Training & Inference")
        self.resize(1300, 850)
        self.setStyleSheet(f"""
            QMainWindow {{ background-color: {C_BG_DARK.name()}; }}
            QLabel {{ color: {C_TEXT.name()}; font-family: Segoe UI; font-size: 13px; }}
            QPushButton {{ background-color: {C_PANEL.name()}; color: white; border: 1px solid {C_INFO_BG.name()}; padding: 8px; border-radius: 4px; }}
            QPushButton:hover {{ background-color: {C_INFO_BG.name()}; }}
            QPushButton:checked {{ background-color: {C_ACCENT.name()}; color: black; }}
            QTextEdit {{ background-color: {C_PANEL.name()}; color: #D8DEE9; border: none; font-family: Consolas; font-size: 11px; }}
        """)

        main_layout = QHBoxLayout()

        # LEFT PANEL
        panel = QFrame()
        panel.setFixedWidth(280)
        panel.setStyleSheet(f"background-color: {C_BG_DARK.name()};")
        vbox = QVBoxLayout(panel)
        vbox.setSpacing(10)
        
        lbl_title = QLabel("TD3 CONTROLS")
        lbl_title.setStyleSheet("font-weight: bold; font-size: 14px;")
        vbox.addWidget(lbl_title)
        
        self.lbl_status = QLabel("1. Click Map → CAR\n2. Click Map → TARGET(S)\n   Right-click when done")
        self.lbl_status.setStyleSheet(f"background-color: {C_INFO_BG.name()}; padding: 10px; border-radius: 5px;")
        vbox.addWidget(self.lbl_status)

        self.btn_run = QPushButton("▶ START (Space)")
        self.btn_run.setCheckable(True)
        self.btn_run.setEnabled(False)
        self.btn_run.clicked.connect(self.toggle_training)
        vbox.addWidget(self.btn_run)
        
        self.btn_reset = QPushButton("↺ RESET")
        self.btn_reset.clicked.connect(self.hard_reset)
        vbox.addWidget(self.btn_reset)
        
        self.btn_load = QPushButton("📂 LOAD MAP")
        self.btn_load.clicked.connect(self.load_map_dialog)
        vbox.addWidget(self.btn_load)
        
        self.btn_save = QPushButton("💾 SAVE MODEL")
        self.btn_save.clicked.connect(self.save_model)
        vbox.addWidget(self.btn_save)
        
        self.btn_load_model = QPushButton("📥 LOAD MODEL")
        self.btn_load_model.clicked.connect(self.load_model)
        vbox.addWidget(self.btn_load_model)

        vbox.addSpacing(10)
        vbox.addWidget(QLabel("REWARD HISTORY"))
        self.chart = RewardChart()
        vbox.addWidget(self.chart)

        stats_frame = QFrame()
        stats_frame.setStyleSheet(f"background-color: {C_PANEL.name()}; border-radius: 5px;")
        sf_layout = QGridLayout(stats_frame)
        sf_layout.setContentsMargins(10, 10, 10, 10)
        
        sf_layout.addWidget(QLabel("Explore Noise:"), 0, 0)
        self.val_noise = QLabel("0.100")
        self.val_noise.setStyleSheet(f"color: {C_ACCENT.name()}; font-weight: bold;")
        sf_layout.addWidget(self.val_noise, 0, 1)
        
        sf_layout.addWidget(QLabel("Last Reward:"), 1, 0)
        self.val_last_rew = QLabel("0")
        self.val_last_rew.setStyleSheet(f"color: {C_ACCENT.name()}; font-weight: bold;")
        sf_layout.addWidget(self.val_last_rew, 1, 1)
        
        sf_layout.addWidget(QLabel("Action [S,T]:"), 2, 0)
        self.val_action = QLabel("0.00, 0.00")
        self.val_action.setStyleSheet(f"color: {C_ACCENT.name()}; font-weight: bold;")
        sf_layout.addWidget(self.val_action, 2, 1)
        
        vbox.addWidget(stats_frame)

        vbox.addWidget(QLabel("LOGS"))
        self.log_console = QTextEdit()
        self.log_console.setReadOnly(True)
        vbox.addWidget(self.log_console)

        main_layout.addWidget(panel)

        # RIGHT PANEL
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        self.view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.view.setStyleSheet(f"border: 2px solid {C_PANEL.name()}; background-color: {C_BG_DARK.name()}")
        self.view.mousePressEvent = self.on_scene_click
        main_layout.addWidget(self.view)

        # BOTTOM STATUS BAR
        bottom_bar = QFrame()
        bottom_bar.setStyleSheet(f"background-color: {C_PANEL.name()}; border-top: 1px solid {C_INFO_BG.name()};")
        bottom_layout = QHBoxLayout(bottom_bar)
        bottom_layout.setContentsMargins(15, 5, 15, 5)
        
        self.lbl_timesteps = QLabel("Timesteps: 0")
        self.lbl_timesteps.setStyleSheet(f"color: {C_ACCENT.name()}; font-weight: bold;")
        bottom_layout.addWidget(self.lbl_timesteps)
        
        bottom_layout.addStretch()
        
        self.lbl_episode = QLabel("Episode: 0")
        self.lbl_episode.setStyleSheet(f"color: {C_ACCENT.name()}; font-weight: bold;")
        bottom_layout.addWidget(self.lbl_episode)
        
        bottom_layout.addStretch()
        
        self.lbl_reward = QLabel("Reward: 0")
        self.lbl_reward.setStyleSheet(f"color: {C_ACCENT.name()}; font-weight: bold;")
        bottom_layout.addWidget(self.lbl_reward)
        
        # Add bottom bar to main window
        central_with_bottom = QVBoxLayout()
        central_with_bottom.setContentsMargins(0, 0, 0, 0)
        central_with_bottom.setSpacing(0)
        central_with_bottom.addLayout(main_layout)
        central_with_bottom.addWidget(bottom_bar)
        
        container = QWidget()
        container.setLayout(central_with_bottom)
        self.setCentralWidget(container)

        # Setup
        self.setup_map("mumbai_city_map.png")
        self.setup_state = 0
        self.sim_timer = QTimer()
        self.sim_timer.timeout.connect(self.training_loop)
        
        self.car_item = CarItem()
        self.target_items = []
        self.sensor_items = []
        for _ in range(7):
            si = SensorItem()
            self.scene.addItem(si)
            self.sensor_items.append(si)
        
        self.log("TD3 Algorithm Initialized")
        self.log(f"Device: {device}")

    def log(self, msg):
        self.log_console.append(msg)
        sb = self.log_console.verticalScrollBar()
        sb.setValue(sb.maximum())

    def setup_map(self, path):
        if not os.path.exists(path):
            self.create_dummy_map(path)
        self.map_img = QImage(path).convertToFormat(QImage.Format.Format_RGB32)
        self.scene.clear()
        self.scene.addPixmap(QPixmap.fromImage(self.map_img))
        self.brain = CarBrain(self.map_img)
        self.log(f"Map Loaded: {path}")

    def create_dummy_map(self, path):
        img = QImage(1000, 800, QImage.Format.Format_RGB32)
        img.fill(C_BG_DARK)
        p = QPainter(img)
        p.setBrush(Qt.GlobalColor.white)
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(100, 100, 800, 600)
        p.setBrush(C_BG_DARK)
        p.drawEllipse(250, 250, 500, 300)
        p.end()
        img.save(path)

    def load_map_dialog(self):
        f, _ = QFileDialog.getOpenFileName(self, "Load Map", "", "Images (*.png *.jpg)")
        if f:
            self.hard_reset()
            self.setup_map(f)

    def save_model(self):
        if not os.path.exists("./pytorch_models"):
            os.makedirs("./pytorch_models")
        filename = f"TD3_CarNav_ep{self.brain.episode_num}"
        self.brain.policy.save(filename, "./pytorch_models")
        self.log(f"<font color='#A3BE8C'>Model saved: {filename}</font>")

    def load_model(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Load Model", "./pytorch_models", 
                                                  "Model Files (*_actor.pth)")
        if filename:
            try:
                base_name = filename.replace("_actor.pth", "")
                directory = os.path.dirname(filename)
                file_only = os.path.basename(base_name)
                self.brain.policy.load(file_only, directory)
                self.log(f"<font color='#A3BE8C'>Model loaded: {file_only}</font>")
            except Exception as e:
                self.log(f"<font color='#BF616A'>Error loading: {e}</font>")

    def on_scene_click(self, event):
        pt = self.view.mapToScene(event.pos())
        
        if self.setup_state == 0:
            self.brain.set_start_pos(pt)
            self.scene.addItem(self.car_item)
            self.car_item.setPos(pt)
            self.setup_state = 1
            self.lbl_status.setText("Click Map → TARGET(S)\nRight-click when done")
            
        elif self.setup_state == 1:
            if event.button() == Qt.MouseButton.LeftButton:
                self.brain.add_target(pt)
                target_idx = len(self.brain.targets) - 1
                color = TARGET_COLORS[target_idx % len(TARGET_COLORS)]
                is_active = (target_idx == 0)
                
                target_item = TargetItem(color, is_active, target_idx + 1)
                target_item.setPos(pt)
                self.scene.addItem(target_item)
                self.target_items.append(target_item)
                
                num_targets = len(self.brain.targets)
                self.lbl_status.setText(f"Targets: {num_targets}\nRight-click to finish")
                self.log(f"Target #{num_targets} added")
            
            elif event.button() == Qt.MouseButton.RightButton:
                if len(self.brain.targets) > 0:
                    self.setup_state = 2
                    self.lbl_status.setText(f"READY. {len(self.brain.targets)} target(s). Press SPACE.")
                    self.lbl_status.setStyleSheet(f"background-color: {C_SUCCESS.name()}; color: #2E3440; font-weight: bold; padding: 10px; border-radius: 5px;")
                    self.btn_run.setEnabled(True)
                    self.update_visuals()

    def hard_reset(self):
        self.sim_timer.stop()
        self.btn_run.setChecked(False)
        self.btn_run.setEnabled(False)
        self.setup_state = 0
        
        if self.car_item.scene() == self.scene:
            self.scene.removeItem(self.car_item)
        
        for target_item in self.target_items:
            if target_item.scene() == self.scene:
                self.scene.removeItem(target_item)
        self.target_items = []
        
        for s in self.sensor_items:
            if s.scene() == self.scene:
                self.scene.removeItem(s)
        
        self.brain = CarBrain(self.map_img)
        
        self.lbl_status.setText("1. Click Map → CAR\n2. Click Map → TARGET(S)")
        self.lbl_status.setStyleSheet(f"background-color: {C_INFO_BG.name()}; color: white; padding: 10px; border-radius: 5px;")
        self.log("═══ RESET ═══")
        self.chart.scores = []
        self.chart.update()

    def toggle_training(self):
        if self.btn_run.isChecked():
            obs = self.brain.reset()
            self.sim_timer.start(16)
            self.btn_run.setText("⏸ PAUSE")
            self.log("Training STARTED")
        else:
            self.sim_timer.stop()
            self.btn_run.setText("▶ RESUME")
            self.log("Training PAUSED")

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Space and self.setup_state == 2:
            self.btn_run.click()

    def training_loop(self):
        if self.setup_state != 2:
            return

        obs, _ = self.brain.get_state()
        
        # Select action (with exploration during early training)
        if self.brain.total_timesteps < START_TIMESTEPS:
            action = np.array([random.uniform(-self.brain.max_action, self.brain.max_action)])
        else:
            action = self.brain.policy.select_action(obs)
            if self.brain.explore_noise != 0:
                action = (action + np.random.normal(0, self.brain.explore_noise, size=self.brain.action_dim)).clip(
                    -self.brain.max_action, self.brain.max_action)
        
        # Store action for display
        self.brain.last_action = action
        
        # Step environment
        prev_target_idx = self.brain.current_target_idx
        new_obs, reward, done = self.brain.step(action[0])
        
        # Store reward for display
        self.brain.last_reward = reward
        
        # Check if episode is actually done (not max steps)
        done_bool = float(done)
        
        # Store transition
        self.brain.replay_buffer.add((obs, new_obs, action, reward, done_bool))
        
        # Train continuously (not just at episode end) - KEY CHANGE for speed!
        # Only train if we have enough data and not in initial random phase
        if self.brain.total_timesteps > START_TIMESTEPS and len(self.brain.replay_buffer.storage) >= BATCH_SIZE:
            # Train with just 1 iteration per step (much faster than waiting until episode end)
            self.brain.policy.train(self.brain.replay_buffer, iterations=1,
                                   batch_size=BATCH_SIZE, discount=DISCOUNT, 
                                   tau=TAU, policy_noise=POLICY_NOISE, 
                                   noise_clip=NOISE_CLIP, policy_freq=POLICY_FREQ)
        
        # Update target visual
        if self.brain.current_target_idx != prev_target_idx:
            for i, target_item in enumerate(self.target_items):
                target_item.set_active(i == self.brain.current_target_idx)
            self.log(f"🎯 Target {prev_target_idx + 1} reached!")
        
        # Episode done
        if done:
            self.brain.episode_num += 1
            
            # Track consecutive crashes
            if not self.brain.alive:
                self.brain.consecutive_crashes += 1
            else:
                self.brain.consecutive_crashes = 0
            
            # Log episode result
            if self.brain.alive:
                if self.brain.targets_reached == len(self.brain.targets) - 1:
                    self.log(f"Episode {self.brain.episode_num}: ALL TARGETS | Reward: {int(self.brain.episode_reward)} | Steps: {self.brain.episode_timesteps}")
                    for i, target_item in enumerate(self.target_items):
                        self.log(f"🎯 Target {i+1} reached!")
                else:
                    self.log(f"Episode {self.brain.episode_num}: GOAL | Reward: {int(self.brain.episode_reward)} | Steps: {self.brain.episode_timesteps}")
            else:
                self.log(f"Episode {self.brain.episode_num}: CRASH | Reward: {int(self.brain.episode_reward)} | Steps: {self.brain.episode_timesteps}")
                
                # Warning for consecutive crashes
                if self.brain.consecutive_crashes >= 2:
                    self.log(f"⚠️ {self.brain.consecutive_crashes} consecutive crashes! Resetting to origin...")
                    if self.brain.consecutive_crashes >= 3:
                        self.log(f"💡 Tip: Adjust hyperparameters or simplify map")
            
            self.chart.update_chart(self.brain.episode_reward)
            obs = self.brain.reset()
        
        # Update UI (every 5 steps to reduce overhead)
        if self.brain.total_timesteps % 5 == 0:
            self.update_visuals()
            
            # Update statistics display
            self.val_noise.setText(f"{self.brain.explore_noise:.3f}")
            self.val_last_rew.setText(f"{int(self.brain.last_reward)}")
            # Show steering and turn angle
            steering = self.brain.last_action[0]
            turn = steering * MAX_TURN_ANGLE
            self.val_action.setText(f"{steering:.2f}, {turn:.0f}")
            
            # Update bottom status bar
            self.lbl_timesteps.setText(f"Timesteps: {self.brain.total_timesteps}")
            self.lbl_episode.setText(f"Episode: {self.brain.episode_num}")
            self.lbl_reward.setText(f"Reward: {int(self.brain.score)}")

    def update_visuals(self):
        self.car_item.setPos(self.brain.car_pos)
        self.car_item.setRotation(self.brain.car_angle)
        
        for i, target_item in enumerate(self.target_items):
            is_active = (i == self.brain.current_target_idx)
            target_item.set_active(is_active)
        
        self.scene.update()
        
        for i, coord in enumerate(self.brain.sensor_coords):
            self.sensor_items[i].setPos(coord)
            s_val = self.brain.get_state()[0][i]
            self.sensor_items[i].set_detecting(s_val > 0.5)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = TD3NavApp()
    win.show()
    sys.exit(app.exec())
