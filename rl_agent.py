import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from typing import List, Tuple, Dict, Optional, Any
from collections import deque
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import os

from sokoban_env import SokobanEnv

class CNNPolicy(nn.Module):
    """
    CNN 기반 Actor-Critic 신경망
    소코반 게임 보드를 입력으로 받아 행동 확률과 가치를 출력합니다.
    """
    
    def __init__(self, input_shape: Tuple[int, int], action_size: int = 4, hidden_size: int = 256):
        super(CNNPolicy, self).__init__()
        
        self.input_shape = input_shape
        self.action_size = action_size
        
        # CNN 레이어들
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        # Feature map 크기 계산
        conv_out_size = input_shape[0] * input_shape[1] * 64
        
        # Fully connected 레이어들
        self.fc_common = nn.Linear(conv_out_size, hidden_size)
        
        # Actor 헤드 (정책)
        self.fc_actor = nn.Linear(hidden_size, hidden_size // 2)
        self.action_head = nn.Linear(hidden_size // 2, action_size)
        
        # Critic 헤드 (가치)
        self.fc_critic = nn.Linear(hidden_size, hidden_size // 2)
        self.value_head = nn.Linear(hidden_size // 2, 1)
        
        # 가중치 초기화
        self._initialize_weights()
    
    def _initialize_weights(self):
        """가중치 초기화"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """순전파"""
        # 입력 형태 조정: (batch, height, width) -> (batch, 1, height, width)
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        
        # CNN 특성 추출
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # 공통 특성
        common_features = F.relu(self.fc_common(x))
        
        # Actor (정책)
        actor_features = F.relu(self.fc_actor(common_features))
        action_logits = self.action_head(actor_features)
        action_probs = F.softmax(action_logits, dim=-1)
        
        # Critic (가치)
        critic_features = F.relu(self.fc_critic(common_features))
        state_value = self.value_head(critic_features)
        
        return action_probs, state_value
    
    def get_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[int, float, float]:
        """상태에서 행동 선택"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_probs, state_value = self.forward(state_tensor)
            
            if deterministic:
                action = torch.argmax(action_probs, dim=-1).item()
                log_prob = torch.log(action_probs[0, action]).item()
            else:
                dist = torch.distributions.Categorical(action_probs)
                action = dist.sample().item()
                log_prob = dist.log_prob(torch.tensor(action)).item()
            
            return action, log_prob, state_value.item()


class PPOAgent:
    """
    PPO (Proximal Policy Optimization) 에이전트
    소코반 퍼즐을 해결하는 강화학습 에이전트입니다.
    """
    
    def __init__(self, 
                 input_shape: Tuple[int, int],
                 action_size: int = 4,
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 eps_clip: float = 0.2,
                 k_epochs: int = 4,
                 value_coef: float = 0.5,
                 entropy_coef: float = 0.01):
        
        self.action_size = action_size
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        
        # 디바이스 설정
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"PPO Agent가 {self.device}에서 실행됩니다.")
        
        # 정책 네트워크
        self.policy = CNNPolicy(input_shape, action_size).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # 경험 버퍼
        self.memory = PPOMemory()
        
        # 훈련 통계
        self.training_stats = {
            'episode_rewards': [],
            'episode_lengths': [],
            'policy_losses': [],
            'value_losses': [],
            'total_losses': []
        }
    
    def select_action(self, state: np.ndarray, training: bool = True) -> Tuple[int, float, float]:
        """행동 선택"""
        return self.policy.get_action(state, deterministic=not training)
    
    def store_transition(self, state: np.ndarray, action: int, reward: float, 
                        next_state: np.ndarray, done: bool, log_prob: float, value: float):
        """경험을 메모리에 저장"""
        self.memory.push(state, action, reward, next_state, done, log_prob, value)
    
    def update(self) -> Dict[str, float]:
        """PPO 업데이트"""
        if len(self.memory) == 0:
            return {'policy_loss': 0, 'value_loss': 0, 'total_loss': 0}
        
        # 메모리에서 경험 가져오기
        states, actions, rewards, next_states, dones, old_log_probs, old_values = self.memory.get_all()
        
        # 디스카운트된 보상 계산
        discounted_rewards = self._compute_discounted_rewards(rewards, dones)
        
        # 어드밴티지 계산
        advantages = discounted_rewards - old_values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 텐서로 변환
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
        discounted_rewards = torch.FloatTensor(discounted_rewards).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        
        # PPO 업데이트 수행
        total_policy_loss = 0
        total_value_loss = 0
        total_loss = 0
        
        for _ in range(self.k_epochs):
            # 현재 정책으로 행동 확률과 가치 계산
            action_probs, state_values = self.policy(states)
            state_values = state_values.squeeze()
            
            # 행동 확률 분포 생성
            dist = torch.distributions.Categorical(action_probs)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy()
            
            # 중요도 비율 계산
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # PPO 클립된 목적 함수
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # 가치 손실
            value_loss = F.mse_loss(state_values, discounted_rewards)
            
            # 엔트로피 보너스
            entropy_loss = -entropy.mean()
            
            # 총 손실
            loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
            
            # 역전파
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_loss += loss.item()
        
        # 메모리 정리
        self.memory.clear()
        
        loss_dict = {
            'policy_loss': total_policy_loss / self.k_epochs,
            'value_loss': total_value_loss / self.k_epochs,
            'total_loss': total_loss / self.k_epochs
        }
        
        # 통계 업데이트
        self.training_stats['policy_losses'].append(loss_dict['policy_loss'])
        self.training_stats['value_losses'].append(loss_dict['value_loss'])
        self.training_stats['total_losses'].append(loss_dict['total_loss'])
        
        return loss_dict
    
    def _compute_discounted_rewards(self, rewards: List[float], dones: List[bool]) -> List[float]:
        """디스카운트된 보상 계산"""
        discounted_rewards = []
        running_reward = 0
        
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                running_reward = 0
            running_reward = reward + self.gamma * running_reward
            discounted_rewards.insert(0, running_reward)
        
        return discounted_rewards
    
    def train(self, env: SokobanEnv, num_episodes: int = 1000, 
              max_steps_per_episode: int = 200, update_interval: int = 2048,
              save_interval: int = 100, model_path: str = "sokoban_ppo.pth"):
        """에이전트 훈련"""
        
        print(f"PPO 에이전트 훈련 시작: {num_episodes} 에피소드")
        
        step_count = 0
        best_reward = float('-inf')
        
        for episode in tqdm(range(num_episodes), desc="Training"):
            state, _ = env.reset()
            episode_reward = 0
            episode_length = 0
            
            for step in range(max_steps_per_episode):
                # 행동 선택
                action, log_prob, value = self.select_action(state, training=True)
                
                # 환경에서 행동 수행
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                # 경험 저장
                self.store_transition(state, action, reward, next_state, done, log_prob, value)
                
                state = next_state
                episode_reward += reward
                episode_length += 1
                step_count += 1
                
                # 업데이트 수행
                if step_count % update_interval == 0:
                    self.update()
                
                if done:
                    break
            
            # 에피소드 통계 업데이트
            self.training_stats['episode_rewards'].append(episode_reward)
            self.training_stats['episode_lengths'].append(episode_length)
            
            # 최고 성능 모델 저장
            if episode_reward > best_reward:
                best_reward = episode_reward
                self.save_model(f"best_{model_path}")
            
            # 주기적 모델 저장
            if (episode + 1) % save_interval == 0:
                self.save_model(model_path)
                print(f"에피소드 {episode + 1}: 평균 보상 = {np.mean(self.training_stats['episode_rewards'][-100:]):.2f}")
        
        # 최종 모델 저장
        self.save_model(model_path)
        print("훈련 완료!")
        
        return self.training_stats
    
    def evaluate(self, env: SokobanEnv, num_episodes: int = 10, render: bool = False) -> Dict[str, float]:
        """에이전트 평가"""
        total_rewards = []
        total_steps = []
        success_count = 0
        
        for episode in range(num_episodes):
            state, _ = env.reset()
            episode_reward = 0
            episode_steps = 0
            
            while episode_steps < 500:  # 최대 스텝 제한
                action, _, _ = self.select_action(state, training=False)
                state, reward, terminated, truncated, info = env.step(action)
                
                episode_reward += reward
                episode_steps += 1
                
                if render:
                    print(env.render())
                    print(f"Action: {action}, Reward: {reward}")
                    print("-" * 40)
                
                if terminated:
                    if info.get('is_solved', False):
                        success_count += 1
                    break
                elif truncated:
                    break
            
            total_rewards.append(episode_reward)
            total_steps.append(episode_steps)
        
        return {
            'mean_reward': np.mean(total_rewards),
            'std_reward': np.std(total_rewards),
            'mean_steps': np.mean(total_steps),
            'std_steps': np.std(total_steps),
            'success_rate': success_count / num_episodes
        }
    
    def save_model(self, path: str):
        """모델 저장"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_stats': self.training_stats
        }, path)
    
    def load_model(self, path: str):
        """모델 로드"""
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'training_stats' in checkpoint:
                self.training_stats = checkpoint['training_stats']
            print(f"모델을 {path}에서 로드했습니다.")
        else:
            print(f"모델 파일 {path}을 찾을 수 없습니다.")
    
    def get_play_data(self, env: SokobanEnv) -> Dict[str, Any]:
        """플레이 데이터 수집 (난이도 평가용)"""
        state, _ = env.reset()
        
        play_data = {
            'total_steps': 0,
            'total_reward': 0,
            'box_pushes': 0,
            'policy_entropy': [],
            'value_estimates': [],
            'action_sequence': [],
            'solved': False,
            'inference_times': []
        }
        
        import time
        
        for step in range(500):  # 최대 스텝
            start_time = time.time()
            
            # 행동 선택 및 정책 정보 수집
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                action_probs, state_value = self.policy(state_tensor)
                
                # 엔트로피 계산
                dist = torch.distributions.Categorical(action_probs)
                entropy = dist.entropy().item()
                
                action = dist.sample().item()
                
            inference_time = time.time() - start_time
            
            # 환경에서 행동 수행
            next_state, reward, terminated, truncated, info = env.step(action)
            
            # 데이터 수집
            play_data['total_steps'] += 1
            play_data['total_reward'] += reward
            play_data['policy_entropy'].append(entropy)
            play_data['value_estimates'].append(state_value.item())
            play_data['action_sequence'].append(action)
            play_data['inference_times'].append(inference_time)
            
            # 박스 푸시 카운트 (보상이 1 이상인 경우)
            if reward >= 1:
                play_data['box_pushes'] += 1
            
            state = next_state
            
            if terminated:
                play_data['solved'] = info.get('is_solved', False)
                break
            elif truncated:
                break
        
        # 통계 계산
        if play_data['policy_entropy']:
            play_data['mean_entropy'] = np.mean(play_data['policy_entropy'])
            play_data['std_entropy'] = np.std(play_data['policy_entropy'])
        
        if play_data['value_estimates']:
            play_data['mean_value'] = np.mean(play_data['value_estimates'])
            play_data['std_value'] = np.std(play_data['value_estimates'])
        
        if play_data['inference_times']:
            play_data['mean_inference_time'] = np.mean(play_data['inference_times'])
        
        return play_data


class PPOMemory:
    """PPO를 위한 경험 메모리"""
    
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.log_probs = []
        self.values = []
    
    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool, log_prob: float, value: float):
        """경험 추가"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
    
    def get_all(self) -> Tuple[List, List, List, List, List, List, List]:
        """모든 경험 반환"""
        return (self.states, self.actions, self.rewards, self.next_states, 
                self.dones, self.log_probs, self.values)
    
    def clear(self):
        """메모리 정리"""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.next_states.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.values.clear()
    
    def __len__(self):
        return len(self.states)


def create_curriculum_levels() -> List[np.ndarray]:
    """커리큘럼 학습을 위한 쉬운 레벨들 생성"""
    levels = []
    
    # 레벨 1: 매우 간단한 레벨
    level1 = np.array([
        [1, 1, 1, 1, 1],
        [1, 4, 2, 3, 1],
        [1, 1, 1, 1, 1]
    ])
    levels.append(level1)
    
    # 레벨 2: 조금 더 복잡한 레벨
    level2 = np.array([
        [1, 1, 1, 1, 1],
        [1, 4, 0, 2, 1],
        [1, 0, 0, 3, 1],
        [1, 1, 1, 1, 1]
    ])
    levels.append(level2)
    
    # 레벨 3: 더 복잡한 레벨
    level3 = np.array([
        [1, 1, 1, 1, 1, 1],
        [1, 4, 0, 2, 3, 1],
        [1, 0, 0, 0, 0, 1],
        [1, 3, 2, 0, 0, 1],
        [1, 1, 1, 1, 1, 1]
    ])
    levels.append(level3)
    
    return levels


if __name__ == "__main__":
    # 기본 훈련 실행
    env = SokobanEnv()
    agent = PPOAgent(env.observation_space.shape)
    
    # 훈련 실행
    stats = agent.train(env, num_episodes=500)
    
    # 평가 실행
    eval_results = agent.evaluate(env, num_episodes=10)
    print("평가 결과:", eval_results) 