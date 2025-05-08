"""
경험 리플레이 메모리 모듈

PPO 알고리즘의 학습에 필요한 경험(transition)을 저장하고 관리하는 메모리 클래스를 구현합니다.
상태, 행동, 로그 확률, 보상, 종료 여부, 상태 가치 등의 정보를 저장합니다.
"""

class Memory:
    """
    PPO 학습을 위한 경험(Experience) 저장 버퍼입니다.
    NumPy 기반으로 상태, 행동, 로그 확률, 보상, 종료 여부, 상태 가치를 저장합니다.
    """

    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.values = []

    def clear_memory(self):
        """메모리에 저장된 모든 경험을 삭제합니다."""
        del self.states[:]
        del self.actions[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.values[:]

    def add_experience(self, state, action, logprob, reward, is_terminal, value):
        """새로운 경험을 메모리에 추가합니다."""
        self.states.append(state)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.is_terminals.append(is_terminal)
        self.values.append(value) 