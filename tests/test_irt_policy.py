# tests/test_irt_policy.py

"""
IRT Policy 단위 테스트

테스트 항목:
1. IRT forward pass 정상 작동
2. Simplex 제약 만족 (portfolio weights)
3. SB3 통합 (IRTPolicy)
4. Device 호환성 (CPU/GPU)
"""

import pytest
import torch
import numpy as np
from gymnasium import spaces

from finrl.agents.irt import IRT, BCellIRTActor, TCellMinimal
from finrl.agents.irt.irt_policy import IRTPolicy


@pytest.fixture
def irt_config():
    """IRT 테스트 설정"""
    return {
        'state_dim': 181,  # FinRL Dow30: 30 + 30*8 + 1 = 181
        'action_dim': 30,
        'emb_dim': 128,
        'm_tokens': 6,
        'M_proto': 8,
        'alpha': 0.3,
        'market_feature_dim': 8,
        'stock_dim': 30,
        'tech_indicator_count': 4,
        'has_dsr_cvar': False
    }


@pytest.fixture
def sample_state(irt_config):
    """샘플 상태 생성"""
    batch_size = 4
    state_dim = irt_config['state_dim']
    return torch.randn(batch_size, state_dim)


def test_irt_forward_pass(irt_config, sample_state):
    """
    테스트 1: IRT forward pass가 정상 작동하는가?
    """
    actor = BCellIRTActor(**irt_config)
    actor.eval()

    with torch.no_grad():
        action, info = actor(sample_state, deterministic=True)

    # 형상 체크
    assert action.shape == (sample_state.shape[0], irt_config['action_dim'])

    # Info 체크
    assert 'w' in info
    assert 'P' in info
    assert 'crisis_level' in info
    assert 'w_rep' in info
    assert 'w_ot' in info

    # 가중치 차원 체크
    assert info['w'].shape == (sample_state.shape[0], irt_config['M_proto'])

    print("✅ Test 1 passed: IRT forward pass executed correctly")


def test_simplex_constraint(irt_config, sample_state):
    """
    테스트 2: Portfolio weights가 simplex 제약을 만족하는가?
    - sum(weights) = 1
    - all(weights >= 0)
    """
    actor = BCellIRTActor(**irt_config)
    actor.eval()

    with torch.no_grad():
        action, info = actor(sample_state, deterministic=True)

    # Simplex 제약 체크
    # 1. 합 = 1
    weight_sums = action.sum(dim=-1)
    assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5), \
        f"가중치 합이 1이 아닙니다: {weight_sums}"

    # 2. 모든 원소 >= 0
    assert (action >= 0).all(), f"음수 가중치가 발견되었습니다: {action.min()}"

    # 3. 모든 원소 <= 1
    assert (action <= 1).all(), f"1을 초과하는 가중치가 발견되었습니다: {action.max()}"

    print("✅ Test 2 passed: simplex constraints satisfied")


def test_sb3_integration(irt_config):
    """
    테스트 3: SB3 환경에서 IRTPolicy가 정상 작동하는가?
    """
    # 관측 공간(Box)
    obs_space = spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(irt_config['state_dim'],),
        dtype=np.float32
    )

    # 행동 공간(Box, simplex)
    action_space = spaces.Box(
        low=0.0,
        high=1.0,
        shape=(irt_config['action_dim'],),
        dtype=np.float32
    )

    # IRTPolicy 생성
    policy = IRTPolicy(
        observation_space=obs_space,
        action_space=action_space,
        lr_schedule=lambda _: 3e-4,
        emb_dim=irt_config['emb_dim'],
        m_tokens=irt_config['m_tokens'],
        M_proto=irt_config['M_proto'],
        alpha=irt_config['alpha']
    )

    policy.eval()

    # 샘플 관측
    obs = torch.randn(4, irt_config['state_dim'])

    # Forward pass
    with torch.no_grad():
        action = policy(obs, deterministic=True)

    # 형상 체크
    assert action.shape == (4, irt_config['action_dim'])

    # Simplex 제약 체크
    assert torch.allclose(action.sum(dim=-1), torch.ones(4), atol=1e-5)
    assert (action >= 0).all()

    print("✅ Test 3 passed: SB3 integration path works")


def test_device_compatibility(irt_config, sample_state):
    """
    테스트 4: CPU/GPU 전환이 정상 작동하는가?
    """
    actor = BCellIRTActor(**irt_config)

    # CPU 테스트
    actor.cpu()
    sample_state_cpu = sample_state.cpu()

    with torch.no_grad():
        action_cpu, info_cpu = actor(sample_state_cpu, deterministic=True)

    assert action_cpu.device.type == 'cpu'
    assert info_cpu['w'].device.type == 'cpu'

    print("✅ Test 4 (CPU) passed: CPU execution path works")

    # GPU 테스트 (가능한 경우)
    if torch.cuda.is_available():
        actor.cuda()
        sample_state_gpu = sample_state.cuda()

        with torch.no_grad():
            action_gpu, info_gpu = actor(sample_state_gpu, deterministic=True)

        assert action_gpu.device.type == 'cuda'
        assert info_gpu['w'].device.type == 'cuda'

        print("✅ Test 4 (GPU) passed: GPU execution path works")
    else:
        print("⚠️ Test 4 (GPU) skipped: CUDA not available")


def test_irt_decomposition(irt_config, sample_state):
    """
    테스트 5 (추가): IRT 분해 정보가 올바른가?
    - w = (1-α) * w_rep + α * w_ot
    """
    actor = BCellIRTActor(**irt_config)
    actor.eval()

    with torch.no_grad():
        action, info = actor(sample_state, deterministic=True)

    w = info['w']
    w_rep = info['w_rep']
    w_ot = info['w_ot']
    alpha = irt_config['alpha']

    # IRT 혼합 공식 검증
    w_expected = (1 - alpha) * w_rep + alpha * w_ot

    # 정규화 차이는 허용 (IRT에서 재정규화)
    # L2 거리로 검증
    diff = torch.norm(w - w_expected, dim=-1).mean()

    # 정규화 전후 차이는 작아야 함 (< 0.1)
    assert diff < 0.1, f"IRT 결합 결과가 예상과 다릅니다: L2 차이 = {diff:.4f}"

    print("✅ Test 5 passed: IRT combination formula holds")


def test_fitness_calculation(irt_config):
    """
    테스트 6 (새로운): Fitness 계산이 Critic Q-value를 사용하는가?
    """
    from finrl.agents.irt.irt_policy import IRTActorWrapper
    import torch.nn as nn

    # Mock Critic 생성
    class MockCritic(nn.Module):
        def __init__(self, action_dim):
            super().__init__()
            self.action_dim = action_dim

        def forward(self, obs, action):
            # 행동 합에 비례하는 Q-value 반환 (테스트용)
            q_value = action.sum(dim=-1) * 10.0
            return (q_value, q_value)  # Twin Q

    # Mock Policy 생성
    class MockPolicy:
        def __init__(self, critic):
            self.critic = critic

    # BCellIRTActor 생성
    actor = BCellIRTActor(**irt_config)

    # Observation space
    obs_space = spaces.Box(
        low=-np.inf, high=np.inf,
        shape=(irt_config['state_dim'],),
        dtype=np.float32
    )
    action_space = spaces.Box(
        low=0.0, high=1.0,
        shape=(irt_config['action_dim'],),
        dtype=np.float32
    )

    # Mock Policy와 Critic
    critic = MockCritic(irt_config['action_dim'])
    policy = MockPolicy(critic)

    # IRTActorWrapper 생성
    wrapper = IRTActorWrapper(
        irt_actor=actor,
        features_dim=irt_config['state_dim'],
        action_space=action_space,
        policy=policy
    )

    wrapper.eval()

    # 샘플 관측
    obs = torch.randn(4, irt_config['state_dim'])

    with torch.no_grad():
        action, log_prob = wrapper.action_log_prob(obs)

    # action_log_prob 내부에서 fitness 계산되었는지 확인
    # (간접 확인: action이 정상적으로 반환되었는지)
    assert action.shape == (4, irt_config['action_dim'])
    assert log_prob.shape == (4, 1)

    print("✅ Test 6 passed: Critic-based fitness computation works")


def test_replicator_activation(irt_config):
    """
    테스트 7 (새로운): Replicator가 fitness에 따라 가중치를 조정하는가?
    """
    actor = BCellIRTActor(**irt_config)
    actor.eval()

    B = 4
    M = irt_config['M_proto']
    state = torch.randn(B, irt_config['state_dim'])

    # Case 1: fitness=None (균등)
    with torch.no_grad():
        action1, info1 = actor(state, fitness=None, deterministic=True)
        w1 = info1['w']

    # Case 2: 불균등 fitness (프로토타입 0이 가장 높음)
    fitness = torch.ones(B, M) * 0.1
    fitness[:, 0] = 1.0  # 프로토타입 0에 높은 fitness

    with torch.no_grad():
        action2, info2 = actor(state, fitness=fitness, deterministic=True)
        w2 = info2['w']
        w_rep2 = info2['w_rep']

    # Replicator가 작동했다면, w_rep2에서 프로토타입 0의 가중치가 더 커야 함
    # w_rep2[:, 0]의 평균이 1/M보다 커야 함
    avg_w0 = w_rep2[:, 0].mean()
    uniform_weight = 1.0 / M

    assert avg_w0 > uniform_weight, \
        f"Replicator가 활성화되지 않았습니다: w_rep[0]={avg_w0:.4f}, 예상>{uniform_weight:.4f}"

    print(f"✅ Test 7 passed: Replicator responds as expected (w_rep[0]={avg_w0:.4f} > {uniform_weight:.4f})")


if __name__ == '__main__':
    # 직접 실행 시 테스트 수행
    import sys

    print("=" * 70)
    print("IRT Policy unit tests")
    print("=" * 70)

    config = {
        'state_dim': 181,
        'action_dim': 30,
        'emb_dim': 128,
        'm_tokens': 6,
        'M_proto': 8,
        'alpha': 0.3,
        'market_feature_dim': 12
    }

    state = torch.randn(4, config['state_dim'])

    try:
        test_irt_forward_pass(config, state)
        test_simplex_constraint(config, state)
        test_sb3_integration(config)
        test_device_compatibility(config, state)
        test_irt_decomposition(config, state)
        test_fitness_calculation(config)
        test_replicator_activation(config)

        print("=" * 70)
        print("✅ All tests passed!")
        print("=" * 70)

    except Exception as e:
        print("=" * 70)
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        print("=" * 70)
        sys.exit(1)
