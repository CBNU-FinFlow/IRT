from __future__ import annotations

from typing import Dict, List, Optional

import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gymnasium import spaces
from gymnasium.utils import seeding
from stable_baselines3.common.vec_env import DummyVecEnv

matplotlib.use("Agg")

# from stable_baselines3.common.logger import Logger, KVWriter, CSVOutputFormat


class StockTradingEnv(gym.Env):
    """
    A stock trading environment for OpenAI gym

    Parameters:
        df (pandas.DataFrame): Dataframe containing data
        hmax (int): Maximum cash to be traded in each trade per asset.
        initial_amount (int): Amount of cash initially available
        buy_cost_pct (float, array): Cost for buying shares, each index corresponds to each asset
        sell_cost_pct (float, array): Cost for selling shares, each index corresponds to each asset
        turbulence_threshold (float): Maximum turbulence allowed in market for purchases to occur. If exceeded, positions are liquidated
        print_verbosity(int): When iterating (step), how often to print stats about state of env
        use_weighted_action (bool): If True (default), interpret actions as portfolio weights and expose turnover telemetry
        adaptive_lambda_sharpe (float): Base κ weight for adaptive risk reward (default: 0.20)
        adaptive_lambda_cvar (float): CVaR penalty weight for adaptive risk reward (default: 0.40)
        adaptive_lambda_turnover (float): Turnover penalty weight for adaptive risk reward (default: 0.0)
        adaptive_crisis_gain_sharpe (float): Crisis gain g_s for Sharpe term (keep negative, default: -0.15)
        adaptive_crisis_gain_cvar (float): Crisis gain g_c for CVaR penalty (default: 0.25)
        adaptive_dsr_beta (float): DSR EMA β for adaptive risk reward (default: 0.92)
        adaptive_cvar_window (int): CVaR estimation window for adaptive risk reward (default: 40)
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        stock_dim: int,
        hmax: int,
        initial_amount: int,
        num_stock_shares: list[int],
        buy_cost_pct: list[float],
        sell_cost_pct: list[float],
        reward_scaling: float,
        state_space: int,
        action_space: int,
        tech_indicator_list: list[str],
        turbulence_threshold=None,
        risk_indicator_col="turbulence",
        make_plots: bool = False,
        print_verbosity=10,
        day=0,
        initial=True,
        previous_state=[],
        model_name="",
        mode="",
        iteration="",
        # DSR + CVaR 보상 관련 파라미터
        reward_type: str = "basic",
        lambda_dsr: float = 0.1,
        lambda_cvar: float = 0.05,
        dsr_beta: float = 0.99,
        cvar_alpha: float = 0.05,
        cvar_window: int = 50,
        use_weighted_action: bool = True,
        weight_slippage: float = 0.001,
        weight_transaction_cost: float = 0.0005,
        adaptive_lambda_sharpe: float = 0.20,
        adaptive_lambda_cvar: float = 0.40,
        adaptive_lambda_turnover: float = 0.0,
        adaptive_crisis_gain_sharpe: float = -0.15,
        adaptive_crisis_gain_cvar: float = 0.25,
        adaptive_dsr_beta: float = 0.92,
        adaptive_cvar_window: int = 40,
    ):
        self.day = day
        self.df = df
        if isinstance(self.df.index, pd.MultiIndex) and "tic" in self.df.index.names:
            self.tickers = (
                self.df.index.get_level_values("tic").unique().tolist()
            )
        elif "tic" in self.df.columns:
            self.tickers = self.df["tic"].drop_duplicates().tolist()
        else:
            self.tickers = [f"asset_{idx}" for idx in range(stock_dim)]
        self.stock_dim = stock_dim
        self.hmax = hmax
        self.num_stock_shares = num_stock_shares
        self.initial_amount = initial_amount  # get the initial cash
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.reward_scaling = reward_scaling
        self.use_weighted_action = use_weighted_action
        self.weight_slippage = weight_slippage
        self.weight_transaction_cost = weight_transaction_cost
        self.adaptive_lambda_sharpe = adaptive_lambda_sharpe
        self.adaptive_lambda_cvar = adaptive_lambda_cvar
        self.adaptive_lambda_turnover = adaptive_lambda_turnover
        self.adaptive_crisis_gain_sharpe = adaptive_crisis_gain_sharpe
        self.adaptive_crisis_gain_cvar = adaptive_crisis_gain_cvar
        self.adaptive_dsr_beta = adaptive_dsr_beta
        self.adaptive_cvar_window = adaptive_cvar_window
        self.weights_memory: list[np.ndarray] = []
        self.executed_weights_memory: list[np.ndarray] = []
        self.turnover_memory: list[float] = []
        self._last_turnover: float = 0.0
        self._last_tc: float = 0.0
        self._last_turnover_executed: float = 0.0
        self._crisis_level: float = 0.5
        self._last_bridge_step: int = -1
        self._crisis_history: list[tuple[int, float]] = []
        self._last_reward_info: dict[str, float] = {}

        self.last_kappa_cvar = 0.0
        self.reward_type = reward_type

        if self.reward_type == "adaptive_risk" and not self.use_weighted_action:
            raise ValueError(
                "Adaptive risk reward requires weight-based actions. "
                "Set use_weighted_action=True."
            )

        # 보상 유형에 따라 상태 공간에 추가 신호를 포함시킨다.
        if self.reward_type == "adaptive_risk":
            # ΔSharpe, CVaR, crisis_level 노출
            self.state_space = state_space + 3
        elif self.reward_type == "dsr_cvar":
            self.state_space = state_space + 2  # DSR/CVaR 신호 추가
        else:
            self.state_space = state_space

        self.action_space = action_space
        self.tech_indicator_list = tech_indicator_list
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_space,))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_space,)
        )
        self.data = self.df.loc[self.day, :]
        self.terminal = False
        self.make_plots = make_plots
        self.print_verbosity = print_verbosity
        self.turbulence_threshold = turbulence_threshold
        self.risk_indicator_col = risk_indicator_col
        self.initial = initial
        self.previous_state = previous_state
        self.model_name = model_name
        self.mode = mode
        self.iteration = iteration

        # 리스크 민감 보상 함수를 state 초기화 전에 구성한다.
        if self.reward_type == "dsr_cvar":
            from .reward_functions import RiskSensitiveReward

            self.risk_reward = RiskSensitiveReward(
                lambda_dsr=lambda_dsr,
                lambda_cvar=lambda_cvar,
                dsr_beta=dsr_beta,
                cvar_alpha=cvar_alpha,
                cvar_window=cvar_window,
            )

            # DSR/CVaR 신호를 상태 벡터에 포함하기 위한 버퍼
            self.last_dsr = 0.0
            self.last_cvar = 0.0
        elif self.reward_type == "adaptive_risk":
            # 위기 민감형 보상을 사용하는 경우
            from .reward_functions import AdaptiveRiskReward

            self.risk_reward = AdaptiveRiskReward(
                lambda_sharpe=self.adaptive_lambda_sharpe,
                lambda_cvar=self.adaptive_lambda_cvar,
                lambda_turnover=self.adaptive_lambda_turnover,
                crisis_gain_sharpe=self.adaptive_crisis_gain_sharpe,
                crisis_gain_cvar=self.adaptive_crisis_gain_cvar,
                dsr_beta=self.adaptive_dsr_beta,
                cvar_alpha=cvar_alpha,
                cvar_window=self.adaptive_cvar_window,
            )

            # 위기 레벨 및 보상 계수 추적용 변수
            self.last_crisis_level = 0.5
            self.last_kappa = self.adaptive_lambda_sharpe
            self.last_kappa_cvar = self.adaptive_lambda_cvar
            self.last_delta_sharpe = 0.0
            self.last_cvar = 0.0
        else:
            self.risk_reward = None

        # initalize state (reward_type 설정 후 호출)
        self.state = self._initiate_state()

        # initialize reward
        self.reward = 0
        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.episode = 0
        # memorize all the total balance change
        self.asset_memory = [
            self.initial_amount
            + np.sum(
                np.array(self.num_stock_shares)
                * np.array(self.state[1 : 1 + self.stock_dim])
            )
        ]  # the initial total asset is calculated by cash + sum (num_share_stock_i * price_stock_i)
        self.rewards_memory = []
        self.actions_memory = []
        self.state_memory = (
            []
        )  # we need sometimes to preserve the state in the middle of trading process
        self.date_memory = [self._get_date()]
        #         self.logger = Logger('results',[CSVOutputFormat])
        # self.reset()
        self._seed()

    def _sell_stock(self, index, action):
        def _do_sell_normal():
            if (
                self.state[index + 2 * self.stock_dim + 1] != True
            ):  # check if the stock is able to sell, for simlicity we just add it in techical index
                # if self.state[index + 1] > 0: # if we use price<0 to denote a stock is unable to trade in that day, the total asset calculation may be wrong for the price is unreasonable
                # Sell only if the price is > 0 (no missing data in this particular date)
                # perform sell action based on the sign of the action
                if self.state[index + self.stock_dim + 1] > 0:
                    # Sell only if current asset is > 0
                    sell_num_shares = min(
                        abs(action), self.state[index + self.stock_dim + 1]
                    )
                    sell_amount = (
                        self.state[index + 1]
                        * sell_num_shares
                        * (1 - self.sell_cost_pct[index])
                    )
                    # update balance
                    self.state[0] += sell_amount

                    self.state[index + self.stock_dim + 1] -= sell_num_shares
                    self.cost += (
                        self.state[index + 1]
                        * sell_num_shares
                        * self.sell_cost_pct[index]
                    )
                    self.trades += 1
                else:
                    sell_num_shares = 0
            else:
                sell_num_shares = 0

            return sell_num_shares

        # perform sell action based on the sign of the action
        if self.turbulence_threshold is not None:
            if self.turbulence >= self.turbulence_threshold:
                if self.state[index + 1] > 0:
                    # Sell only if the price is > 0 (no missing data in this particular date)
                    # if turbulence goes over threshold, just clear out all positions
                    if self.state[index + self.stock_dim + 1] > 0:
                        # Sell only if current asset is > 0
                        sell_num_shares = self.state[index + self.stock_dim + 1]
                        sell_amount = (
                            self.state[index + 1]
                            * sell_num_shares
                            * (1 - self.sell_cost_pct[index])
                        )
                        # update balance
                        self.state[0] += sell_amount
                        self.state[index + self.stock_dim + 1] = 0
                        self.cost += (
                            self.state[index + 1]
                            * sell_num_shares
                            * self.sell_cost_pct[index]
                        )
                        self.trades += 1
                    else:
                        sell_num_shares = 0
                else:
                    sell_num_shares = 0
            else:
                sell_num_shares = _do_sell_normal()
        else:
            sell_num_shares = _do_sell_normal()

        return sell_num_shares

    def _buy_stock(self, index, action):
        def _do_buy():
            if (
                self.state[index + 2 * self.stock_dim + 1] != True
            ):  # check if the stock is able to buy
                # if self.state[index + 1] >0:
                # Buy only if the price is > 0 (no missing data in this particular date)
                available_amount = self.state[0] // (
                    self.state[index + 1] * (1 + self.buy_cost_pct[index])
                )  # when buying stocks, we should consider the cost of trading when calculating available_amount, or we may be have cash<0
                # print('available_amount:{}'.format(available_amount))

                # update balance
                buy_num_shares = min(available_amount, action)
                buy_amount = (
                    self.state[index + 1]
                    * buy_num_shares
                    * (1 + self.buy_cost_pct[index])
                )
                self.state[0] -= buy_amount

                self.state[index + self.stock_dim + 1] += buy_num_shares

                self.cost += (
                    self.state[index + 1] * buy_num_shares * self.buy_cost_pct[index]
                )
                self.trades += 1
            else:
                buy_num_shares = 0

            return buy_num_shares

        # perform buy action based on the sign of the action
        if self.turbulence_threshold is None:
            buy_num_shares = _do_buy()
        else:
            if self.turbulence < self.turbulence_threshold:
                buy_num_shares = _do_buy()
            else:
                buy_num_shares = 0
                pass

        return buy_num_shares

    def _make_plot(self):
        plt.plot(self.asset_memory, "r")
        plt.savefig(f"results/account_value_trade_{self.episode}.png")
        plt.close()

    def step(self, actions):
        self.terminal = self.day >= len(self.df.index.unique()) - 1
        if self.terminal:
            return self._step_terminal()

        if self.use_weighted_action:
            return self._step_weight_actions(actions)

        return self._step_share_actions(actions)

    def _step_terminal(self):
        if self.make_plots:
            self._make_plot()
        end_total_asset = self.state[0] + sum(
            np.array(self.state[1 : (self.stock_dim + 1)])
            * np.array(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
        )
        df_total_value = pd.DataFrame(self.asset_memory)
        tot_reward = (
            self.state[0]
            + sum(
                np.array(self.state[1 : (self.stock_dim + 1)])
                * np.array(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
            )
            - self.asset_memory[0]
        )
        df_total_value.columns = ["account_value"]
        df_total_value["date"] = self.date_memory
        df_total_value["daily_return"] = df_total_value["account_value"].pct_change(1)
        sharpe = None
        if df_total_value["daily_return"].std() != 0:
            sharpe = (
                (252**0.5)
                * df_total_value["daily_return"].mean()
                / df_total_value["daily_return"].std()
            )
        df_rewards = pd.DataFrame(self.rewards_memory)
        df_rewards.columns = ["account_rewards"]
        df_rewards["date"] = self.date_memory[:-1]
        if self.episode % self.print_verbosity == 0:
            print(f"day: {self.day}, episode: {self.episode}")
            print(f"begin_total_asset: {self.asset_memory[0]:0.2f}")
            print(f"end_total_asset: {end_total_asset:0.2f}")
            print(f"total_reward: {tot_reward:0.2f}")
            print(f"total_cost: {self.cost:0.2f}")
            print(f"total_trades: {self.trades}")
            if sharpe is not None:
                print(f"Sharpe: {sharpe:0.3f}")
            print("=================================")

        if (self.model_name != "") and (self.mode != ""):
            df_actions = self.save_action_memory()
            df_actions.to_csv(
                "results/actions_{}_{}_{}.csv".format(
                    self.mode, self.model_name, self.iteration
                )
            )
            df_total_value.to_csv(
                "results/account_value_{}_{}_{}.csv".format(
                    self.mode, self.model_name, self.iteration
                ),
                index=False,
            )
            df_rewards.to_csv(
                "results/account_rewards_{}_{}_{}.csv".format(
                    self.mode, self.model_name, self.iteration
                ),
                index=False,
            )
            plt.plot(self.asset_memory, "r")
            plt.savefig(
                "results/account_value_{}_{}_{}.png".format(
                    self.mode, self.model_name, self.iteration
                )
            )
            plt.close()

        return self.state, self.reward, self.terminal, False, self._build_step_info(
            turnover_target=0.0
        )

    def _build_step_info(
        self,
        *,
        turnover_target: float,
        executed_weights: Optional[np.ndarray] = None,
        target_weights: Optional[np.ndarray] = None,
    ) -> Dict[str, object]:
        """
        Assemble per-step telemetry dictionary consumed by logging callbacks.
        """
        reward_info = self._last_reward_info or {}
        components = reward_info.get("components") or {}

        info: Dict[str, object] = {
            "turnover_target": float(turnover_target),
            "turnover_executed": float(getattr(self, "_last_turnover_executed", 0.0)),
            "turnover": float(getattr(self, "_last_turnover", 0.0)),
            "transaction_cost": float(getattr(self, "_last_tc", 0.0)),
            "reward_components": components.copy() if isinstance(components, dict) else {},
            "reward_log_return": float(
                reward_info.get(
                    "log_return",
                    components.get("log_return", 0.0),
                )
            ),
            "reward_total": float(
                reward_info.get(
                    "reward_total",
                    getattr(self, "reward", 0.0),
                )
            ),
            "reward_sharpe_term": float(components.get("sharpe_term", 0.0)),
            "reward_cvar_term": float(components.get("cvar_term", 0.0)),
            "cvar_value": float(reward_info.get("cvar_value", 0.0)),
            "crisis_level": float(
                reward_info.get("crisis_level", getattr(self, "_crisis_level", 0.5))
            ),
            "kappa_sharpe": float(
                reward_info.get(
                    "kappa_sharpe",
                    reward_info.get(
                        "kappa",
                        getattr(self, "adaptive_lambda_sharpe", 0.0),
                    ),
                )
            ),
            "kappa_cvar": float(
                reward_info.get(
                    "kappa_cvar",
                    getattr(self, "adaptive_lambda_cvar", 0.0),
                )
            ),
        }

        if executed_weights is not None:
            executed_np = np.asarray(executed_weights, dtype=np.float64).flatten()
            sum_weights = float(np.sum(executed_np))
            cash_weight = float(max(0.0, 1.0 - sum_weights))
            holdings = {}
            for idx, weight in enumerate(executed_np):
                ticker = self.tickers[idx] if idx < len(self.tickers) else f"asset_{idx}"
                holdings[ticker] = float(weight)
            holdings["CASH"] = cash_weight
            info.update(
                {
                    "weights_full": executed_np.tolist(),
                    "executed_weights": executed_np.tolist(),
                    "sum_weights": sum_weights,
                    "cash_weight": cash_weight,
                    "cash": float(max(self.state[0], 0.0)),
                    "holdings_full": holdings,
                }
            )

        if target_weights is not None:
            target_np = np.asarray(target_weights, dtype=np.float64).flatten()
            info["target_weights_full"] = target_np.tolist()
            info["target_weights"] = target_np.tolist()

        return info

    def _step_share_actions(self, actions):
        self._last_turnover = 0.0
        self._last_tc = 0.0
        self._last_turnover_executed = 0.0
        actions = actions * self.hmax  # actions initially is scaled between 0 to 1
        actions = actions.astype(
            int
        )  # convert into integer because we can't by fraction of shares
        if self.turbulence_threshold is not None:
            if self.turbulence >= self.turbulence_threshold:
                actions = np.array([-self.hmax] * self.stock_dim)
        begin_total_asset = self.state[0] + sum(
            np.array(self.state[1 : (self.stock_dim + 1)])
            * np.array(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
        )

        argsort_actions = np.argsort(actions)
        sell_index = argsort_actions[: np.where(actions < 0)[0].shape[0]]
        buy_index = argsort_actions[::-1][: np.where(actions > 0)[0].shape[0]]

        for index in sell_index:
            actions[index] = self._sell_stock(index, actions[index]) * (-1)

        for index in buy_index:
            actions[index] = self._buy_stock(index, actions[index])

        self.actions_memory.append(actions)

        # state: s -> s+1
        self.day += 1
        self.data = self.df.loc[self.day, :]
        if self.turbulence_threshold is not None:
            if len(self.df.tic.unique()) == 1:
                self.turbulence = self.data[self.risk_indicator_col]
            elif len(self.df.tic.unique()) > 1:
                self.turbulence = self.data[self.risk_indicator_col].values[0]
        self.state = self._update_state()

        end_total_asset = self.state[0] + sum(
            np.array(self.state[1 : (self.stock_dim + 1)])
            * np.array(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
        )
        self.asset_memory.append(end_total_asset)
        self.date_memory.append(self._get_date())

        # 리스크 민감 보상을 계산
        basic_reward = end_total_asset - begin_total_asset

        if self.reward_type == "dsr_cvar" and self.risk_reward is not None:
            log_return = np.log(end_total_asset / (begin_total_asset + 1e-8))
            risk_reward, reward_info = self.risk_reward.compute(log_return)
            self.last_dsr = float(reward_info["dsr_bonus"])
            self.last_cvar = float(reward_info["cvar_value"])
            self.reward = risk_reward * self.reward_scaling
            self._last_reward_info = reward_info.copy()

        elif self.reward_type == "adaptive_risk" and self.risk_reward is not None:
            log_return = np.log(end_total_asset / (begin_total_asset + 1e-8))
            prices = np.array(self.state[1:(self.stock_dim + 1)])
            holdings = np.array(self.state[(self.stock_dim + 1):(self.stock_dim * 2 + 1)])
            portfolio_value = end_total_asset + 1e-8
            current_weights = (prices * holdings) / portfolio_value

            risk_reward, reward_info = self.risk_reward.compute(log_return, current_weights)
            self.last_crisis_level = float(reward_info["crisis_level"])
            self.last_kappa = float(
                reward_info.get("kappa_sharpe", reward_info.get("kappa", self.adaptive_lambda_sharpe))
            )
            self.last_kappa_cvar = float(
                reward_info.get("kappa_cvar", self.adaptive_lambda_cvar)
            )
            self.last_delta_sharpe = float(reward_info["delta_sharpe"])
            self.last_cvar = float(reward_info.get("cvar_value", 0.0))
            self._crisis_level = self.last_crisis_level
            self.reward = risk_reward * self.reward_scaling
            self._last_reward_info = reward_info.copy()

        else:
            self.reward = basic_reward * self.reward_scaling
            self._last_reward_info = {
                "log_return": np.log(
                    max(end_total_asset, 1e-8) / max(begin_total_asset, 1e-8)
                ),
                "delta_sharpe": 0.0,
                "cvar_value": 0.0,
                "cvar_penalty": 0.0,
                "crisis_level": getattr(self, "_crisis_level", 0.5),
                "kappa": getattr(self, "adaptive_lambda_sharpe", 0.0),
                "kappa_sharpe": getattr(self, "adaptive_lambda_sharpe", 0.0),
                "kappa_cvar": getattr(self, "adaptive_lambda_cvar", 0.0),
                "risk_bonus": 0.0,
                "turnover_penalty": 0.0,
                "sharpe_online": 0.0,
                "reward_pre_clip": basic_reward,
                "reward_total": np.clip(basic_reward, -1.0, 1.0),
                "components": {
                    "log_return": np.log(
                        max(end_total_asset, 1e-8) / max(begin_total_asset, 1e-8)
                    ),
                    "sharpe_term": 0.0,
                    "cvar_term": 0.0,
                    "turnover": 0.0,
                },
            }

        self.rewards_memory.append(self.reward)
        self.state_memory.append(self.state)

        return self.state, self.reward, self.terminal, False, self._build_step_info(
            turnover_target=self._last_turnover
        )

    def _step_weight_actions(self, actions):
        action_vector = np.asarray(actions, dtype=np.float64).flatten()
        if action_vector.size != self.stock_dim:
            raise ValueError(
                f"Expected action dimension {self.stock_dim}, received {action_vector.size}"
            )
        action_vector = np.clip(action_vector, 0.0, 1.0)
        sum_actions = float(np.sum(action_vector))
        if sum_actions <= 1e-12:
            # 전액 현금 보유
            target_weights = np.zeros_like(action_vector)
        else:
            scale = max(1.0, sum_actions)
            target_weights = action_vector / scale
        if not np.all(np.isfinite(target_weights)):
            target_weights = np.ones_like(action_vector) / len(action_vector)

        if self.executed_weights_memory:
            prev_weights = np.asarray(self.executed_weights_memory[-1], dtype=np.float64)
        else:
            prev_weights = np.ones(self.stock_dim, dtype=np.float64) / self.stock_dim

        delta = target_weights - prev_weights
        prev_cash_weight = max(0.0, 1.0 - float(np.sum(prev_weights)))
        target_cash_weight = max(0.0, 1.0 - float(np.sum(target_weights)))
        turnover_target = 0.5 * float(
            np.sum(np.abs(delta)) + abs(target_cash_weight - prev_cash_weight)
        )
        executed_weights = prev_weights + (1.0 - self.weight_slippage) * delta
        executed_weights = np.clip(executed_weights, 0.0, 1.0)
        sum_executed = executed_weights.sum()
        if sum_executed <= 0 or not np.isfinite(sum_executed):
            executed_weights = np.ones_like(executed_weights) / len(executed_weights)
        elif sum_executed > 1.0:
            executed_weights /= sum_executed
        if not np.all(np.isfinite(executed_weights)):
            executed_weights = np.ones_like(executed_weights) / len(executed_weights)

        executed_cash_weight = max(0.0, 1.0 - float(np.sum(executed_weights)))
        actual_turnover = 0.5 * float(
            np.sum(np.abs(executed_weights - prev_weights))
            + abs(executed_cash_weight - prev_cash_weight)
        )
        self.weights_memory.append(target_weights.copy())
        self.executed_weights_memory.append(executed_weights.copy())
        self.turnover_memory.append(turnover_target)
        self._last_turnover = turnover_target
        self._last_turnover_executed = actual_turnover

        prices = np.array(self.state[1:(self.stock_dim + 1)], dtype=np.float64)
        holdings = np.array(
            self.state[(self.stock_dim + 1):(self.stock_dim * 2 + 1)],
            dtype=np.float64,
        )
        begin_total_asset = float(self.state[0] + np.dot(prices, holdings))
        portfolio_value = max(begin_total_asset, 1e-8)

        tc_value = self.weight_transaction_cost * turnover_target * portfolio_value
        self.cost += tc_value
        self._last_tc = tc_value
        self.state[0] = max(self.state[0] - tc_value, 0.0)

        investable_value = max(portfolio_value - tc_value, 1e-8)
        new_holdings = (executed_weights * investable_value) / (prices + 1e-8)
        self.state[(self.stock_dim + 1):(self.stock_dim * 2 + 1)] = new_holdings.tolist()
        spent = float(np.dot(prices, new_holdings))
        self.state[0] = max(investable_value - spent, 0.0)
        self.actions_memory.append(executed_weights.copy())
        self.trades += int(np.sum(np.abs(delta) > 1e-6))

        # state: s -> s+1
        self.day += 1
        self.data = self.df.loc[self.day, :]
        if self.turbulence_threshold is not None:
            if len(self.df.tic.unique()) == 1:
                self.turbulence = self.data[self.risk_indicator_col]
            elif len(self.df.tic.unique()) > 1:
                self.turbulence = self.data[self.risk_indicator_col].values[0]
        self.state = self._update_state()

        end_total_asset = self.state[0] + sum(
            np.array(self.state[1 : (self.stock_dim + 1)])
            * np.array(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
        )
        self.asset_memory.append(end_total_asset)
        self.date_memory.append(self._get_date())

        log_return = np.log((end_total_asset + 1e-8) / (begin_total_asset + 1e-8))
        if self.reward_type == "dsr_cvar" and self.risk_reward is not None:
            risk_reward, reward_info = self.risk_reward.compute(log_return)
            self.last_dsr = float(reward_info["dsr_bonus"])
            self.last_cvar = float(reward_info["cvar_value"])
            self.reward = risk_reward * self.reward_scaling
            self._last_reward_info = reward_info.copy()
        elif self.reward_type == "adaptive_risk" and self.risk_reward is not None:
            risk_reward, reward_info = self.risk_reward.compute(
                log_return, executed_weights
            )
            self.last_crisis_level = float(reward_info["crisis_level"])
            self.last_kappa = float(
                reward_info.get("kappa_sharpe", reward_info.get("kappa", self.adaptive_lambda_sharpe))
            )
            self.last_kappa_cvar = float(
                reward_info.get("kappa_cvar", self.adaptive_lambda_cvar)
            )
            self.last_delta_sharpe = float(reward_info["delta_sharpe"])
            self.last_cvar = float(reward_info.get("cvar_value", 0.0))
            self._crisis_level = self.last_crisis_level
            self.reward = risk_reward * self.reward_scaling
            self._last_reward_info = reward_info.copy()
        else:
            basic_reward = end_total_asset - begin_total_asset
            self.reward = basic_reward * self.reward_scaling
            self._last_reward_info = {
                "log_return": log_return,
                "delta_sharpe": 0.0,
                "cvar_value": 0.0,
                "cvar_penalty": 0.0,
                "crisis_level": getattr(self, "_crisis_level", 0.5),
                "kappa": getattr(self, "adaptive_lambda_sharpe", 0.0),
                "kappa_sharpe": getattr(self, "adaptive_lambda_sharpe", 0.0),
                "kappa_cvar": getattr(self, "adaptive_lambda_cvar", 0.0),
                "risk_bonus": 0.0,
                "turnover_penalty": 0.0,
                "sharpe_online": 0.0,
                "reward_pre_clip": basic_reward,
                "reward_total": np.clip(basic_reward, -1.0, 1.0),
                "components": {
                    "log_return": log_return,
                    "sharpe_term": 0.0,
                    "cvar_term": 0.0,
                    "turnover": 0.0,
                },
            }

        self.rewards_memory.append(self.reward)
        self.state_memory.append(self.state)

        return self.state, self.reward, self.terminal, False, self._build_step_info(
            turnover_target=turnover_target,
            executed_weights=executed_weights,
            target_weights=target_weights,
        )

    def reset(
        self,
        *,
        seed=None,
        options=None,
    ):
        # initiate state
        self.day = 0
        self.data = self.df.loc[self.day, :]
        self.state = self._initiate_state()

        if self.initial:
            self.asset_memory = [
                self.initial_amount
                + np.sum(
                    np.array(self.num_stock_shares)
                    * np.array(self.state[1 : 1 + self.stock_dim])
                )
            ]
        else:
            previous_total_asset = self.previous_state[0] + sum(
                np.array(self.state[1 : (self.stock_dim + 1)])
                * np.array(
                    self.previous_state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)]
                )
            )
            self.asset_memory = [previous_total_asset]

        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.terminal = False
        # self.iteration=self.iteration
        self.rewards_memory = []
        self.actions_memory = []
        self.date_memory = [self._get_date()]
        self.weights_memory = []
        self.executed_weights_memory = []
        self.turnover_memory = []
        self._last_turnover = 0.0
        self._last_tc = 0.0
        self._last_turnover_executed = 0.0
        self._crisis_level = 0.5
        self._last_bridge_step = -1
        self._crisis_history.clear()
        self._last_reward_info = {}

        # 리스크 민감 보상 모듈을 초기화한다.
        if self.risk_reward is not None:
            self.risk_reward.reset()

            # DSR/CVaR 관련 버퍼 초기화
            if self.reward_type == "dsr_cvar":
                self.last_dsr = 0.0
                self.last_cvar = 0.0
            # adaptive_risk 보상용 상태 변수 초기화
            elif self.reward_type == "adaptive_risk":
                self.last_crisis_level = 0.5
                self.last_kappa = self.adaptive_lambda_sharpe
                self.last_kappa_cvar = self.adaptive_lambda_cvar
                self.last_delta_sharpe = 0.0
                self.last_cvar = 0.0
                self._crisis_level = 0.5

        self.episode += 1

        return self.state, {}

    def render(self, mode="human", close=False):
        return self.state

    def _initiate_state(self):
        if self.initial:
            # For Initial State
            if len(self.df.tic.unique()) > 1:
                # for multiple stock
                state = (
                    [self.initial_amount]
                    + self.data.close.values.tolist()
                    + self.num_stock_shares
                    + sum(
                        (
                            self.data[tech].values.tolist()
                            for tech in self.tech_indicator_list
                        ),
                        [],
                    )
                )  # append initial stocks_share to initial state, instead of all zero
            else:
                # for single stock
                state = (
                    [self.initial_amount]
                    + [self.data.close]
                    + [0] * self.stock_dim
                    + sum(([self.data[tech]] for tech in self.tech_indicator_list), [])
                )
        else:
            # Using Previous State
            if len(self.df.tic.unique()) > 1:
                # for multiple stock
                state = (
                    [self.previous_state[0]]
                    + self.data.close.values.tolist()
                    + self.previous_state[
                        (self.stock_dim + 1) : (self.stock_dim * 2 + 1)
                    ]
                    + sum(
                        (
                            self.data[tech].values.tolist()
                            for tech in self.tech_indicator_list
                        ),
                        [],
                    )
                )
            else:
                # for single stock
                state = (
                    [self.previous_state[0]]
                    + [self.data.close]
                    + self.previous_state[
                        (self.stock_dim + 1) : (self.stock_dim * 2 + 1)
                    ]
                    + sum(([self.data[tech]] for tech in self.tech_indicator_list), [])
                )

        # 상태 벡터에 DSR/CVaR 및 위기 레벨 초기값을 추가한다.
        if self.reward_type == "dsr_cvar":
            state = state + [0.0, 0.0]
        elif self.reward_type == "adaptive_risk":
            state = state + [0.0, 0.0, 0.5]

        return state

    def _update_state(self):
        if len(self.df.tic.unique()) > 1:
            # for multiple stock
            state = (
                [self.state[0]]
                + self.data.close.values.tolist()
                + list(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
                + sum(
                    (
                        self.data[tech].values.tolist()
                        for tech in self.tech_indicator_list
                    ),
                    [],
                )
            )

        else:
            # for single stock
            state = (
                [self.state[0]]
                + [self.data.close]
                + list(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
                + sum(([self.data[tech]] for tech in self.tech_indicator_list), [])
            )

        # 이전 스텝에서 계산한 DSR/CVaR 및 위기 레벨을 상태 벡터에 포함한다.
        if self.reward_type == "dsr_cvar":
            state = state + [self.last_dsr, self.last_cvar]
        elif self.reward_type == "adaptive_risk":
            state = state + [
                self.last_delta_sharpe,
                self.last_cvar,
                self._crisis_level,
            ]

        return state

    def _get_date(self):
        if len(self.df.tic.unique()) > 1:
            date = self.data.date.unique()[0]
        else:
            date = self.data.date
        return date

    # add save_state_memory to preserve state in the trading process
    def save_state_memory(self):
        if len(self.df.tic.unique()) > 1:
            # date and close price length must match actions length
            date_list = self.date_memory[:-1]
            df_date = pd.DataFrame(date_list)
            df_date.columns = ["date"]

            state_list = self.state_memory
            df_states = pd.DataFrame(
                state_list,
                columns=[
                    "cash",
                    "Bitcoin_price",
                    "Gold_price",
                    "Bitcoin_num",
                    "Gold_num",
                    "Bitcoin_Disable",
                    "Gold_Disable",
                ],
            )
            df_states.index = df_date.date
            # df_actions = pd.DataFrame({'date':date_list,'actions':action_list})
        else:
            date_list = self.date_memory[:-1]
            state_list = self.state_memory
            df_states = pd.DataFrame({"date": date_list, "states": state_list})
        # print(df_states)
        return df_states

    def save_asset_memory(self):
        date_list = self.date_memory
        asset_list = self.asset_memory
        # print(len(date_list))
        # print(len(asset_list))
        df_account_value = pd.DataFrame(
            {"date": date_list, "account_value": asset_list}
        )
        return df_account_value

    def save_action_memory(self):
        if len(self.df.tic.unique()) > 1:
            # date and close price length must match actions length
            date_list = self.date_memory[:-1]
            df_date = pd.DataFrame(date_list)
            df_date.columns = ["date"]

            action_list = self.actions_memory
            df_actions = pd.DataFrame(action_list)
            df_actions.columns = self.data.tic.values
            df_actions.index = df_date.date
            # df_actions = pd.DataFrame({'date':date_list,'actions':action_list})
        else:
            date_list = self.date_memory[:-1]
            action_list = self.actions_memory
            df_actions = pd.DataFrame({"date": date_list, "actions": action_list})
        return df_actions

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs
