import random
import math
from abc import ABC, abstractmethod
import pulp

role_requirements = {
    'forward': 3,
    'midfielder': 4,
    'defender': 3,
    'goalkeeper': 1
}

class BaseBiddingStrategy:
    def __init__(self, players):
        self.players = players
        self.acquired_players = []

    def can_acquire(self, player_index):
        player_role = self.players[player_index]['role']
        current_role_count = sum(1 for p in self.acquired_players if self.players[p]['role'] == player_role)
        return current_role_count < role_requirements[player_role]

class EpsilonGreedy(BaseBiddingStrategy):
    def __init__(self, players, epsilon=0.1, exploitation_factor=0.8):
        super().__init__(players)
        self.epsilon = epsilon
        self.exploitation_factor = exploitation_factor

    def compute_bid(self, player_index):
        if not self.can_acquire(player_index):
            return 0
        player_value = self.players[player_index]['value']
        if random.random() < self.epsilon:
            return random.uniform(0, player_value)
        return player_value * self.exploitation_factor

class Reactive(BaseBiddingStrategy):
    def __init__(self, players, initial_bid_factor=1.0):
        super().__init__(players)
        self.bid_history = []
        self.initial_bid_factor = initial_bid_factor

    def compute_bid(self, player_index):
        if not self.can_acquire(player_index):
            return 0
        player_value = self.players[player_index]['value']
        if not self.bid_history:
            return player_value * self.initial_bid_factor
        avg_bid = sum(self.bid_history) / len(self.bid_history)
        bid_factor = avg_bid / player_value + 0.05
        return player_value * bid_factor

class ValueBased(BaseBiddingStrategy):
    def compute_bid(self, player_index):
        if not self.can_acquire(player_index):
            return 0
        player_value = self.players[player_index]['value']
        return player_value * 0.8

class OptimalTeamCompositionLP(BaseBiddingStrategy):
    def compute_bid(self, player_index):
        if not self.can_acquire(player_index):
            return 0
        player_value = self.players[player_index]['value']
        return 0.9 * player_value


# UCB1 with different bidding strategies as arms:
class UCB1:
    def __init__(self, strategies):
        self.strategies = strategies
        self.counts = [0] * len(strategies)
        self.values = [0] * len(strategies)

    def select_arm(self):
        n_arms = len(self.strategies)
        
        for i in range(n_arms):
            if self.counts[i] == 0:
                return i

        total_counts = sum(self.counts)
        ucb_values = [
            self.values[i] + math.sqrt((2 * math.log(total_counts)) / self.counts[i])
            for i in range(n_arms)
        ]
        return ucb_values.index(max(ucb_values))

    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        self.values[chosen_arm] = ((n - 1) / n) * value + (1 / n) * reward


def generate_historical_data(n=100):
    roles = ['forward', 'midfielder', 'defender', 'goalkeeper']
    players = []

    for _ in range(n):
        value = random.randint(80, 150)
        cost = int(value * random.uniform(0.6, 0.9))
        role = random.choice(roles)
        
        player = {
            'value': value,
            'cost': cost,
            'role': role
        }
        players.append(player)

    return players


# Simulation using synthetic data based on historical data:
def warm_start_simulation(ucb1_model, historical_data, n_rounds=None):
    if n_rounds is None:
        n_rounds = len(historical_data)

    for i in range(n_rounds):
        player_index = i  # Assuming players are bid on in order

        # Simulate a competitive bid from other bidders
        player_value = historical_data[player_index]['value']
        competitive_bid = player_value * random.uniform(0.7, 1.2)  # Between 70% to 120% of player value

        chosen_strategy_idx = ucb1_model.select_arm()
        chosen_strategy = ucb1_model.strategies[chosen_strategy_idx]
        
        bid = chosen_strategy.compute_bid(player_index)
        
        # Simulating the reward: winning the player if our bid exceeds the competitive bid.
        reward = 1 if bid > competitive_bid else 0
        ucb1_model.update(chosen_strategy_idx, reward)

        if reward == 1:
            chosen_strategy.acquired_players.append(player_index)
            chosen_strategy.remaining_budget -= historical_data[player_index]['cost']


# Generate historical data for 100 players
historical_data = generate_historical_data()
print(historical_data[:10])  # Displaying the first 10 players as an example

# Initial Budget
initial_budget = 1000

# Initialize strategies with the historical_data and initial_budget
strategies = [
    EpsilonGreedy(historical_data, initial_budget),
    Reactive(historical_data, initial_budget),
    ValueBased(historical_data, initial_budget),
    OptimalTeamCompositionLP(historical_data, initial_budget)
]

# Printing the strategies to verify initialization
for strategy in strategies:
    print(type(strategy).__name__, "-> Remaining Budget:", strategy.remaining_budget)


# Initialize UCB1 model with the strategies
ucb1_model = UCB1(strategies)
warm_start_simulation(ucb1_model, historical_data, n_rounds=1000)