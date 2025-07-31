from dataclasses import dataclass, field

@dataclass
class MCTSConfig:
    # raaaaaaa
    num_simulations: int = 400
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25

@dataclass
class SelfPlayConfig:
    num_games: int = 24
    num_simulations: int = 100
    temp_threshold: int = 15
    depth_limit: int = 50


@dataclass
class Config:
    debug: bool = False
    mcts: MCTSConfig = field(default_factory=MCTSConfig)
    self_play: SelfPlayConfig = field(default_factory=SelfPlayConfig)

    # haha so cool
    def __post_init__(self):
        if self.debug:
            self.mcts.num_simulations = 50
            self.self_play.num_games = 12
            self.self_play.num_simulations = 50
            self.self_play.temp_threshold = 15
            self.self_play.depth_limit = 10

