import numpy as np
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class MonteCarloSimulator:
    """
    Monte Carlo simulation for win probability estimation
    
    Simulates remaining rounds to estimate fight outcome probabilities
    """
    
    def __init__(self, n_simulations: int = 10000):
        """
        Initialize simulator
        
        Args:
            n_simulations: Number of Monte Carlo simulations to run
        """
        self.n_simulations = n_simulations
    
    def simulate_win_probability(self, current_scores: Dict,
                                 rounds_completed: int,
                                 total_rounds: int,
                                 round_score_dist: Dict) -> Dict:
        """
        Simulate win probabilities using Monte Carlo
        
        Args:
            current_scores: Current total scores {0: score, 1: score}
            rounds_completed: Number of completed rounds
            total_rounds: Total rounds in fight
            round_score_dist: Round score distribution from Bayesian model
            
        Returns:
            Dict with win probabilities and confidence intervals
        """
        
        rounds_remaining = total_rounds - rounds_completed
        
        if rounds_remaining == 0:
            # Fight is over, determine winner
            return self._final_result(current_scores)
        
        # Run simulations
        outcomes = []
        
        for _ in range(self.n_simulations):
            outcome = self._simulate_single_fight(
                current_scores.copy(),
                rounds_remaining,
                round_score_dist
            )
            outcomes.append(outcome)
        
        # Compute probabilities
        results = self._compute_probabilities(outcomes)
        
        return results
    
    def _simulate_single_fight(self, scores: Dict, rounds_left: int,
                               score_dist: Dict) -> int:
        """
        Simulate a single fight outcome
        
        Returns:
            Winner ID (0, 1, or -1 for draw)
        """
        
        for _ in range(rounds_left):
            # Sample round outcome
            round_outcome = self._sample_round_outcome(score_dist)
            
            scores[0] += round_outcome[0]
            scores[1] += round_outcome[1]
        
        # Determine winner
        if scores[0] > scores[1]:
            return 0
        elif scores[1] > scores[0]:
            return 1
        else:
            return -1  # Draw
    
    def _sample_round_outcome(self, score_dist: Dict) -> Tuple[int, int]:
        """
        Sample a round outcome from score distribution
        
        Returns:
            (fighter_0_score, fighter_1_score) for the round
        """
        
        # Get probabilities
        prob_10_9_f0 = score_dist.get('prob_10_9_f0', 0.0)
        prob_10_9_f1 = score_dist.get('prob_10_9_f1', 0.0)
        prob_10_8_f0 = score_dist.get('prob_10_8_f0', 0.0)
        prob_10_8_f1 = score_dist.get('prob_10_8_f1', 0.0)
        prob_10_10 = score_dist.get('prob_10_10', 0.0)
        
        # Normalize probabilities
        total_prob = (prob_10_9_f0 + prob_10_9_f1 + prob_10_8_f0 +
                     prob_10_8_f1 + prob_10_10)
        
        if total_prob == 0:
            # Default to even split
            return (10, 10)
        
        probs = np.array([
            prob_10_9_f0,
            prob_10_9_f1,
            prob_10_8_f0,
            prob_10_8_f1,
            prob_10_10
        ]) / total_prob
        
        # Sample outcome
        outcome_idx = np.random.choice(5, p=probs)
        
        outcomes = [
            (10, 9),   # 10-9 fighter 0
            (9, 10),   # 10-9 fighter 1
            (10, 8),   # 10-8 fighter 0
            (8, 10),   # 10-8 fighter 1
            (10, 10)   # Draw round
        ]
        
        return outcomes[outcome_idx]
    
    def _compute_probabilities(self, outcomes: List[int]) -> Dict:
        """Compute win probabilities from simulation outcomes"""
        
        fighter_0_wins = sum(1 for x in outcomes if x == 0)
        fighter_1_wins = sum(1 for x in outcomes if x == 1)
        draws = sum(1 for x in outcomes if x == -1)
        
        total = len(outcomes)
        
        return {
            'win_prob_f0': fighter_0_wins / total,
            'win_prob_f1': fighter_1_wins / total,
            'draw_prob': draws / total,
            'confidence_interval_95': self._compute_confidence_interval(outcomes)
        }
    
    def _compute_confidence_interval(self, outcomes: List[int]) -> Dict:
        """Compute 95% confidence intervals"""
        
        # Bootstrap confidence intervals
        n = len(outcomes)
        
        # For fighter 0 win probability
        f0_wins = [1 if x == 0 else 0 for x in outcomes]
        f0_mean = np.mean(f0_wins)
        f0_std = np.std(f0_wins)
        
        # 95% CI using normal approximation
        margin = 1.96 * f0_std / np.sqrt(n)
        
        return {
            'f0_lower': max(0, f0_mean - margin),
            'f0_upper': min(1, f0_mean + margin)
        }
    
    def _final_result(self, scores: Dict) -> Dict:
        """Determine result when fight is complete"""
        
        if scores[0] > scores[1]:
            return {
                'win_prob_f0': 1.0,
                'win_prob_f1': 0.0,
                'draw_prob': 0.0,
                'winner': 0
            }
        elif scores[1] > scores[0]:
            return {
                'win_prob_f0': 0.0,
                'win_prob_f1': 1.0,
                'draw_prob': 0.0,
                'winner': 1
            }
        else:
            return {
                'win_prob_f0': 0.0,
                'win_prob_f1': 0.0,
                'draw_prob': 1.0,
                'winner': -1
            }


def update_live_probabilities(completed_rounds: List[Dict],
                              rounds_remaining: int) -> Dict:
    """
    Update win probabilities based on completed rounds
    
    Args:
        completed_rounds: List of round score dicts
        rounds_remaining: Number of rounds left
        
    Returns:
        Updated win probabilities
    """
    
    # Compute current total scores
    current_scores = {0: 0, 1: 0}
    
    for round_data in completed_rounds:
        current_scores[0] += round_data.get('fighter_0_score', 10)
        current_scores[1] += round_data.get('fighter_1_score', 10)
    
    # Use last round's score distribution for future rounds
    # (simplified - could use more sophisticated prediction)
    if completed_rounds:
        last_round_dist = completed_rounds[-1]
    else:
        # Default even distribution
        last_round_dist = {
            'prob_10_9_f0': 0.425,
            'prob_10_9_f1': 0.425,
            'prob_10_8_f0': 0.05,
            'prob_10_8_f1': 0.05,
            'prob_10_10': 0.05
        }
    
    # Run Monte Carlo simulation
    simulator = MonteCarloSimulator(n_simulations=10000)
    
    total_rounds = len(completed_rounds) + rounds_remaining
    
    probabilities = simulator.simulate_win_probability(
        current_scores,
        len(completed_rounds),
        total_rounds,
        last_round_dist
    )
    
    return probabilities
