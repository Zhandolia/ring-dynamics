import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class BayesianScorer:
    """
    Bayesian state-space model for round scoring
    
    Estimates round score probabilities based on fight events
    """
    
    def __init__(self):
        """Initialize Bayesian scorer with priors"""
        
        # Judge criteria weights (based on boxing rules)
        self.criteria_weights = {
            'effective_aggression': 0.40,
            'ring_generalship': 0.30,
            'defense': 0.20,
            'clean_punching': 0.10
        }
        
        # Prior probabilities for score outcomes
        self.priors = {
            '10-9': 0.85,  # Most common
            '10-8': 0.10,   # Dominant rounds
            '10-10': 0.05   # Even rounds
        }
    
    def compute_round_score(self, round_events: Dict) -> Dict:
        """
        Compute round score probabilities
        
        Args:
            round_events: Dict containing:
                - punches: List of punch events
                - metrics: List of fight metrics over the round
                - fighter_ids: [0, 1]
                
        Returns:
            Dict with score probabilities for each fighter
        """
        
        punches = round_events.get('punches', [])
        metrics = round_events.get('metrics', [])
        
        # Aggregate statistics for each fighter
        stats = self._aggregate_statistics(punches, metrics)
        
        # Compute criterion scores
        criteria_scores = self._compute_criteria_scores(stats)
        
        # Compute overall scores using weighted combination
        overall_scores = self._compute_overall_scores(criteria_scores)
        
        # Determine winner and score probabilities
        score_probs = self._compute_score_probabilities(overall_scores)
        
        return score_probs
    
    def _aggregate_statistics(self, punches: List[Dict],
                              metrics: List[Dict]) -> Dict:
        """Aggregate fight statistics for each fighter"""
        
        stats = {
            0: self._init_fighter_stats(),
            1: self._init_fighter_stats()
        }
        
        # Process punches
        for punch in punches:
            fighter_id = punch['fighter_id']
            
            stats[fighter_id]['punches_thrown'] += 1
            
            if punch['outcome'] == 'landed':
                stats[fighter_id]['punches_landed'] += 1
                stats[fighter_id]['impact_total'] += punch.get('impact_score', 0)
                
                if punch['punch_type'] in ['cross', 'hook', 'uppercut']:
                    stats[fighter_id]['power_punches_landed'] += 1
            
            elif punch['outcome'] == 'blocked':
                # Credit opponent for defense
                opponent_id = 1 - fighter_id
                stats[opponent_id]['punches_blocked'] += 1
        
        # Process metrics for ring control and aggression
        for frame_metrics in metrics:
            for fighter_id, m in frame_metrics.items():
                if m.get('distance') == 'inside':
                    stats[fighter_id]['aggression_frames'] += 1
                
                # Ring control (center position)
                pos = m.get('ring_position', (0.5, 0.5))
                if 0.4 <= pos[0] <= 0.6 and 0.4 <= pos[1] <= 0.6:
                    stats[fighter_id]['center_ring_frames'] += 1
        
        return stats
    
    def _init_fighter_stats(self) -> Dict:
        """Initialize empty statistics dict"""
        return {
            'punches_thrown': 0,
            'punches_landed': 0,
            'power_punches_landed': 0,
            'punches_blocked': 0,
            'impact_total': 0.0,
            'aggression_frames': 0,
            'center_ring_frames': 0
        }
    
    def _compute_criteria_scores(self, stats: Dict) -> Dict:
        """
        Compute scores for each judging criterion
        
        Returns dict mapping criterion -> [score_f0, score_f1]
        """
        
        scores = {}
        
        # Clean punching (accuracy and power)
        scores['clean_punching'] = self._score_clean_punching(stats)
        
        # Effective aggression
        scores['effective_aggression'] = self._score_effective_aggression(stats)
        
        # Ring generalship
        scores['ring_generalship'] = self._score_ring_generalship(stats)
        
        # Defense
        scores['defense'] = self._score_defense(stats)
        
        return scores
    
    def _score_clean_punching(self, stats: Dict) -> Tuple[float, float]:
        """Score clean punching criterion"""
        
        f0_accuracy = (stats[0]['punches_landed'] / max(stats[0]['punches_thrown'], 1))
        f1_accuracy = (stats[1]['punches_landed'] / max(stats[1]['punches_thrown'], 1))
        
        # Combine accuracy with total impact
        f0_score = f0_accuracy * 0.5 + (stats[0]['impact_total'] / 10.0) * 0.5
        f1_score = f1_accuracy * 0.5 + (stats[1]['impact_total'] / 10.0) * 0.5
        
        # Normalize to 0-1
        total = f0_score + f1_score
        if total > 0:
            f0_score /= total
            f1_score /= total
        else:
            f0_score, f1_score = 0.5, 0.5
        
        return (f0_score, f1_score)
    
    def _score_effective_aggression(self, stats: Dict) -> Tuple[float, float]:
        """Score effective aggression"""
        
        # Aggression with landed punches
        f0_agg = stats[0]['punches_landed'] + stats[0]['aggression_frames'] / 100.0
        f1_agg = stats[1]['punches_landed'] + stats[1]['aggression_frames'] / 100.0
        
        total = f0_agg + f1_agg
        if total > 0:
            return (f0_agg / total, f1_agg / total)
        
        return (0.5, 0.5)
    
    def _score_ring_generalship(self, stats: Dict) -> Tuple[float, float]:
        """Score ring generalship (ring control)"""
        
        f0_control = stats[0]['center_ring_frames']
        f1_control = stats[1]['center_ring_frames']
        
        total = f0_control + f1_control
        if total > 0:
            return (f0_control / total, f1_control / total)
        
        return (0.5, 0.5)
    
    def _score_defense(self, stats: Dict) -> Tuple[float, float]:
        """Score defensive ability"""
        
        # Lower score is better (fewer punches landed on you)
        f0_defense = 1.0 - (stats[1]['punches_landed'] / max(stats[1]['punches_thrown'], 1))
        f1_defense = 1.0 - (stats[0]['punches_landed'] / max(stats[0]['punches_thrown'], 1))
        
        # Add blocking credit
        f0_defense += stats[0]['punches_blocked'] / 20.0
        f1_defense += stats[1]['punches_blocked'] / 20.0
        
        # Normalize
        total = f0_defense + f1_defense
        if total > 0:
            return (f0_defense / total, f1_defense / total)
        
        return (0.5, 0.5)
    
    def _compute_overall_scores(self, criteria_scores: Dict) -> Tuple[float, float]:
        """Compute weighted overall scores"""
        
        f0_total = 0.0
        f1_total = 0.0
        
        for criterion, weight in self.criteria_weights.items():
            f0_score, f1_score = criteria_scores[criterion]
            f0_total += f0_score * weight
            f1_total += f1_score * weight
        
        return (f0_total, f1_total)
    
    def _compute_score_probabilities(self, overall_scores: Tuple[float, float]) -> Dict:
        """
        Convert overall scores to round score probabilities
        
        Returns dict with score probabilities
        """
        
        f0_score, f1_score = overall_scores
        
        # Compute score margin
        margin = abs(f0_score - f1_score)
        
        # Determine winner
        if f0_score > f1_score:
            winner = 0
            winner_score = f0_score
        else:
            winner = 1
            winner_score = f1_score
        
        # Determine score type probabilities
        # Larger margin -> higher chance of 10-9 or 10-8
        
        if margin < 0.1:
            # Very close -> likely 10-10 or narrow 10-9
            prob_10_10 = 0.3
            prob_10_9 = 0.7
            prob_10_8 = 0.0
        elif margin < 0.3:
            # Clear winner -> 10-9
            prob_10_10 = 0.05
            prob_10_9 = 0.90
            prob_10_8 = 0.05
        else:
            # Dominant -> potential 10-8
            prob_10_10 = 0.0
            prob_10_9 = 0.75
            prob_10_8 = 0.25
        
        # Build result
        if winner == 0:
            result = {
                'round_winner': 0,
                'fighter_0_score': 10,
                'fighter_1_score': 9,  # Most likely
                'prob_10_9_f0': prob_10_9,
                'prob_10_9_f1': 0.0,
                'prob_10_8_f0': prob_10_8,
                'prob_10_8_f1': 0.0,
                'prob_10_10': prob_10_10
            }
        else:
            result = {
                'round_winner': 1,
                'fighter_0_score': 9,
                'fighter_1_score': 10,
                'prob_10_9_f0': 0.0,
                'prob_10_9_f1': prob_10_9,
                'prob_10_8_f0': 0.0,
                'prob_10_8_f1': prob_10_8,
                'prob_10_10': prob_10_10
            }
        
        return result
