from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class JudgeScorer:
    """
    Judge-style scoring with punch impact weighting
    
    Emulates professional boxing judge scoring methodology
    """
    
    def __init__(self):
        """Initialize judge scorer"""
        
        # Base weights for punch types
        self.punch_weights = {
            'jab': 1.0,
            'cross': 2.5,
            'hook': 2.8,
            'uppercut': 3.0
        }
        
        # Target multipliers
        self.target_multipliers = {
            'head': 1.2,
            'body': 1.0
        }
        
        # Outcome multipliers
        self.outcome_multipliers = {
            'landed': 1.0,
            'blocked': 0.3,
            'missed': 0.0
        }
    
    def compute_punch_impact(self, punch: Dict, context: Dict = None) -> float:
        """
        Compute weighted impact score for a punch
        
        Args:
            punch: Punch event dict
            context: Optional context (for counter-punch detection, etc.)
            
        Returns:
            Impact score (0-10 scale)
        """
        
        punch_type = punch.get('punch_type', 'jab')
        target = punch.get('target', 'head')
        outcome = punch.get('outcome', 'missed')
        impact_score = punch.get('impact_score', 0.5)
        
        # Base weight
        base_weight = self.punch_weights.get(punch_type, 1.0)
        
        # Apply multipliers
        target_mult = self.target_multipliers.get(target, 1.0)
        outcome_mult = self.outcome_multipliers.get(outcome, 0.0)
        
        # Counter-punch bonus
        counter_bonus = 1.3 if self._is_counter_punch(punch, context) else 1.0
        
        # Compute final impact
        impact = (base_weight *
                 target_mult *
                 outcome_mult *
                 counter_bonus *
                 impact_score)
        
        # Scale to 0-10
        impact = min(impact, 10.0)
        
        return impact
    
    def _is_counter_punch(self, punch: Dict, context: Dict = None) -> bool:
        """
        Determine if punch is a counter-punch
        
        Counter-punches are scored higher by judges
        """
        
        if context is None:
            return False
        
        # Check if opponent threw punch in last 0.5 seconds
        opponent_recent_punches = context.get('opponent_recent_punches', [])
        
        punch_time = punch.get('timestamp', 0)
        
        for opp_punch in opponent_recent_punches:
            opp_time = opp_punch.get('timestamp', 0)
            
            # If opponent punched within 0.5s before this punch
            if 0 < (punch_time - opp_time) < 0.5:
                return True
        
        return False
    
    def score_effective_aggression(self, punches: List[Dict],
                                   metrics: List[Dict],
                                   fighter_id: int) -> float:
        """
        Score effective aggression
        
        Forward movement WITH landed punches
        """
        
        # Count landed punches while moving forward
        effective_punches = 0
        
        for punch in punches:
            if punch['fighter_id'] != fighter_id:
                continue
            
            if punch['outcome'] == 'landed':
                effective_punches += 1
        
        # Factor in ring control (moving forward)
        forward_frames = 0
        
        for frame_metrics in metrics:
            if fighter_id not in frame_metrics:
                continue
            
            m = frame_metrics[fighter_id]
            
            # Simplified: if in "inside" distance, assume aggression
            if m.get('distance') == 'inside':
                forward_frames += 1
        
        # Combine
        aggression_score = (effective_punches * 2.0 +
                           forward_frames / 10.0)
        
        return min(aggression_score, 100.0)
    
    def score_ring_generalship(self, metrics: List[Dict],
                               fighter_id: int) -> float:
        """
        Score ring generalship (control of the ring)
        
        Based on center control and making opponent fight on back foot
        """
        
        center_frames = 0
        total_frames = len(metrics)
        
        for frame_metrics in metrics:
            if fighter_id not in frame_metrics:
                continue
            
            m = frame_metrics[fighter_id]
            pos = m.get('ring_position', (0.5, 0.5))
            
            # Center is (0.4-0.6, 0.4-0.6)
            if 0.4 <= pos[0] <= 0.6 and 0.4 <= pos[1] <= 0.6:
                center_frames += 1
        
        if total_frames == 0:
            return 0.0
        
        control_pct = center_frames / total_frames
        
        return control_pct * 100.0
    
    def score_defense(self, punches: List[Dict], fighter_id: int) -> float:
        """
        Score defensive ability
        
        Based on punches blocked/avoided
        """
        
        opponent_id = 1 - fighter_id
        
        # Count opponent's punches
        opponent_punches = [p for p in punches if p['fighter_id'] == opponent_id]
        
        if len(opponent_punches) == 0:
            return 100.0  # Perfect defense (no punches faced)
        
        # Count how many were blocked or missed
        defended = sum(1 for p in opponent_punches
                      if p['outcome'] in ['blocked', 'missed'])
        
        defense_rate = defended / len(opponent_punches)
        
        return defense_rate * 100.0
    
    def generate_scorecard(self, rounds: List[Dict]) -> Dict:
        """
        Generate final scorecard
        
        Args:
            rounds: List of round score dicts
            
        Returns:
            Complete scorecard with totals
        """
        
        total_0 = sum(r.get('fighter_0_score', 10) for r in rounds)
        total_1 = sum(r.get('fighter_1_score', 10) for r in rounds)
        
        # Determine winner
        if total_0 > total_1:
            winner = 0
            decision = f"Fighter 0 wins {total_0}-{total_1}"
        elif total_1 > total_0:
            winner = 1
            decision = f"Fighter 1 wins {total_1}-{total_0}"
        else:
            winner = -1
            decision = f"Draw {total_0}-{total_1}"
        
        scorecard = {
            'rounds': rounds,
            'total_fighter_0': total_0,
            'total_fighter_1': total_1,
            'winner': winner,
            'decision': decision,
            'total_rounds': len(rounds)
        }
        
        return scorecard
