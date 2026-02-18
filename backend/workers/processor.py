"""
Worker task for processing fight videos

This module handles the end-to-end video processing workflow
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

# Try to import CV pipeline (may not be available if dependencies not installed)
try:
    from workers.cv_pipeline.pipeline import CVPipeline
    CV_AVAILABLE = True
except ImportError as e:
    logger.warning(f"CV pipeline not available: {e}")
    CVPipeline = None
    CV_AVAILABLE = False

from workers.scoring.bayesian_model import BayesianScorer
from workers.scoring.monte_carlo import MonteCarloSimulator
from workers.scoring.judge_scoring import JudgeScorer

logger = logging.getLogger(__name__)


class FightProcessor:
    """Process complete fight videos through CV and scoring pipelines"""
    
    def __init__(self, device: str = 'cpu'):
        """
        Initialize fight processor
        
        Args:
            device: Device for CV processing ('cpu' or 'cuda:0')
        """
        self.device = device
        self.cv_pipeline = None
        self.bayesian_scorer = BayesianScorer()
        self.monte_carlo = MonteCarloSimulator(self.bayesian_scorer)
        self.judge_scorer = JudgeScorer()
        
    def process_fight(self, fight_id: str, video_path: str, 
                     total_rounds: int = 12, round_duration: int = 180,
                     websocket_callback=None) -> Dict[str, Any]:
        """
        Process a complete fight video
        
        Args:
            fight_id: Unique fight identifier
            video_path: Path to video file
            total_rounds: Total number of rounds in fight
            round_duration: Duration of each round in seconds
            websocket_callback: Optional callback for real-time updates
            
        Returns:
            Complete fight analysis results
        """
        logger.info(f"Starting fight processing: {fight_id}")
        
        # Check if CV pipeline is available
        if not CV_AVAILABLE:
            logger.warning(f"CV pipeline not available. Using mock mode.")
            return self._mock_process_fight(fight_id, video_path, total_rounds)
        
        # Initialize CV pipeline
        try:
            self.cv_pipeline = CVPipeline(device=self.device)
        except Exception as e:
            logger.warning(f"CV pipeline initialization failed: {e}. Using mock mode.")
            return self._mock_process_fight(fight_id, video_path, total_rounds)
        
        # Process video through CV pipeline
        logger.info("Running CV pipeline...")
        frame_results = self.cv_pipeline.process_video(
            video_path,
            fps=30,
            max_frames=3000  # Process first 100 seconds for demo
        )
        
        # Aggregate events by round
        logger.info("Aggregating events by round...")
        round_data = self._aggregate_by_round(frame_results, round_duration, fps=30)
        
        # Score each round
        logger.info("Scoring rounds...")
        round_scores = []
        for round_num, data in enumerate(round_data, start=1):
            score_probs = self.bayesian_scorer.score_round(data)
            round_scores.append({
                'round_number': round_num,
                'probabilities': score_probs,
                'events': data
            })
            
            # Send WebSocket update
            if websocket_callback:
                websocket_callback({
                    'type': 'round_scored',
                    'round': round_num,
                    'scores': score_probs
                })
        
        # Calculate win probabilities
        logger.info("Running Monte Carlo simulation...")
        win_probs = self.monte_carlo.simulate_fight(round_scores, total_rounds)
        
        # Generate final scorecard
        logger.info("Generating judge scorecard...")
        scorecard = self.judge_scorer.generate_scorecard(round_data)
        
        results = {
            'fight_id': fight_id,
            'status': 'completed',
            'processed_at': datetime.now().isoformat(),
            'round_scores': round_scores,
            'win_probabilities': win_probs,
            'scorecard': scorecard,
            'total_frames': len(frame_results),
            'total_punches': sum(len(r.get('punches', [])) for r in frame_results)
        }
        
        logger.info(f"Fight processing complete: {fight_id}")
        return results
    
    def _aggregate_by_round(self, frame_results, round_duration, fps=30):
        """Aggregate frame-level results into round-level data"""
        frames_per_round = round_duration * fps
        rounds = []
        
        current_round = []
        for i, frame_data in enumerate(frame_results):
            current_round.append(frame_data)
            
            if (i + 1) % frames_per_round == 0:
                rounds.append(self._summarize_round(current_round))
                current_round = []
        
        # Add final partial round if exists
        if current_round:
            rounds.append(self._summarize_round(current_round))
        
        return rounds
    
    def _summarize_round(self, frame_data_list):
        """Summarize a round from frame data"""
        summary = {
            'punches_landed_f0': 0,
            'punches_landed_f1': 0,
            'punches_thrown_f0': 0,
            'punches_thrown_f1': 0,
            'total_impact_f0': 0.0,
            'total_impact_f1': 0.0,
            'aggression_frames_f0': 0,
            'aggression_frames_f1': 0,
            'center_frames_f0': 0,
            'center_frames_f1': 0,
            'blocks_f0': 0,
            'blocks_f1': 0
        }
        
        for frame in frame_data_list:
            # Aggregate punch stats
            for punch in frame.get('punches', []):
                fighter_id = punch.get('fighter_id', 0)
                suffix = f'_f{fighter_id}'
                
                summary[f'punches_thrown{suffix}'] += 1
                if punch.get('outcome') == 'landed':
                    summary[f'punches_landed{suffix}'] += 1
                    summary[f'total_impact{suffix}'] += punch.get('impact_score', 0)
                elif punch.get('outcome') == 'blocked':
                    opp_suffix = f'_f{1-fighter_id}'
                    summary[f'blocks{opp_suffix}'] += 1
            
            # Aggregate metrics
            metrics = frame.get('metrics', {})
            for fighter_id in [0, 1]:
                suffix = f'_f{fighter_id}'
                fighter_metrics = metrics.get(fighter_id, {})
                
                if fighter_metrics.get('distance') == 'inside':
                    summary[f'aggression_frames{suffix}'] += 1
                    
                if fighter_metrics.get('ring_position', {}).get('center_control'):
                    summary[f'center_frames{suffix}'] += 1
        
        return summary
    
    def _mock_process_fight(self, fight_id: str, video_path: str, total_rounds: int):
        """Mock processing for when CV dependencies not available"""
        logger.info(f"Running MOCK processing for {fight_id}")
        
        import random
        random.seed(42)
        
        round_scores = []
        for round_num in range(1, min(4, total_rounds + 1)):  # Mock 3 rounds
            round_scores.append({
                'round_number': round_num,
                'probabilities': {
                    'prob_10_9_f0': random.uniform(0.3, 0.6),
                    'prob_10_9_f1': random.uniform(0.3, 0.6),
                    'prob_10_8_f0': random.uniform(0.0, 0.05),
                    'prob_10_8_f1': random.uniform(0.0, 0.05),
                    'prob_draw': random.uniform(0.05, 0.15)
                }
            })
        
        return {
            'fight_id': fight_id,
            'status': 'completed_mock',
            'processed_at': datetime.now().isoformat(),
            'round_scores': round_scores,
            'win_probabilities': {
                'fighter_0': random.uniform(0.4, 0.6),
                'fighter_1': random.uniform(0.4, 0.6),
                'draw': random.uniform(0.0, 0.1)
            },
            'message': 'Mock processing - CV dependencies not available'
        }
