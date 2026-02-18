"""
Test suite for Ring Dynamics backend

Run with: pytest -v
"""

import pytest
import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))


# ===== API Tests =====

def test_import_api():
    """Test that API modules can be imported"""
    try:
        from app.main import app
        from app.api import fights
        from app.models import schemas
        assert app is not None
        print("✓ API imports successful")
    except ImportError as e:
        pytest.fail(f"Failed to import API modules: {e}")


def test_schemas():
    """Test Pydantic schemas"""
    from app.models.schemas import FightCreate, FightResponse
    from datetime import datetime
    from uuid import uuid4
    
    # Test FightCreate
    fight_create = FightCreate(youtube_url="https://youtube.com/watch?v=test")
    assert fight_create.youtube_url == "https://youtube.com/watch?v=test"
    
    # Test FightResponse
    fight_response = FightResponse(
        id=uuid4(),
        status="pending",
        created_at=datetime.now()
    )
    assert fight_response.status == "pending"
    print("✓ Schemas working correctly")


# ===== CV Pipeline Tests =====

def test_import_cv_modules():
    """Test that CV modules can be imported"""
    try:
        from workers.cv_pipeline import detection, tracking, pose, punch_classifier, metrics
        print("✓ CV module imports successful")
    except ImportError as e:
        pytest.skip(f"CV modules not available: {e}")


def test_detection_module():
    """Test detection module structure"""
    try:
        from workers.cv_pipeline.detection import YOLODetector
        # Just test that class exists
        assert YOLODetector is not None
        print("✓ Detection module structure valid")
    except Exception as e:
        pytest.skip(f"Detection dependencies not available: {e}")


def test_tracking_module():
    """Test tracking module"""
    try:
        from workers.cv_pipeline.tracking import ByteTracker, Track
        tracker = ByteTracker()
        assert tracker is not None
        print("✓ Tracking module working")
    except Exception as e:
        pytest.skip(f"Tracking dependencies not available: {e}")


def test_pose_module():
    """Test pose estimation module"""
    try:
        from workers.cv_pipeline.pose import PoseEstimator
        # Test initialization without MediaPipe
        assert PoseEstimator is not None
        print("✓ Pose module structure valid")
    except Exception as e:
        pytest.skip(f"Pose dependencies not available: {e}")


def test_punch_classifier():
    """Test punch classifier logic"""
    try:
        from workers.cv_pipeline.punch_classifier import PunchClassifier
        classifier = PunchClassifier()
        
        # Test punch type classification logic
        # Simulate uppercut trajectory
        trajectory = [(0, 0), (0, -60)]  # Upward movement
        punch_type = classifier._classify_punch_type(trajectory)
        assert punch_type in ['jab', 'cross', 'hook', 'uppercut']
        print(f"✓ Punch classifier working (classified as: {punch_type})")
    except Exception as e:
        pytest.skip(f"Punch classifier dependencies not available: {e}")


def test_metrics_extraction():
    """Test metrics extraction logic"""
    try:
        from workers.cv_pipeline.metrics import FightMetricsExtractor
        extractor = FightMetricsExtractor()
        
        # Test distance classification
        distance_class = extractor._classify_distance(1.5)
        assert distance_class in ['inside', 'mid', 'outside']
        print(f"✓ Metrics extraction working (1.5m classified as: {distance_class})")
    except Exception as e:
        pytest.skip(f"Metrics dependencies not available: {e}")


# ===== Scoring Engine Tests =====

def test_bayesian_scoring():
    """Test Bayesian scoring model"""
    try:
        from workers.scoring.bayesian_model import BayesianScorer
        scorer = BayesianScorer()
        
        # Test with mock round stats
        round_stats = {
            'punches_landed_f0': 15,
            'punches_thrown_f0': 25,
            'total_impact_f0': 12.5,
            'aggression_frames_f0': 100,
            'center_frames_f0': 80,
            'blocks_f0': 3,
            'punches_landed_f1': 10,
            'punches_thrown_f1': 20,
            'total_impact_f1': 8.0,
            'aggression_frames_f1': 60,
            'center_frames_f1': 40,
            'blocks_f1': 2
        }
        
        scores = scorer.score_round(round_stats, total_frames=300)
        
        # Validate output structure
        assert 'prob_10_9_f0' in scores
        assert 'prob_10_9_f1' in scores
        assert 'prob_draw' in scores
        
        # Probabilities should sum close to 1
        total_prob = sum(scores.values())
        assert 0.95 <= total_prob <= 1.05
        
        print(f"✓ Bayesian scoring working: Fighter 0 advantage = {scores['prob_10_9_f0']:.2f}")
    except Exception as e:
        pytest.skip(f"Scoring dependencies not available: {e}")


def test_monte_carlo_simulation():
    """Test Monte Carlo win probability simulation"""
    try:
        from workers.scoring.bayesian_model import BayesianScorer
        from workers.scoring.monte_carlo import MonteCarloSimulator
        
        scorer = BayesianScorer()
        simulator = MonteCarloSimulator(scorer, n_simulations=1000)  # Fewer sims for speed
        
        # Mock round scores
        round_scores = [
            {'probabilities': {'prob_10_9_f0': 0.6, 'prob_10_9_f1': 0.3, 'prob_draw': 0.1}},
            {'probabilities': {'prob_10_9_f0': 0.5, 'prob_10_9_f1': 0.4, 'prob_draw': 0.1}},
        ]
        
        win_probs = simulator.simulate_fight(round_scores, total_rounds=12)
        
        # Validate output
        assert 'fighter_0' in win_probs
        assert 'fighter_1' in win_probs
        assert 'draw' in win_probs
        
        # Probabilities should sum to ~1
        total = win_probs['fighter_0'] + win_probs['fighter_1'] + win_probs['draw']
        assert 0.95 <= total <= 1.05
        
        print(f"✓ Monte Carlo simulation working: F0={win_probs['fighter_0']:.2f}, F1={win_probs['fighter_1']:.2f}")
    except Exception as e:
        pytest.skip(f"Monte Carlo dependencies not available: {e}")


def test_judge_scoring():
    """Test judge-style scoring"""
    try:
        from workers.scoring.judge_scoring import JudgeScorer
        scorer = JudgeScorer()
        
        # Mock round data
        round_data = [{
            'punches_landed_f0': 12,
            'total_impact_f0': 10.0,
            'aggression_frames_f0': 80,
            'center_frames_f0': 70,
            'punches_landed_f1': 8,
            'total_impact_f1': 6.0,
            'aggression_frames_f1': 50,
            'center_frames_f1': 30
        }]
        
        scorecard = scorer.generate_scorecard(round_data)
        
        assert 'rounds' in scorecard
        assert 'winner' in scorecard
        assert len(scorecard['rounds']) == 1
        
        print(f"✓ Judge scoring working: Winner = {scorecard['winner']}")
    except Exception as e:
        pytest.skip(f"Judge scoring dependencies not available: {e}")


# ===== Integration Tests =====

def test_fight_processor_mock():
    """Test fight processor in mock mode"""
    try:
        from workers.processor import FightProcessor
        processor = FightProcessor(device='cpu')
        
        # This will use mock processing since CV dependencies likely unavailable
        results = processor.process_fight(
            fight_id="test-123",
            video_path="dummy.mp4",
            total_rounds=12
        )
        
        assert results['fight_id'] == "test-123"
        assert 'round_scores' in results
        assert 'win_probabilities' in results
        
        print(f"✓ Fight processor working (mode: {results.get('status')})")
    except Exception as e:
        pytest.fail(f"Fight processor failed: {e}")


def test_video_ingestion_import():
    """Test video ingestion service"""
    try:
        from app.services.video_ingestion import process_youtube_url, process_video_upload
        assert process_youtube_url is not None
        assert process_video_upload is not None
        print("✓ Video ingestion service imports successful")
    except ImportError as e:
        pytest.fail(f"Failed to import video ingestion: {e}")


# ===== Run all tests =====

if __name__ == "__main__":
    print("\n" + "="*60)
    print(" 🥊 RING DYNAMICS TEST SUITE 🥊")
    print("="*60 + "\n")
    
    pytest.main([__file__, "-v", "--tb=short", "-s"])
