"""
SignLingo Utility Functions
Helper functions for video processing, data management, and user progress tracking
"""

import cv2
import numpy as np
import mediapipe as mp
import torch
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

# MediaPipe initialization
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

class UserProgressManager:
    """Manages user progress, achievements, and learning analytics"""
    
    def __init__(self, profile_path: str = 'user_profile.json'):
        self.profile_path = profile_path
        self.profile = self.load_profile()
    
    def load_profile(self) -> Dict:
        """Load user profile from JSON file"""
        default_profile = {
            'user_id': self.generate_user_id(),
            'username': 'ASL Learner',
            'total_score': 0,
            'level': 1,
            'xp': 0,
            'words_learned': [],
            'words_mastered': [],
            'practice_history': [],
            'test_history': [],
            'streak_days': 0,
            'last_practice': None,
            'total_practice_time': 0,
            'achievements': [],
            'daily_goals': {'target': 5, 'completed': 0},
            'weekly_stats': {},
            'created_at': datetime.now().isoformat()
        }
        
        try:
            if os.path.exists(self.profile_path):
                with open(self.profile_path, 'r') as f:
                    loaded_profile = json.load(f)
                    # Merge with default to ensure all keys exist
                    return {**default_profile, **loaded_profile}
            return default_profile
        except Exception as e:
            print(f"Error loading profile: {e}")
            return default_profile
    
    def save_profile(self):
        """Save user profile to JSON file"""
        try:
            with open(self.profile_path, 'w') as f:
                json.dump(self.profile, f, indent=2)
        except Exception as e:
            print(f"Error saving profile: {e}")
    
    @staticmethod
    def generate_user_id() -> str:
        """Generate unique user ID"""
        return f"user_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    def update_streak(self):
        """Update daily streak"""
        today = datetime.now().date()
        last_practice = self.profile.get('last_practice')
        
        if last_practice:
            last_date = datetime.fromisoformat(last_practice).date()
            days_diff = (today - last_date).days
            
            if days_diff == 0:
                return  # Already practiced today
            elif days_diff == 1:
                self.profile['streak_days'] += 1
            else:
                self.profile['streak_days'] = 1
        else:
            self.profile['streak_days'] = 1
        
        self.profile['last_practice'] = datetime.now().isoformat()
        self.save_profile()
    
    def add_xp(self, amount: int, reason: str = "practice"):
        """Add XP and handle level ups"""
        old_level = self.calculate_level(self.profile['xp'])
        self.profile['xp'] += amount
        new_level = self.calculate_level(self.profile['xp'])
        
        if new_level > old_level:
            return {'leveled_up': True, 'new_level': new_level}
        return {'leveled_up': False}
    
    @staticmethod
    def calculate_level(xp: int) -> int:
        """Calculate level from XP (100 XP per level)"""
        return int(xp / 100) + 1
    
    @staticmethod
    def xp_to_next_level(xp: int) -> int:
        """Calculate XP needed for next level"""
        current_level = UserProgressManager.calculate_level(xp)
        return current_level * 100 - xp
    
    def log_practice(self, word: str, score: int, duration: int, accuracy: float):
        """Log a practice session"""
        practice_entry = {
            'word': word,
            'score': score,
            'duration': duration,
            'accuracy': accuracy,
            'date': datetime.now().isoformat(),
            'timestamp': datetime.now().timestamp()
        }
        self.profile['practice_history'].append(practice_entry)
        self.profile['total_practice_time'] += duration
        self.profile['total_score'] += score
        
        # Update daily goals
        self.profile['daily_goals']['completed'] += 1
        
        # Check for achievements
        self.check_achievements()
        
        self.update_streak()
        self.save_profile()
    
    def log_test(self, test_type: str, words: List[str], score: int, 
                 total: int, duration: int):
        """Log a test session"""
        test_entry = {
            'test_type': test_type,
            'words': words,
            'score': score,
            'total': total,
            'accuracy': (score / total) * 100 if total > 0 else 0,
            'duration': duration,
            'date': datetime.now().isoformat()
        }
        self.profile['test_history'].append(test_entry)
        self.save_profile()
    
    def mark_word_learned(self, word: str, points: int = 10):
        """Mark a word as learned"""
        if word not in self.profile['words_learned']:
            self.profile['words_learned'].append(word)
            self.add_xp(points, "learning")
            self.save_profile()
            return True
        return False
    
    def mark_word_mastered(self, word: str, points: int = 20):
        """Mark a word as mastered (high accuracy over multiple sessions)"""
        if word not in self.profile['words_mastered']:
            self.profile['words_mastered'].append(word)
            self.add_xp(points, "mastery")
            self.save_profile()
            return True
        return False
    
    def check_achievements(self):
        """Check and award achievements"""
        achievements = []
        
        # First sign achievement
        if len(self.profile['words_learned']) == 1 and 'first_sign' not in self.profile['achievements']:
            achievements.append('first_sign')
        
        # 5 signs learned
        if len(self.profile['words_learned']) >= 5 and 'five_signs' not in self.profile['achievements']:
            achievements.append('five_signs')
        
        # Week streak
        if self.profile['streak_days'] >= 7 and 'week_streak' not in self.profile['achievements']:
            achievements.append('week_streak')
        
        # 100 practice score
        if self.profile['total_score'] >= 100 and 'century' not in self.profile['achievements']:
            achievements.append('century')
        
        # Add new achievements to profile
        for achievement in achievements:
            if achievement not in self.profile['achievements']:
                self.profile['achievements'].append(achievement)
        
        return achievements


class SignRecognitionEngine:
    """Handles sign language recognition using MediaPipe and PyTorch"""
    
    def __init__(self, model_path: str = 'sign_language_model.pth'):
        self.model_path = model_path
        self.model = None
        self.holistic = None
        self.sequence_buffer = []
        self.sequence_length = 30
        
    def load_model(self, model_class):
        """Load trained PyTorch model"""
        try:
            self.model = model_class()
            self.model.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu')))
            self.model.eval()
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def initialize_holistic(self):
        """Initialize MediaPipe Holistic"""
        self.holistic = mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    def extract_keypoints(self, results) -> np.ndarray:
        """Extract hand keypoints from MediaPipe results"""
        if results.left_hand_landmarks:
            return np.array([[res.x, res.y, res.z] 
                           for res in results.left_hand_landmarks.landmark]).flatten()
        elif results.right_hand_landmarks:
            return np.array([[res.x, res.y, res.z] 
                           for res in results.right_hand_landmarks.landmark]).flatten()
        return np.zeros(21 * 3)
    
    def normalize_sequence(self, sequence_list: List[np.ndarray]) -> np.ndarray:
        """Normalize sequence relative to wrist position"""
        seq_arr = np.array(sequence_list)
        seq_reshaped = seq_arr.reshape(self.sequence_length, 21, 3)
        wrists = seq_reshaped[:, 0, :].reshape(self.sequence_length, 1, 3)
        seq_norm = seq_reshaped - wrists
        return seq_norm.reshape(self.sequence_length, 63)
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[Dict]]:
        """Process a single frame and return annotated frame and results"""
        if self.holistic is None:
            self.initialize_holistic()
        
        # Convert to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        
        # Process with MediaPipe
        results = self.holistic.process(image_rgb)
        
        # Convert back to BGR for OpenCV
        image_rgb.flags.writeable = True
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        # Draw landmarks
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(
                image_bgr,
                results.left_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
        
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(
                image_bgr,
                results.right_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
        
        return image_bgr, results
    
    def predict_sign(self, actions: List[str], confidence_threshold: float = 0.8) -> Optional[Tuple[str, float]]:
        """Predict sign from sequence buffer"""
        if self.model is None or len(self.sequence_buffer) < self.sequence_length:
            return None
        
        # Normalize and prepare input
        norm_seq = self.normalize_sequence(self.sequence_buffer[-self.sequence_length:])
        input_tensor = torch.FloatTensor([norm_seq])
        
        # Get prediction
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            prob_np = probabilities.detach().cpu().numpy()[0]
            
            best_class_idx = np.argmax(prob_np)
            confidence = prob_np[best_class_idx]
        
        if confidence > confidence_threshold and best_class_idx < len(actions):
            return actions[best_class_idx], float(confidence)
        
        return None
    
    def add_frame_to_buffer(self, results):
        """Add keypoints from current frame to sequence buffer"""
        keypoints = self.extract_keypoints(results)
        self.sequence_buffer.append(keypoints)
        
        # Keep only last N frames
        if len(self.sequence_buffer) > self.sequence_length:
            self.sequence_buffer.pop(0)
    
    def clear_buffer(self):
        """Clear sequence buffer"""
        self.sequence_buffer = []
    
    def cleanup(self):
        """Clean up resources"""
        if self.holistic:
            self.holistic.close()


class VideoAnnotator:
    """Utility for adding UI elements to video frames"""
    
    @staticmethod
    def add_header(frame: np.ndarray, text: str, 
                   bg_color: Tuple[int, int, int] = (245, 117, 16)) -> np.ndarray:
        """Add header text to frame"""
        height, width = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (width, 60), bg_color, -1)
        cv2.putText(frame, text, (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        return frame
    
    @staticmethod
    def add_footer(frame: np.ndarray, text: str,
                   color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        """Add footer text to frame"""
        height, width = frame.shape[:2]
        cv2.putText(frame, text, (20, height - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        return frame
    
    @staticmethod
    def add_success_banner(frame: np.ndarray, message: str = "Correct!") -> np.ndarray:
        """Add success banner to frame"""
        height, width = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (width, 80), (0, 255, 0), -1)
        cv2.putText(frame, message, (width // 2 - 100, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)
        return frame
    
    @staticmethod
    def add_countdown(frame: np.ndarray, seconds: int) -> np.ndarray:
        """Add countdown timer to frame"""
        height, width = frame.shape[:2]
        cv2.putText(frame, f"Time: {seconds}s", (width - 150, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        return frame


# Achievement definitions
ACHIEVEMENTS = {
    'first_sign': {
        'name': 'First Steps',
        'description': 'Learn your first ASL sign',
        'icon': '🌟',
        'points': 50
    },
    'five_signs': {
        'name': 'Getting Started',
        'description': 'Learn 5 different signs',
        'icon': '⭐',
        'points': 100
    },
    'week_streak': {
        'name': 'Dedicated Learner',
        'description': 'Practice for 7 days in a row',
        'icon': '🔥',
        'points': 200
    },
    'century': {
        'name': 'Century Club',
        'description': 'Score 100 total points',
        'icon': '💯',
        'points': 150
    }
}


def get_achievement_info(achievement_id: str) -> Dict:
    """Get achievement information"""
    return ACHIEVEMENTS.get(achievement_id, {
        'name': 'Unknown',
        'description': 'Unknown achievement',
        'icon': '❓',
        'points': 0
    })