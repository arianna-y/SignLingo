import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import torch
import random
import time
from datetime import datetime
import json
import os
from model import SignLSTM
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="SignLingo - Learn ASL",
    page_icon="👋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'user_profile' not in st.session_state:
    st.session_state.user_profile = {
        'total_score': 0,
        'level': 1,
        'xp': 0,
        'words_learned': [],
        'practice_history': [],
        'streak_days': 0,
        'last_practice': None
    }

if 'current_page' not in st.session_state:
    st.session_state.current_page = 'Home'

# ASL Dictionary with expanded vocabulary
ASL_DICTIONARY = {
    'hello': {
        'description': 'A friendly greeting gesture',
        'how_to': '1. Start with hand near your forehead\n2. Move hand outward in a small wave\n3. Keep fingers together',
        'difficulty': 'Beginner',
        'category': 'Greetings',
        'points': 10
    },
    'thanks': {
        'description': 'Express gratitude',
        'how_to': '1. Place flat hand on chin\n2. Move hand forward and down\n3. End with palm facing up',
        'difficulty': 'Beginner',
        'category': 'Greetings',
        'points': 10
    },
    'yes': {
        'description': 'Affirmative response',
        'how_to': '1. Make a fist\n2. Nod the fist up and down\n3. Similar to nodding your head',
        'difficulty': 'Beginner',
        'category': 'Basic Responses',
        'points': 10
    },
    'please': {
        'description': 'Polite request',
        'how_to': '1. Place flat hand on chest\n2. Move in circular motion\n3. Maintain contact with chest',
        'difficulty': 'Beginner',
        'category': 'Greetings',
        'points': 15
    },
    'sorry': {
        'description': 'Express apology',
        'how_to': '1. Make a fist\n2. Rub in circular motion on chest\n3. Show sincere expression',
        'difficulty': 'Beginner',
        'category': 'Emotions',
        'points': 15
    },
    'help': {
        'description': 'Request assistance',
        'how_to': '1. Place fist on flat palm\n2. Lift both hands together\n3. Movement is upward',
        'difficulty': 'Intermediate',
        'category': 'Basic Needs',
        'points': 20
    }
}

# Load model
@st.cache_resource
def load_model():
    model = SignLSTM()
    try:
        model.load_state_dict(torch.load('sign_language_model.pth', map_location=torch.device('cpu')))
        model.eval()
        return model
    except:
        return None

# MediaPipe setup
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def extract_keypoints(results):
    if results.left_hand_landmarks:
        return np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten()
    elif results.right_hand_landmarks:
        return np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten()
    return np.zeros(21 * 3)

def normalize_sequence(sequence_list):
    seq_arr = np.array(sequence_list)
    seq_reshaped = seq_arr.reshape(30, 21, 3)
    wrists = seq_reshaped[:, 0, :].reshape(30, 1, 3)
    seq_norm = seq_reshaped - wrists
    return seq_norm.reshape(30, 63)

def calculate_level(xp):
    return int(xp / 100) + 1

def xp_to_next_level(xp):
    current_level = calculate_level(xp)
    return current_level * 100 - xp

def save_user_profile():
    with open('user_profile.json', 'w') as f:
        json.dump(st.session_state.user_profile, f)

def load_user_profile():
    try:
        with open('user_profile.json', 'r') as f:
            st.session_state.user_profile = json.load(f)
    except:
        pass

# Sidebar Navigation
with st.sidebar:
    st.title("SignLingo")
    st.markdown("---")
    
    # User Profile Summary
    profile = st.session_state.user_profile
    st.markdown(f"### Level {calculate_level(profile['xp'])}")
    st.progress(profile['xp'] % 100 / 100)
    st.caption(f"{xp_to_next_level(profile['xp'])} XP to next level")
    st.markdown(f"**Total Score:** {profile['total_score']}")
    st.markdown(f"**Words Learned:** {len(profile['words_learned'])}")
    st.markdown(f"**Streak:** {profile['streak_days']} days 🔥")
    
    st.markdown("---")
    
    # Navigation
    pages = ['🏠 Home', '📚 Learn', '🎮 Practice', '🏆 Test Yourself', '📊 Progress', '📖 Dictionary']
    for page in pages:
        if st.button(page, use_container_width=True):
            st.session_state.current_page = page.split(' ', 1)[1]
            st.rerun()

# Main Content Area
page = st.session_state.current_page

if page == 'Home':
    st.title("Welcome to SignLingo!")
    st.markdown("### Your Journey to ASL Fluency Starts Here")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("### 📚 Learn\nExplore ASL vocabulary with interactive tutorials")
    
    with col2:
        st.success("### 🎮 Practice\nPractice signs with real-time feedback")
    
    with col3:
        st.warning("### 🏆 Test\nChallenge yourself and earn points!")
    
    st.markdown("---")
    
    # Quick Stats
    st.subheader("Today's Quick Stats")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Current Level", calculate_level(profile['xp']))
    with col2:
        st.metric("Total XP", profile['xp'])
    with col3:
        st.metric("Words Learned", len(profile['words_learned']))
    with col4:
        st.metric("Current Streak", f"{profile['streak_days']} days")
    
    # Recent Activity
    if profile['practice_history']:
        st.subheader("Recent Activity")
        recent = profile['practice_history'][-5:]
        for activity in reversed(recent):
            st.markdown(f"- **{activity['word']}** - Score: {activity['score']} - {activity['date']}")

elif page == 'Learn':
    st.title("📚 Learn ASL Signs")
    st.markdown("Choose a category and learn new signs!")
    
    # Category filter
    categories = list(set([word['category'] for word in ASL_DICTIONARY.values()]))
    selected_category = st.selectbox("Filter by Category", ['All'] + categories)
    
    # Display dictionary
    filtered_words = ASL_DICTIONARY.items()
    if selected_category != 'All':
        filtered_words = [(k, v) for k, v in ASL_DICTIONARY.items() if v['category'] == selected_category]
    
    cols = st.columns(2)
    for idx, (word, info) in enumerate(filtered_words):
        with cols[idx % 2]:
            with st.expander(f"**{word.upper()}** - {info['difficulty']}", expanded=False):
                st.markdown(f"**Category:** {info['category']}")
                st.markdown(f"**Points:** {info['points']}")
                st.markdown(f"**Description:** {info['description']}")
                st.markdown("**How to Sign:**")
                st.markdown(info['how_to'])
                
                learned = word in profile['words_learned']
                if st.button(f"{'✓ Learned' if learned else 'Mark as Learned'}", key=f"learn_{word}"):
                    if word not in profile['words_learned']:
                        profile['words_learned'].append(word)
                        profile['xp'] += info['points']
                        st.success(f"Great! You learned '{word}' and earned {info['points']} XP!")
                        save_user_profile()
                        st.rerun()

elif page == 'Practice':
    st.title("🎮 Practice Mode")
    st.markdown("Practice signs with real-time feedback from your webcam!")
    
    model = load_model()
    if model is None:
        st.error("Model not loaded. Please train the model first.")
        st.stop()
    
    # Practice settings
    col1, col2 = st.columns(2)
    with col1:
        practice_words = st.multiselect(
            "Select words to practice",
            options=['hello', 'thanks', 'yes'],
            default=['hello', 'thanks', 'yes']
        )
    with col2:
        confidence_threshold = st.slider("Confidence Threshold", 0.5, 0.95, 0.8)
    
    if not practice_words:
        st.warning("Please select at least one word to practice!")
        st.stop()
    
    start_practice = st.button("Start Practice Session", type="primary")
    stop_practice = st.button("Stop Practice")
    
    FRAME_WINDOW = st.image([])
    status_text = st.empty()
    score_text = st.empty()
    
    if start_practice:
        cap = cv2.VideoCapture(0)
        sequence_buffer = []
        score = 0
        target_word = random.choice(practice_words)
        
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while cap.isOpened() and not stop_practice:
                ret, frame = cap.read()
                if not ret:
                    break
                
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(image_rgb)
                image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                
                # Draw landmarks
                mp_drawing.draw_landmarks(image_bgr, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                mp_drawing.draw_landmarks(image_bgr, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                
                # Extract and buffer keypoints
                keypoints = extract_keypoints(results)
                sequence_buffer.append(keypoints)
                sequence_buffer = sequence_buffer[-30:]
                
                prediction_text = "..."
                
                if len(sequence_buffer) == 30:
                    norm_seq = normalize_sequence(sequence_buffer)
                    input_tensor = torch.FloatTensor([norm_seq])
                    
                    with torch.no_grad():
                        res = model(input_tensor)
                        prob = torch.softmax(res, dim=1)
                        prob_np = prob.detach().cpu().numpy()[0]
                        best_class_idx = np.argmax(prob_np)
                        confidence = prob_np[best_class_idx]
                    
                    if confidence > confidence_threshold:
                        predicted_word = practice_words[best_class_idx]
                        prediction_text = f"{predicted_word} ({confidence*100:.1f}%)"
                        
                        if predicted_word == target_word:
                            score += 1
                            profile['total_score'] += 1
                            profile['xp'] += 5
                            sequence_buffer = []
                            possible = [w for w in practice_words if w != target_word]
                            target_word = random.choice(possible) if possible else random.choice(practice_words)
                
                # Add UI elements to frame
                cv2.rectangle(image_bgr, (0, 0), (640, 60), (245, 117, 16), -1)
                cv2.putText(image_bgr, f"Sign: {target_word.upper()}", (20, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
                FRAME_WINDOW.image(image_bgr, channels="BGR")
                status_text.markdown(f"**Target:** {target_word.upper()} | **Prediction:** {prediction_text}")
                score_text.metric("Score", score)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        cap.release()
        save_user_profile()
        st.success(f"Practice session complete! Final score: {score}")

elif page == 'Test Yourself':
    st.title("🏆 Test Your Skills")
    st.markdown("Take a timed test to earn more points and level up!")
    
    test_type = st.radio("Select Test Type", ['Quick Test (5 signs)', 'Standard Test (10 signs)', 'Expert Test (20 signs)'])
    
    test_config = {
        'Quick Test (5 signs)': {'count': 5, 'time_limit': 60, 'multiplier': 1},
        'Standard Test (10 signs)': {'count': 10, 'time_limit': 120, 'multiplier': 1.5},
        'Expert Test (20 signs)': {'count': 20, 'time_limit': 240, 'multiplier': 2}
    }
    
    config = test_config[test_type]
    
    st.info(f"**Signs to complete:** {config['count']}\n**Time Limit:** {config['time_limit']}s\n**Point Multiplier:** {config['multiplier']}x")
    
    if st.button("Start Test", type="primary"):
        st.warning("Test mode would activate webcam here with timer and scoring!")
        st.markdown("*This would launch the same practice interface but with:*")
        st.markdown("- Timer countdown")
        st.markdown("- Fixed number of signs to complete")
        st.markdown("- Bonus points for speed and accuracy")
        st.markdown("- Final score summary with XP rewards")

elif page == 'Progress':
    st.title("📊 Your Progress")
    
    # Level Progress
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Level Progress")
        current_level = calculate_level(profile['xp'])
        progress = (profile['xp'] % 100) / 100
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=profile['xp'] % 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"Level {current_level}"},
            delta={'reference': 0},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 100], 'color': "gray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Learning Stats")
        st.metric("Total XP", profile['xp'])
        st.metric("Words Learned", f"{len(profile['words_learned'])}/{len(ASL_DICTIONARY)}")
        st.metric("Completion", f"{len(profile['words_learned'])/len(ASL_DICTIONARY)*100:.1f}%")
        st.metric("Current Streak", f"{profile['streak_days']} days 🔥")
    
    # Practice History Chart
    if profile['practice_history']:
        st.subheader("Practice History")
        history_data = profile['practice_history'][-20:]
        
        dates = [h['date'] for h in history_data]
        scores = [h['score'] for h in history_data]
        
        fig = px.line(x=dates, y=scores, labels={'x': 'Date', 'y': 'Score'}, 
                     title='Recent Practice Scores')
        fig.update_traces(mode='lines+markers')
        st.plotly_chart(fig, use_container_width=True)
    
    # Words Mastery
    st.subheader("Vocabulary Mastery")
    categories = {}
    for word, info in ASL_DICTIONARY.items():
        cat = info['category']
        if cat not in categories:
            categories[cat] = {'total': 0, 'learned': 0}
        categories[cat]['total'] += 1
        if word in profile['words_learned']:
            categories[cat]['learned'] += 1
    
    for category, stats in categories.items():
        progress = stats['learned'] / stats['total']
        st.progress(progress, text=f"{category}: {stats['learned']}/{stats['total']}")

elif page == 'Dictionary':
    st.title("📖 ASL Dictionary")
    st.markdown("Complete reference guide for all ASL signs")
    
    # Search functionality
    search_term = st.text_input("Search for a sign", "")
    
    # Difficulty filter
    difficulty_filter = st.multiselect("Filter by Difficulty", ['Beginner', 'Intermediate', 'Advanced'], default=['Beginner', 'Intermediate', 'Advanced'])
    
    # Display all words
    st.markdown("---")
    
    filtered_dict = {k: v for k, v in ASL_DICTIONARY.items() 
                    if (search_term.lower() in k.lower() or search_term.lower() in v['description'].lower())
                    and v['difficulty'] in difficulty_filter}
    
    for word, info in filtered_dict.items():
        with st.container():
            col1, col2, col3 = st.columns([2, 3, 1])
            
            with col1:
                st.markdown(f"### {word.upper()}")
                st.markdown(f"*{info['difficulty']}* | {info['category']}")
            
            with col2:
                st.markdown(f"**{info['description']}**")
                with st.expander("How to sign"):
                    st.markdown(info['how_to'])
            
            with col3:
                learned = word in profile['words_learned']
                status = "✓ Learned" if learned else "Not learned"
                st.markdown(f"**{status}**")
                st.markdown(f"*{info['points']} points*")
            
            st.markdown("---")

# Footer
st.sidebar.markdown("---")
st.sidebar.caption("SignLingo v2.0 | Made with ❤️")
if st.sidebar.button("Save Progress"):
    save_user_profile()
    st.sidebar.success("Progress saved!")