import streamlit as st
import cv2
import mediapipe as mp
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.posechecker import PoseChecker


def main():
    st.title("Pose Recognition App")
    
    # Initialize MediaPipe
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    mp_draw = mp.solutions.drawing_utils
    pose_checker = PoseChecker()
    
    # Streamlit UI elements
    pose_name = st.sidebar.selectbox(
        "Select Pose",
        ["warrior", "tree", "goddess", "downdog", "plank"]  # Add your pose options here
    )
    st.write(f"Selected pose: {pose_name}")

    col1,col2 = st.columns([2,1])
    with col1:
        start_button = st.button("Start")
        video_placeholder = st.empty()
        stop_button = st.button("Stop")
    
    with col2:
        feedback_placeholder = st.empty()
        reference_placeholder = st.empty()
        if st.sidebar.checkbox("Show Reference Pose"):
            pose_path = os.path.join("reference", f"{pose_name}.jpg")
            if os.path.exists(pose_path):
                reference_img = cv2.imread(pose_path)
                if reference_img is not None:
                    reference_img = cv2.cvtColor(reference_img, cv2.COLOR_BGR2RGB)
                    reference_placeholder.image(reference_img, caption="Reference Pose", use_column_width=True)
                else:
                    st.error("Failed to load reference image")
            else:
                st.error("Reference image not found")
    
    # Create a placeholder for the webcam feed

    
    
    # Add a stop button
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened() and start_button and not stop_button:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to access webcam")
            break
            
        # Convert the frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = pose.process(frame_rgb)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Assuming check_pose function exists
            pose_check_result = pose_checker.check_pose(landmarks, pose_name)
            outline_color = (0, 255, 0) if pose_check_result["is_correct"] else (255, 0, 0)
            
            custom_connections = mp_draw.DrawingSpec(color=outline_color, thickness=2)
            custom_landmarks = mp_draw.DrawingSpec(color=outline_color, circle_radius=2)
            
            # Draw landmarks
            mp_draw.draw_landmarks(
                frame_rgb,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=custom_landmarks,
                connection_drawing_spec=custom_connections
            )
            
            # Display feedback
            feedback_text = "<br>".join(pose_check_result["feedback"])
            # print(feedback_text)
            feedback_color = "green" if pose_check_result["is_correct"] else "red"
            feedback_placeholder.markdown(
                f"""<div style="background-color: {feedback_color}20">
                    <h2 class="feedback-text" style="color: {feedback_color}">{feedback_text}</h2>
                </div>""",
                unsafe_allow_html=True
        )
        video_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
        
    # Clean up
    cap.release()
    pose.close()


if __name__ == "__main__":
    st.set_page_config(
        page_title="Pose Recognition",
        page_icon="🧘‍♀️",
        layout="wide"
    )
    main()