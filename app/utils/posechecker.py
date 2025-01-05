import numpy as np
import mediapipe as mp
from pathlib import Path

class PoseChecker:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose_configs = {
            "warrior": {
                "knee_front": {"target": 90, "range": (90, 120)},
                "knee_back": {"target": 180, "range": (165, 195)},
                "arms": {"target": 180, "range": (160, 195)},
                "shoulders": {"target": 100, "range": (80, 120)}
            },
            "tree": {
                "knee_bent": {"target": 30, "range": (0, 50)},
                "knee_straight": {"target": 180, "range": (165, 195)},
                "shoulders": {"target": 150, "range": (150, 180)},
                "elbows": {"target": 180, "range": (160, 195)},
                "torso": {"target": 150, "range": (140, 160)}
            },
            "goddess": {
                "knees": {"target": 90, "range": (0, 110)},
                "shoulders": {"target": 90, "range": (0, 110)},
                "elbows": {"target": 90, "range": (80, 100)},
                "torso": {"target": 90, "range": (0, 120)}
            },
            "downdog": {
                "elbows": {"target": 180, "range": (160, 195)},
                "shoulders": {"target": 180, "range": (160, 195)},
                "torso": {"target": 90, "range": (0, 100)}
            },
            "plank": {
                "elbows": {"target": 180, "range": (160, 195)},
                "shoulders": {"target": 75, "range": (60, 90)},
                "torso": {"target": 180, "range": (160, 195)},
                "knees": {"target": 180, "range": (160, 195)}
            }
        }

    def get_pose_path(self, pose_name: str) -> str:
        """Get the reference image path for a pose."""
        return str(Path('./reference') / f'{pose_name}.jpg')

    @staticmethod
    def calculate_angle(landmark1, landmark2, landmark3) -> float:
        """Calculate the angle between three landmarks."""
        x1, y1 = landmark1.x, landmark1.y
        x2, y2 = landmark2.x, landmark2.y
        x3, y3 = landmark3.x, landmark3.y
        
        radians = np.arctan2(y3-y2, x3-x2) - np.arctan2(y1-y2, x1-x2)
        angle = np.abs(radians * 180.0 / np.pi)
        
        if angle > 180.0:
            angle = 360 - angle
            
        return angle

    def get_body_angles(self, landmarks) -> dict:
        """Calculate all relevant body angles from landmarks."""
        return {
            "left_elbow": self.calculate_angle(
                landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value],
                landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
            ),
            "right_elbow": self.calculate_angle(
                landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
            ),
            "left_shoulder": self.calculate_angle(
                landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value],
                landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
            ),
            "right_shoulder": self.calculate_angle(
                landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value],
                landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value]
            ),
            "left_knee": self.calculate_angle(
                landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value],
                landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value],
                landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
            ),
            "right_knee": self.calculate_angle(
                landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value],
                landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value],
                landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]
            ),
            "right_torso": self.calculate_angle(
                landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value],
                landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value]
            ),
            "left_torso": self.calculate_angle(
                landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value],
                landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value]
            )
        }

    def check_pose(self, landmarks, pose_name: str) -> dict:
        """Check if the current pose matches the target pose."""
        result = {
            "is_correct": False,
            "feedback": []
        }

        if pose_name.lower() not in self.pose_configs:
            result["feedback"].append("Unknown pose")
            return result

        angles = self.get_body_angles(landmarks)
        config = self.pose_configs[pose_name.lower()]

        if pose_name.lower() == "warrior":
            self._check_warrior_pose(angles, result)
        elif pose_name.lower() == "tree":
            self._check_tree_pose(angles, result)
        elif pose_name.lower() == "goddess":
            self._check_goddess_pose(angles, result)
        elif pose_name.lower() == "downdog":
            self._check_downdog_pose(angles, result)
        elif pose_name.lower() == "plank":
            self._check_plank_pose(angles, result)

        # Check if all feedback is positive
        if len(result["feedback"]) > 0 and not any("Adjust" in feedback or "Unknown" in feedback for feedback in result["feedback"]):
            result["is_correct"] = True

        return result

    def _check_warrior_pose(self, angles: dict, result: dict):
        """Check warrior pose specific angles."""
        # warrior_config = self.pose_configs['warrior']
        if (angles["left_knee"] >= 90 and angles["left_knee"] < 120) or (angles["right_knee"] >= 90 and angles["right_knee"] < 120):
            result["feedback"].append("knee angle is correct")
        else:
            result["feedback"].append("Adjust front knee angle, target: 90")

        if (angles["left_knee"] > 165 and angles["left_knee"] < 195) or (angles["right_knee"] > 165 and angles["right_knee"] < 195):
            result["feedback"].append("Back leg is straight")
        else:
            result["feedback"].append("Straighten back leg")

        if (angles["left_elbow"] > 160) and (angles["right_elbow"] > 160):
            if angles["left_shoulder"] > 80 and angles["left_shoulder"] < 120 and angles["right_shoulder"] > 80 and angles["right_shoulder"] < 120:
                result["feedback"].append("Arms are correctly positioned")
            else:
                result["feedback"].append("Adjust arm position, target: 180")
        else:
            result["feedback"].append("Adjust arm position, target: 180")

    def _check_tree_pose(self, angles: dict, result: dict):
        """Check tree pose specific angles."""
        # Check bent knee angle
        if (angles["left_knee"] < 50) or (angles["right_knee"] < 50):
            result["feedback"].append("knee angle is correct")
        else:
            result["feedback"].append("Adjust front knee angle, target: 30°")
        
        # Check straight leg
        if (angles["left_knee"] > 165 and angles["left_knee"] < 195) or \
        (angles["right_knee"] > 165 and angles["right_knee"] < 195):
            result["feedback"].append("Back leg is straight")
        else:
            result["feedback"].append("Straighten back leg")
        
        # Check shoulder angles
        if angles["left_shoulder"] > 150 and angles["right_shoulder"] > 150:
            result["feedback"].append("Shoulder angle is correct")
        else:
            result["feedback"].append("Adjust shoulder angle, target: 150")
        
        # Check elbow angles
        if (angles["left_elbow"] > 160) and (angles["right_elbow"] > 160):
            result["feedback"].append("Elbow angle is correct")
        else:
            result["feedback"].append("Adjust elbow angle, target: 180°")
        
        # Check torso angles
        if (angles["left_torso"] > 160 or angles["right_torso"] > 160) and \
        (angles["left_torso"] < 140 or angles["right_torso"] < 140):
            result["feedback"].append("Torso angle is correct")
        else:
            result["feedback"].append(f"{angles['left_torso']:.1f}Adjust torso angle")

    def _check_goddess_pose(self, angles: dict, result: dict):
        """Check goddess pose specific angles."""
        # Check knee angles
        if angles["left_knee"] < 110 and angles["right_knee"] < 110:
            result["feedback"].append("Knee angle is correct")
        else:
            result["feedback"].append("Adjust knee angle, target: 90°")
        
        # Check shoulder angles
        if angles["left_shoulder"] < 110 and angles["right_shoulder"] < 110:
            result["feedback"].append("Shoulder angle is correct")
        else:
            result["feedback"].append("Adjust shoulder angle, target: 90°")
        
        # Check elbow angles
        if (angles["left_elbow"] < 100 and angles["left_elbow"] > 80) and \
        (angles["right_elbow"] < 110 and angles["right_elbow"] > 80):
            result["feedback"].append("Elbow angle is correct")
        else:
            result["feedback"].append("Adjust elbow angle, target: 90°")
        
        # Check torso angles
        if angles["left_torso"] < 120 and angles["right_torso"] < 120:
            result["feedback"].append("Torso angle is correct")
        else:
            result["feedback"].append("Adjust torso angle, target: 90°")

    def _check_downdog_pose(self, angles: dict, result: dict):
        """Check downward dog pose specific angles."""
        # Check elbow angles
        if angles["left_elbow"] > 160 and angles["right_elbow"] > 160:
            result["feedback"].append("Elbow angle is correct")
        else:
            result["feedback"].append("Adjust elbow angle, target: 180")
        
        # Check shoulder angles
        if angles["left_shoulder"] > 160 and angles["right_shoulder"] > 160:
            result["feedback"].append("Shoulder angle is correct")
        else:
            result["feedback"].append("Adjust shoulder angle, target: 180°")
        
        # Check torso angles
        if angles["left_torso"] < 100 and angles["right_torso"] < 100:
            result["feedback"].append("Torso angle is correct")
        else:
            result["feedback"].append("Adjust torso angle, target: 90")

    def _check_plank_pose(self, angles: dict, result: dict):
        """Check plank pose specific angles."""
        # Check elbow angles
        if angles["left_elbow"] > 160 and angles["right_elbow"] > 160:
            result["feedback"].append("Elbow angle is correct")
        else:
            result["feedback"].append("Adjust elbow angle, target: 180")
        
        # Check shoulder angles
        if (angles["left_shoulder"] > 60 and angles["right_shoulder"] > 60 and 
            angles["left_shoulder"] < 90 and angles["right_shoulder"] < 90):
            result["feedback"].append("Shoulder angle is correct")
        else:
            result["feedback"].append("Adjust shoulder angle, target: 75")
        
        # Check torso angles
        if angles["left_torso"] > 160 and angles["right_torso"] > 160:
            result["feedback"].append("Torso angle is correct")
        else:
            result["feedback"].append("Adjust torso angle, target: 180")
        
        # Check knee angles
        if angles["left_knee"] > 160 and angles["right_knee"] > 160:
            result["feedback"].append("Knee angle is correct")
        else:
            result["feedback"].append("Adjust knee angle, target: 180")