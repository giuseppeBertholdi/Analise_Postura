import cv2
import math
import mediapipe as mp
import numpy as np

# ================= FUNÇÕES =================
def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def calculate_angle(x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    denom = math.sqrt(dx**2 + dy**2) * abs(y1)
    if denom == 0:
        return 0
    theta = math.acos(dy * -y1 / denom)
    return int(180 / math.pi * theta)

def send_warning():
    print("⚠️ Postura ruim por muito tempo!")

# ================= CLASSE DE ANÁLISE =================
class PostureAnalyzer:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.colors = {
            'blue': (255, 127, 0),
            'red': (50, 50, 255),
            'green': (127, 255, 0),
            'light_green': (127, 233, 100),
            'yellow': (0, 255, 255),
            'pink': (255, 0, 255)
        }
        self.good_frames = 0
        self.bad_frames = 0
        self.aligned_frames = 0
        self.total_frames = 0
        self.neck_angles = []
        self.torso_angles = []

    def process_frame(self, frame):
        self.total_frames += 1
        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            l_shldr = (int(landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].x * w),
                       int(landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y * h))
            r_shldr = (int(landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].x * w),
                       int(landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y * h))
            l_ear = (int(landmarks[self.mp_pose.PoseLandmark.LEFT_EAR].x * w),
                     int(landmarks[self.mp_pose.PoseLandmark.LEFT_EAR].y * h))
            l_hip = (int(landmarks[self.mp_pose.PoseLandmark.LEFT_HIP].x * w),
                     int(landmarks[self.mp_pose.PoseLandmark.LEFT_HIP].y * h))
            
            offset = calculate_distance(*l_shldr, *r_shldr)
            neck_inclination = calculate_angle(*l_shldr, *l_ear)
            torso_inclination = calculate_angle(*l_hip, *l_shldr)
            
            self.neck_angles.append(neck_inclination)
            self.torso_angles.append(torso_inclination)
            
            color_align = self.colors['green'] if offset < 100 else self.colors['red']
            color_posture = self.colors['light_green'] if neck_inclination < 40 and torso_inclination < 10 else self.colors['red']
            if color_posture == self.colors['light_green']:
                self.good_frames += 1
            else:
                self.bad_frames += 1
            if offset < 100:
                self.aligned_frames += 1

            # Desenhar linhas e ângulos
            self.draw_posture_lines(frame, l_shldr, r_shldr, l_ear, l_hip, color_posture)
            self.draw_posture_angles(frame, l_shldr, l_hip, neck_inclination, torso_inclination, color_posture)
        
        # Overlay analytics
        self.display_analytics(frame)
        return frame

    def draw_posture_lines(self, frame, l_shldr, r_shldr, l_ear, l_hip, color):
        cv2.line(frame, l_shldr, l_ear, color, 4)
        cv2.line(frame, l_hip, l_shldr, color, 4)
        for point in [l_shldr, r_shldr, l_ear, l_hip]:
            cv2.circle(frame, point, 7, self.colors['yellow'], -1)

    def draw_posture_angles(self, frame, l_shldr, l_hip, neck, torso, color):
        cv2.putText(frame, f'Neck: {neck} Torso: {torso}', (10, 30), self.font, 0.7, color, 2)

    def display_analytics(self, frame):
        analytics = np.zeros((120, 300, 3), dtype=np.uint8)
        cv2.putText(analytics, f"Neck Angle: {self.get_avg_neck_angle():.1f}", (10, 30), self.font, 0.7, self.colors['yellow'], 2)
        cv2.putText(analytics, f"Torso Angle: {self.get_avg_torso_angle():.1f}", (10, 60), self.font, 0.7, self.colors['yellow'], 2)
        cv2.putText(analytics, f"Alignment: {self.get_alignment_percentage():.1f}%", (10, 90), self.font, 0.7, self.colors['yellow'], 2)
        frame[10:130, 10:310] = analytics

    def get_avg_neck_angle(self):
        return np.mean(self.neck_angles) if self.neck_angles else 0
    def get_avg_torso_angle(self):
        return np.mean(self.torso_angles) if self.torso_angles else 0
    def get_alignment_percentage(self):
        return (self.aligned_frames / self.total_frames) * 100 if self.total_frames > 0 else 0

# ================= LOOP PRINCIPAL =================
def main():
    cap = cv2.VideoCapture(0)  # webcam
    analyzer = PostureAnalyzer()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = analyzer.process_frame(frame)
        cv2.imshow("Posture Analysis", processed_frame)

        # Pressione 'q' para sair
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
