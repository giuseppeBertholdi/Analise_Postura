import cv2
import math
import os
import time
from datetime import datetime
import mediapipe as mp
import numpy as np
try:
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False
try:
    # Optional: only used if labels are provided
    from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
    _HAS_SK = True
except Exception:
    _HAS_SK = False

# ================= FUNÇÕES =================
def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def calculate_angle(x1, y1, x2, y2):
    """
    Calcula o ângulo (em graus) entre o vetor base->alvo e o eixo vertical (para cima).
    Resulta em 0° quando perfeitamente alinhado ao vertical e aumenta conforme inclina.
    """
    dx = x2 - x1
    dy = y2 - y1
    norm = math.sqrt(dx * dx + dy * dy)
    if norm == 0:
        return 0
    # Vetor vertical para cima é (0, -1) em coordenadas de imagem
    dot = dx * 0 + dy * (-1)
    cos_theta = max(-1.0, min(1.0, dot / norm))
    theta_rad = math.acos(cos_theta)
    return int(round(180.0 / math.pi * theta_rad))

def send_warning():
    print("⚠️ Postura ruim por muito tempo!")

# ================= CLASSE DE ANÁLISE =================
class PostureAnalyzer:
    def __init__(self):
        # Mediapipe setup
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        # UI and colors
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.colors = {
            'blue': (255, 127, 0),
            'red': (50, 50, 255),
            'green': (127, 255, 0),
            'light_green': (127, 233, 100),
            'yellow': (0, 255, 255),
            'pink': (255, 0, 255),
            'white': (255, 255, 255),
            'gray': (120, 120, 120)
        }
        # Thresholds (configuráveis)
        self.neck_good_threshold_deg = 40
        self.torso_good_threshold_deg = 10
        self.align_threshold_px = 100
        # Frame counters
        self.good_frames = 0
        self.bad_frames = 0
        self.aligned_frames = 0
        self.total_frames = 0
        # Streaks
        self.current_good_streak = 0
        self.current_bad_streak = 0
        self.longest_good_streak = 0
        self.longest_bad_streak = 0
        # Time / FPS
        self.session_start_ts = time.time()
        self._last_time = time.perf_counter()
        self.fps = 0.0
        # Angles and signals
        self.neck_angles = []
        self.torso_angles = []
        self.offsets = []
        self.posture_scores = []  # 0..1
        self.predictions = []     # 1=boa postura, 0=ruim
        self.labels = []          # -1=sem rótulo, 0=ruim, 1=boa
        # Report path
        now = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.report_dir = os.path.join(os.getcwd(), 'reports', f'posture_session_{now}')
        os.makedirs(self.report_dir, exist_ok=True)

    def process_frame(self, frame):
        self.total_frames += 1
        # FPS update
        now_t = time.perf_counter()
        dt = now_t - self._last_time
        if dt > 0:
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt) if self.fps > 0 else (1.0 / dt)
        self._last_time = now_t

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

            score = self._compute_posture_score(neck_inclination, torso_inclination, offset)
            is_good = 1 if score >= 0.7 else 0

            self.neck_angles.append(neck_inclination)
            self.torso_angles.append(torso_inclination)
            self.offsets.append(offset)
            self.posture_scores.append(score)
            self.predictions.append(is_good)
            self.labels.append(-1)  # placeholder, pode ser atualizado via teclado

            # Contadores
            if is_good:
                self.good_frames += 1
                self.current_good_streak += 1
                self.current_bad_streak = 0
                self.longest_good_streak = max(self.longest_good_streak, self.current_good_streak)
            else:
                self.bad_frames += 1
                self.current_bad_streak += 1
                self.current_good_streak = 0
                self.longest_bad_streak = max(self.longest_bad_streak, self.current_bad_streak)
            if offset < self.align_threshold_px:
                self.aligned_frames += 1

            # Desenhar linhas e ângulos
            color_posture = self.colors['light_green'] if is_good else self.colors['red']
            self.draw_posture_lines(frame, l_shldr, r_shldr, l_ear, l_hip, color_posture)
            self.draw_posture_angles(frame, l_shldr, l_hip, neck_inclination, torso_inclination, color_posture)

        # Overlay analytics
        self.display_analytics(frame)
        return frame

    def _compute_posture_score(self, neck_deg, torso_deg, offset):
        """
        Score 0..1: 1 excelente; considera pescoço, tronco e alinhamento de ombros.
        Penaliza excedentes sobre thresholds, com maior peso no pescoço.
        """
        neck_penalty = max(0.0, (neck_deg - self.neck_good_threshold_deg) / 50.0)
        torso_penalty = max(0.0, (torso_deg - self.torso_good_threshold_deg) / 30.0)
        misalign_penalty = 0.1 if offset >= self.align_threshold_px else 0.0
        combined_penalty = min(1.0, 0.6 * neck_penalty + 0.4 * torso_penalty + misalign_penalty)
        return float(max(0.0, 1.0 - combined_penalty))

    def draw_posture_lines(self, frame, l_shldr, r_shldr, l_ear, l_hip, color):
        cv2.line(frame, l_shldr, l_ear, color, 4)
        cv2.line(frame, l_hip, l_shldr, color, 4)
        for point in [l_shldr, r_shldr, l_ear, l_hip]:
            cv2.circle(frame, point, 7, self.colors['yellow'], -1)

    def draw_posture_angles(self, frame, l_shldr, l_hip, neck, torso, color):
        self._put_text_with_bg(frame, f'Neck: {neck}°  Torso: {torso}°', (10, 30), color, 0.7)

    def display_analytics(self, frame):
        # Painel
        panel_w, panel_h = 360, 150
        x0, y0 = 10, 10
        overlay = frame.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (0, 0, 0), -1)
        frame[:] = cv2.addWeighted(overlay, 0.25, frame, 0.75, 0)

        avg_neck = self.get_avg_neck_angle()
        avg_torso = self.get_avg_torso_angle()
        align_pct = self.get_alignment_percentage()
        good_pct = self.get_good_percentage()
        score = self.posture_scores[-1] if self.posture_scores else 0.0
        status = 'GOOD' if (self.predictions[-1] if self.predictions else 0) == 1 else 'BAD'
        status_color = self.colors['light_green'] if status == 'GOOD' else self.colors['red']
        elapsed = max(0.0, time.time() - self.session_start_ts)

        # Linhas de texto
        self._put_text_with_bg(frame, f'Status: {status}', (x0 + 10, y0 + 25), status_color, 0.7)
        self._put_text_with_bg(frame, f'Avg Neck: {avg_neck:.1f}°  Avg Torso: {avg_torso:.1f}°', (x0 + 10, y0 + 55), self.colors['yellow'], 0.6)
        self._put_text_with_bg(frame, f'Good%: {good_pct:.1f}%  Align%: {align_pct:.1f}%', (x0 + 10, y0 + 80), self.colors['yellow'], 0.6)
        self._put_text_with_bg(frame, f'FPS: {self.fps:.1f}  Time: {int(elapsed)}s', (x0 + 10, y0 + 105), self.colors['yellow'], 0.6)
        self._put_text_with_bg(frame, f'Streak G/B: {self.current_good_streak}/{self.current_bad_streak}', (x0 + 10, y0 + 130), self.colors['yellow'], 0.6)

        # Barras de progresso
        self._draw_progress_bar(frame, (x0 + 220, y0 + 20), (120, 14), score, self.colors['green'], self.colors['gray'], label='Score')
        self._draw_progress_bar(frame, (x0 + 220, y0 + 45), (120, 14), good_pct / 100.0, self.colors['green'], self.colors['gray'], label='Good%')
        self._draw_progress_bar(frame, (x0 + 220, y0 + 70), (120, 14), align_pct / 100.0, self.colors['green'], self.colors['gray'], label='Align%')

    def get_avg_neck_angle(self):
        return np.mean(self.neck_angles) if self.neck_angles else 0
    def get_avg_torso_angle(self):
        return np.mean(self.torso_angles) if self.torso_angles else 0
    def get_alignment_percentage(self):
        return (self.aligned_frames / self.total_frames) * 100 if self.total_frames > 0 else 0
    def get_good_percentage(self):
        return (self.good_frames / self.total_frames) * 100 if self.total_frames > 0 else 0

    def _put_text_with_bg(self, img, text, org, color, scale):
        thickness = 2
        (tw, th), _ = cv2.getTextSize(text, self.font, scale, thickness)
        x, y = org
        cv2.rectangle(img, (x - 3, y - th - 6), (x + tw + 3, y + 4), (0, 0, 0), -1)
        cv2.putText(img, text, org, self.font, scale, color, thickness)

    def _draw_progress_bar(self, img, origin, size, ratio, fg_color, bg_color, label=None):
        x, y = origin
        w, h = size
        ratio = float(max(0.0, min(1.0, ratio)))
        cv2.rectangle(img, (x, y), (x + w, y + h), bg_color, -1)
        cv2.rectangle(img, (x, y), (x + int(w * ratio), y + h), fg_color, -1)
        cv2.rectangle(img, (x, y), (x + w, y + h), (30, 30, 30), 1)
        if label:
            self._put_text_with_bg(img, f'{label}', (x, y - 5), self.colors['white'], 0.5)

    def register_label_for_last_frame(self, label_value):
        """
        label_value: 1 (boa), 0 (ruim), -1 (sem rótulo)
        """
        if not self.labels:
            return
        self.labels[-1] = int(label_value)

    def finalize_and_save_report(self):
        if self.total_frames == 0:
            return
        # Salvar CSV com sinais por frame
        csv_path = os.path.join(self.report_dir, 'frames.csv')
        header = 'frame,neck_deg,torso_deg,offset_px,score,pred,label\n'
        with open(csv_path, 'w') as f:
            f.write(header)
            for i in range(self.total_frames):
                neck = self.neck_angles[i] if i < len(self.neck_angles) else ''
                torso = self.torso_angles[i] if i < len(self.torso_angles) else ''
                off = self.offsets[i] if i < len(self.offsets) else ''
                sc = self.posture_scores[i] if i < len(self.posture_scores) else ''
                pr = self.predictions[i] if i < len(self.predictions) else ''
                lb = self.labels[i] if i < len(self.labels) else -1
                f.write(f'{i},{neck},{torso},{off},{sc},{pr},{lb}\n')

        # Salvar resumo JSON simples
        summary_path = os.path.join(self.report_dir, 'summary.txt')
        elapsed = max(0.0, time.time() - self.session_start_ts)
        with open(summary_path, 'w') as f:
            f.write(f'Total frames: {self.total_frames}\n')
            f.write(f'Good frames: {self.good_frames}\n')
            f.write(f'Bad frames: {self.bad_frames}\n')
            f.write(f'Good%: {self.get_good_percentage():.2f}\n')
            f.write(f'Alignment%: {self.get_alignment_percentage():.2f}\n')
            f.write(f'Avg neck: {self.get_avg_neck_angle():.2f}\n')
            f.write(f'Avg torso: {self.get_avg_torso_angle():.2f}\n')
            f.write(f'Longest good streak: {self.longest_good_streak}\n')
            f.write(f'Longest bad streak: {self.longest_bad_streak}\n')
            f.write(f'Elapsed(s): {int(elapsed)}\n')

        # Visualizações
        if _HAS_MPL:
            try:
                self._save_plots()
            except Exception:
                pass

        # Métricas supervisionadas (se houver rótulos)
        if any(lb in (0, 1) for lb in self.labels):
            y_true = np.array([lb if lb in (0, 1) else -1 for lb in self.labels])
            mask = y_true >= 0
            if np.any(mask):
                y_true = y_true[mask]
                y_pred = np.array(self.predictions)[mask]
                scores = np.array(self.posture_scores)[mask]
                cm = self._confusion_matrix(y_true, y_pred)
                metrics_txt = self._compute_supervised_metrics(y_true, y_pred, scores)
                with open(os.path.join(self.report_dir, 'supervised_metrics.txt'), 'w') as f:
                    f.write(metrics_txt)
                if _HAS_MPL:
                    try:
                        self._save_confusion_matrix(cm)
                    except Exception:
                        pass

    def _save_plots(self):
        # Time series
        fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        ax[0].plot(self.neck_angles, label='Neck (deg)')
        ax[0].axhline(self.neck_good_threshold_deg, color='g', linestyle='--', alpha=0.6)
        ax[0].legend(); ax[0].grid(True)
        ax[1].plot(self.torso_angles, label='Torso (deg)', color='tab:orange')
        ax[1].axhline(self.torso_good_threshold_deg, color='g', linestyle='--', alpha=0.6)
        ax[1].legend(); ax[1].grid(True)
        ax[2].plot(self.posture_scores, label='Score (0..1)', color='tab:green')
        ax[2].legend(); ax[2].grid(True)
        ax[2].set_xlabel('Frame')
        fig.tight_layout()
        fig.savefig(os.path.join(self.report_dir, 'timeseries.png'), dpi=150)
        plt.close(fig)

        # Histograms
        fig, ax = plt.subplots(1, 3, figsize=(12, 3))
        ax[0].hist(self.neck_angles, bins=30, color='tab:blue', alpha=0.8)
        ax[0].axvline(self.neck_good_threshold_deg, color='r', linestyle='--'); ax[0].set_title('Neck (deg)')
        ax[1].hist(self.torso_angles, bins=30, color='tab:orange', alpha=0.8)
        ax[1].axvline(self.torso_good_threshold_deg, color='r', linestyle='--'); ax[1].set_title('Torso (deg)')
        ax[2].hist(self.posture_scores, bins=30, color='tab:green', alpha=0.8)
        ax[2].axvline(0.7, color='r', linestyle='--'); ax[2].set_title('Score (0..1)')
        fig.tight_layout()
        fig.savefig(os.path.join(self.report_dir, 'histograms.png'), dpi=150)
        plt.close(fig)

    def _confusion_matrix(self, y_true, y_pred):
        if _HAS_SK:
            return confusion_matrix(y_true, y_pred)
        # Manual: rows true=[0,1], cols pred=[0,1]
        cm = np.zeros((2, 2), dtype=int)
        for t, p in zip(y_true, y_pred):
            cm[int(t), int(p)] += 1
        return cm

    def _compute_supervised_metrics(self, y_true, y_pred, scores):
        if _HAS_SK:
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            try:
                auc = roc_auc_score(y_true, scores)
            except Exception:
                auc = float('nan')
            try:
                ap = average_precision_score(y_true, scores)
            except Exception:
                ap = float('nan')
        else:
            # Básico sem sklearn
            tp = int(np.sum((y_true == 1) & (y_pred == 1)))
            fp = int(np.sum((y_true == 0) & (y_pred == 1)))
            fn = int(np.sum((y_true == 1) & (y_pred == 0)))
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            auc = float('nan')
            ap = float('nan')
        lines = [
            f'Precision: {prec:.4f}',
            f'Recall:    {rec:.4f}',
            f'F1-score:  {f1:.4f}',
            f'ROC AUC:   {auc}',
            f'PR AUC:    {ap}'
        ]
        return '\n'.join(lines) + '\n'

    def _save_confusion_matrix(self, cm):
        fig, ax = plt.subplots(figsize=(3.2, 3.2))
        im = ax.imshow(cm, cmap='Blues')
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(['Pred 0', 'Pred 1'])
        ax.set_yticklabels(['True 0', 'True 1'])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(int(cm[i, j])), ha='center', va='center', color='black')
        ax.set_title('Confusion Matrix')
        fig.tight_layout()
        fig.savefig(os.path.join(self.report_dir, 'confusion_matrix.png'), dpi=150)
        plt.close(fig)

# ================= LOOP PRINCIPAL =================
def main():
    cap = cv2.VideoCapture(0)
    analyzer = PostureAnalyzer()
    paused = False

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            processed_frame = analyzer.process_frame(frame)
            cv2.imshow("Posture Analysis", processed_frame)
        else:
            # Mostrar último frame com indicação de pausa
            if 'processed_frame' in locals():
                temp = processed_frame.copy()
                analyzer._put_text_with_bg(temp, 'PAUSED - press p to resume', (20, 60), analyzer.colors['yellow'], 0.8)
                cv2.imshow("Posture Analysis", temp)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = not paused
        elif key == ord('g'):
            analyzer.register_label_for_last_frame(1)
        elif key == ord('b'):
            analyzer.register_label_for_last_frame(0)
        elif key == ord('u'):
            analyzer.register_label_for_last_frame(-1)

    # Finalização e relatórios
    analyzer.finalize_and_save_report()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
