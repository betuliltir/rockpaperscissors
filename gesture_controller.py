import cv2
import mediapipe as mp
import asyncio
import websockets
import json
import threading
import logging
import random
import time

logging.basicConfig(level=logging.INFO)

class RPSGestureController:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.websocket = None
        self.game_active = False
        self.current_gesture = None
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.start_websocket_server()
        self.last_hand_position = None
        self.last_scroll_y = None
        
        # Gesture stabilization için değişkenler
        self.gesture_history = []
        self.gesture_history_size = 3
        self.last_stable_gesture = None
        self.last_gesture_time = time.time()
        self.gesture_cooldown = 0.5

        # Yeni oyun başlatma bayrağı
        self.game_started = False

    def calculate_distance(self, p1, p2):
        return ((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2) ** 0.5

    def detect_pinch(self, hand_landmarks):
        thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
        index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        distance = self.calculate_distance(thumb_tip, index_tip)
        return distance < 0.05

    def detect_two_finger_scroll(self, hand_landmarks):
        index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        index_pip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_PIP]
        middle_pip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
        
        index_extended = index_tip.y < index_pip.y
        middle_extended = middle_tip.y < middle_pip.y
        
        if index_extended and middle_extended:
            avg_y = (index_tip.y + middle_tip.y) / 2
            
            if self.last_scroll_y is not None:
                movement_threshold = 0.01
                y_difference = avg_y - self.last_scroll_y
                
                if abs(y_difference) > movement_threshold:
                    scroll_direction = 'up' if y_difference < 0 else 'down'
                    self.last_scroll_y = avg_y
                    return scroll_direction
            
            self.last_scroll_y = avg_y
            return None
        
        self.last_scroll_y = None
        return None

    def detect_gesture(self, hand_landmarks):
        fingers_extended = []
        
        # Başparmak kontrolü - geliştirilmiş 3B analiz
        thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
        thumb_ip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_IP]
        thumb_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_MCP]
        
        thumb_extended = (thumb_tip.x - thumb_ip.x) ** 2 + (thumb_tip.y - thumb_ip.y) ** 2 > \
                        (thumb_ip.x - thumb_mcp.x) ** 2 + (thumb_ip.y - thumb_mcp.y) ** 2
        fingers_extended.append(thumb_extended)

        # Diğer parmakların kontrolü - geliştirilmiş hassasiyet
        finger_tips = [
            self.mp_hands.HandLandmark.INDEX_FINGER_TIP,
            self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
            self.mp_hands.HandLandmark.RING_FINGER_TIP,
            self.mp_hands.HandLandmark.PINKY_TIP
        ]
        finger_pips = [
            self.mp_hands.HandLandmark.INDEX_FINGER_PIP,
            self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
            self.mp_hands.HandLandmark.RING_FINGER_PIP,
            self.mp_hands.HandLandmark.PINKY_PIP
        ]
        finger_mcps = [
            self.mp_hands.HandLandmark.INDEX_FINGER_MCP,
            self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP,
            self.mp_hands.HandLandmark.RING_FINGER_MCP,
            self.mp_hands.HandLandmark.PINKY_MCP
        ]

        for tip, pip, mcp in zip(finger_tips, finger_pips, finger_mcps):
            tip_landmark = hand_landmarks.landmark[tip]
            pip_landmark = hand_landmarks.landmark[pip]
            mcp_landmark = hand_landmarks.landmark[mcp]
            
            extended = (tip_landmark.y < pip_landmark.y) and (pip_landmark.y < mcp_landmark.y)
            fingers_extended.append(extended)

        # Hareket kararlılığı kontrolü
        current_time = time.time()
        if current_time - self.last_gesture_time < self.gesture_cooldown:
            return self.last_stable_gesture or "Waiting..."

        extended_count = sum(fingers_extended)
        
        # Hareket tespiti
        if extended_count <= 1:
            raw_gesture = "rock"
        elif extended_count >= 4:
            raw_gesture = "paper"
        elif fingers_extended[1] and fingers_extended[2] and not fingers_extended[3] and not fingers_extended[4]:
            raw_gesture = "scissors"
        else:
            raw_gesture = "Waiting..."

        # Hareket geçmişi güncelleme ve stabilizasyon
        self.gesture_history.append(raw_gesture)
        if len(self.gesture_history) > self.gesture_history_size:
            self.gesture_history.pop(0)

        if len(self.gesture_history) == self.gesture_history_size:
            most_common = max(set(self.gesture_history), key=self.gesture_history.count)
            if self.gesture_history.count(most_common) >= self.gesture_history_size * 0.6:
                if most_common != self.last_stable_gesture:
                    self.last_stable_gesture = most_common
                    self.last_gesture_time = current_time
                return most_common

        return self.last_stable_gesture or "Waiting..."

    def get_hand_position(self, hand_landmarks):
        index_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        is_pinching = self.detect_pinch(hand_landmarks)
        
        return {
            'x': index_finger_tip.x,
            'web_x': 1.0 - index_finger_tip.x,
            'y': index_finger_tip.y,
            'is_clicking': is_pinching
        }



    controller = RPSGestureController()
    controller.start()
