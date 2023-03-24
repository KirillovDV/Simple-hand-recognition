import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = hands.process(image)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Получение координат ключевых точек руки
                thumb = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                middle = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                ring = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
                pinky = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
                wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]

                # Определение текущей позы руки
                if thumb.y < index.y and thumb.y < middle.y and thumb.y < ring.y and thumb.y < pinky.y:
                    if wrist.x < thumb.x and wrist.x < index.x and wrist.x < middle.x and wrist.x < ring.x and wrist.x < pinky.x:
                        print("Кулак")
                # Определение текущей позы руки
                elif hand_landmarks.landmark[8].y < hand_landmarks.landmark[5].y and hand_landmarks.landmark[8].y < \
                        hand_landmarks.landmark[6].y and hand_landmarks.landmark[8].y < hand_landmarks.landmark[7].y:
                    print("Раскрытая ладонь обнаружена!")

        # Display the resulting image with landmarks.
        cv2.imshow('MediaPipe Hands', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
