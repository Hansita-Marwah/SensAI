from flask import Flask,jsonify, request
import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle 
def callfun():
    cap = cv2.VideoCapture(0)
    # Curl counter variables
    counter = 0 
    stage = None
    ## Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
        
            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
      
            # Make detection
            results = pose.process(image)
    
            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark
            
                # Get coordinates
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                verticalhips = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,0]
                verticalknee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,0]
                verticalankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,0]

                # Calculate angle
                angleatback = calculate_angle(verticalhips, hip, shoulder)
                angleatknee = calculate_angle(verticalknee, knee, hip)
                angleatankle = calculate_angle(verticalankle, ankle, knee)

                cv2.putText(image, str(angleatback), tuple(np.multiply(hip, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )
                cv2.putText(image, str(angleatknee), tuple(np.multiply(knee, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                               )
                cv2.putText(image, str(angleatankle), tuple(np.multiply(ankle, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                               )
           
                if angleatback < 25 or angleatknee < 85 :
                    stage = "move down"
                if angleatback >=25 and stage == 'move down':
                    stage="move up"
                    counter +=1
            

            # Visualize angle
            
            except:
                pass
        
            # Render curl counter
            # Setup status box
            cv2.rectangle(image, (0,0), (160,73), (255,0,127), -1)
        
            # Rep data
            cv2.putText(image, 'REPS', (15,12), 
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter), 
                    (10,60), 
                    cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
            # Stage data
            cv2.putText(image, 'STAGE', (65,12), 
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, stage, 
                    (60,60), 
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
        
        
            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                 )               
        
            cv2.imshow('Mediapipe Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('s'):
                break

    cap.release()
    cv2.destroyAllWindows()

callfun()