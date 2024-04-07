import cv2
import time
import mediapipe as mp

cap = cv2.VideoCapture("Resoures/2.mp4")
desired_width = 640
desired_height = 480
Ptime = 0

mpDraw = mp.solutions.drawing_utils
Mp_face_Mesh = mp.solutions.face_mesh
faceMesh = Mp_face_Mesh.FaceMesh(max_num_faces=5)
draw_spec = mpDraw.DrawingSpec(color=(0,128,0), thickness=4, circle_radius=4)


while True:
    success, img = cap.read()
    ImgRgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = faceMesh.process(ImgRgb)
    smile_detected = False  # Flag to track if smile is detected..

    if result.multi_face_landmarks:
        for facelms in result.multi_face_landmarks:
            mpDraw.draw_landmarks(img, facelms, Mp_face_Mesh.FACEMESH_CONTOURS,draw_spec,draw_spec)
            for lm in facelms.landmark:
                ih, iw, ic = img.shape
                x, y = int(lm.x * iw), int(lm.y * ih)
                print("Width:",x,"Height:",y)
                # smile detection logic here:
                if lm == facelms.landmark[10] or lm == facelms.landmark[11]:  # Lip corners landmarks indices
                    if lm.y < facelms.landmark[0].y:  # Check if lip corners are above upper lip landmark
                        smile_detected = True

    if smile_detected:
        cv2.putText(img, "Smile Detected :)", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
    else:

        cv2.putText(img, "Smile Detected :)", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

    #To Find Out FPS

    Ctime = time.time()
    FPS = 1 / (Ctime - Ptime)
    Ptime = Ctime
    cv2.putText(img, f"FPS: {int(FPS)}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 3)
    img_resized = cv2.resize(img, (desired_width, desired_height))
    cv2.imshow("Image", img_resized)
    cv2.waitKey(1)
