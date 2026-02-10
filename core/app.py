import cv2

cap = cv2.VideoCapture("http://192.0.0.4:8080/video")
while True:
    ret , frames = cap.read()
    if not ret :
        print(f"Can't read frames")
        break 
    cv2.imshow("Webcam feed",frames)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        print(f"Quittinggg.....")
        break

cap.release()
cv2.destroyAllWindows()