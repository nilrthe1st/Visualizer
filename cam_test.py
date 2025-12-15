import cv2

# open camera
cap = cv2.VideoCapture(1)

while True:
    # read frame
    ret , frame = cap.read()

    # print whether frame was read
    print(ret)

    if not ret:
        print("Ret is false, camera not working")
        break;

    #print the type of the frame
    print(type(frame))

    # print the dimensions of the frame
    print(frame.shape)
    print(frame.mean())
    print(frame.min())
    print(frame.max())

    cv2.imshow("Camera", frame)

    # waitKey
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

# close all windows
cv2.destroyAllWindows()

print(cap.isOpened())

cap.release()

