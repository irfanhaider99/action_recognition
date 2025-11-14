import cv2
import time
from pypylon import pylon
from ultralytics import YOLO

# === Load YOLOv11 Pretrained Model ===
model = YOLO("yolo11n.pt")  

# === Camera Serial Numbers  ===
SERIAL_LEFT = "40312157"   # replace with left camera serial
SERIAL_RIGHT = "40312158"  # replace with right camera serial

# === Initialize Basler Cameras by Serial Number ===
factory = pylon.TlFactory.GetInstance()
devices = factory.EnumerateDevices()
serial_map = {dev.GetSerialNumber(): dev for dev in devices}

if SERIAL_LEFT not in serial_map or SERIAL_RIGHT not in serial_map:
    raise RuntimeError("One or both Basler cameras not found.")

camL = pylon.InstantCamera(factory.CreateDevice(serial_map[SERIAL_LEFT]))
camR = pylon.InstantCamera(factory.CreateDevice(serial_map[SERIAL_RIGHT]))

camL.Open()
camR.Open()

camL.StartGrabbing()
camR.StartGrabbing()

converter = pylon.ImageFormatConverter()
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

#cv2.namedWindow("Left Camera", cv2.WINDOW_NORMAL)
#cv2.namedWindow("Right Camera", cv2.WINDOW_NORMAL)

# === Person Detection Function ===
def detect_person(yolo_result):
    return any(int(cls) == 0 for cls in yolo_result.boxes.cls.tolist()) if yolo_result.boxes is not None else False

# === Main Loop ===
print("Human presence detection. Press ESC to quit.")
while camL.IsGrabbing() and camR.IsGrabbing():
    grabL = camL.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
    grabR = camR.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

    if grabL.GrabSucceeded() and grabR.GrabSucceeded():
        imgL = converter.Convert(grabL).GetArray()
        imgR = converter.Convert(grabR).GetArray()

        # Run YOLOv11 on both images
        resL = model(imgL, verbose=False)[0]
        resR = model(imgR, verbose=False)[0]

        personL = detect_person(resL)
        personR = detect_person(resR)

        if personL and personR:
            status = "HUMAN AVAILABLE"
            color = (0, 255, 0)
        else:
            status = "NO HUMAN"
            color = (0, 0, 255)

        # Draw result text
        cv2.putText(imgL, status, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(imgR, status, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Optionally draw boxes (if needed)
        for box in resL.boxes:
            if int(box.cls) == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(imgL, (x1, y1), (x2, y2), color, 2)

        for box in resR.boxes:
            if int(box.cls) == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(imgR, (x1, y1), (x2, y2), color, 2)

        # Show windows
        # Resize for display
        resizedL = cv2.resize(imgL, (960, 600))
        resizedR = cv2.resize(imgR, (960, 600))

        cv2.imshow("Left Camera", resizedL)
        cv2.imshow("Right Camera", resizedR)

        key = cv2.waitKey(1)
        if key == 27:  # ESC to quit
            break

    grabL.Release()
    grabR.Release()

# === Cleanup ===
camL.StopGrabbing()
camR.StopGrabbing()
camL.Close()
camR.Close()
cv2.destroyAllWindows()  

