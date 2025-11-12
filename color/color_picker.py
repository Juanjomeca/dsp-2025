import cv2
import numpy as np


WINDOW_CAMERA = 'Camera'
WINDOW_SELECTED = 'Selected Color'


def get_tolerances():
    h = cv2.getTrackbarPos('H_tol', WINDOW_CAMERA)
    s = cv2.getTrackbarPos('S_tol', WINDOW_CAMERA)
    v = cv2.getTrackbarPos('V_tol', WINDOW_CAMERA)
    return int(h), int(s), int(v)


class ColorPicker:
    def __init__(self, cam_index=0):
        self.cap = cv2.VideoCapture(cam_index)
        if not self.cap.isOpened():
            raise RuntimeError(f'No se pudo abrir la cámara index={cam_index}')
        self.selected_hsv = None
        self.selected_bgr = None
        self.last_frame = None

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and self.last_frame is not None:
            frame = self.last_frame
            h_img, w_img = frame.shape[:2]
            if x < 0 or y < 0 or x >= w_img or y >= h_img:
                return
            bgr = frame[y, x].astype(np.uint8)
            hsv = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0, 0]
            self.selected_hsv = (int(hsv[0]), int(hsv[1]), int(hsv[2]))
            self.selected_bgr = (int(bgr[0]), int(bgr[1]), int(bgr[2]))
            print(f'Seleccionado BGR={self.selected_bgr}  HSV={self.selected_hsv}  en ({x},{y})')

    def make_mask(self, hsv_frame, tol_h, tol_s, tol_v):
        h, s, v = self.selected_hsv
        low_h = (h - tol_h) % 180
        upper_h = (h + tol_h) % 180
        low_saturation = max(0, s - tol_s); upper_saturation = min(255, s + tol_s)
        low_value = max(0, v - tol_v); upper_value = min(255, v + tol_v)

        if low_h <= upper_h:
            mask = cv2.inRange(hsv_frame, (low_h, low_saturation, low_value), (upper_h, upper_saturation, upper_value))
        else:
            m1 = cv2.inRange(hsv_frame, (0, low_saturation, low_value), (upper_h, upper_saturation, upper_value))
            m2 = cv2.inRange(hsv_frame, (low_h, low_saturation, low_value), (179, upper_saturation, upper_value))
            mask = cv2.bitwise_or(m1, m2)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    def run(self):
        cv2.namedWindow(WINDOW_CAMERA, cv2.WINDOW_NORMAL)
        cv2.createTrackbar('H_tol', WINDOW_CAMERA, 10, 90, lambda x: None)
        cv2.createTrackbar('S_tol', WINDOW_CAMERA, 60, 255, lambda x: None)
        cv2.createTrackbar('V_tol', WINDOW_CAMERA, 60, 255, lambda x: None)
        cv2.setMouseCallback(WINDOW_CAMERA, self.mouse_callback)

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print('No se pudo leer frame de la cámara. Saliendo.')
                break

            self.last_frame = frame

            display = frame.copy()

            if self.selected_hsv is not None:
                tol_h, tol_s, tol_v = get_tolerances()
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                mask = self.make_mask(hsv, tol_h, tol_s, tol_v)
                result = cv2.bitwise_and(frame, frame, mask=mask)
                cv2.imshow(WINDOW_SELECTED, result)
                cv2.rectangle(display, (10, 10), (70, 70), self.selected_bgr, -1)
                text = f'HSV={self.selected_hsv} tol=({tol_h},{tol_s},{tol_v})'
                cv2.putText(display, text, (80, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            else:
                try:
                    cv2.destroyWindow(WINDOW_SELECTED)
                except Exception:
                    pass

            cv2.imshow(WINDOW_CAMERA, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('s'):
                print('Deseleccionado. Haga click para seleccionar un nuevo color.')
                self.selected_hsv = None
                self.selected_bgr = None

        self.cap.release()
        cv2.destroyAllWindows()


def main():
    try:
        picker = ColorPicker(0)
    except RuntimeError as e:
        print(e)
        return
    picker.run()


if __name__ == '__main__':
    main()
