import pyrealsense2 as rs
import cv2
from cv2 import aruco
import numpy as np

from  realsensen_align import rs_align
from realsense_filter import rs_filter

dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()

def main():
    # ストリームの設定
    pipeline = rs.pipeline()
    config = rs.config()

    # カラーストリームを設定
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # デプスストリームを設定
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    # ストリーミング開始
    pipeline.start(config)

    try:
        while True:
            frames = pipeline.wait_for_frames()

            depth_frame, color_frame = rs_align(frames)

            if not color_frame or not depth_frame:
                continue

            depth_frame = rs_filter(depth_frame)
            # Numpy配列に変換
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            # デプス画像をカラーマップに変換
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            # arUcoマーカー
            corners, ids, rejectedCandidates = aruco.detectMarkers(color_image, dictionary, parameters=parameters)
            if ids is not None:
                depth_data = depth_frame.get_distance()
                print(depth_data)
            # arUcoマーカーの描画
            aruco.drawDetectedMarkers(color_image, corners, ids)

            images = color_image
            cv2.imshow("realsense", images)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()