#!/usr/bin/env python3
# coding: utf-8

import pyrealsense2 as rs
import numpy as np
import cv2
from cv2 import aruco

# get dicionary and get parameters
dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()

# decimarion_filterのパラメータ
decimate = rs.decimation_filter()
decimate.set_option(rs.option.filter_magnitude, 1)
# spatial_filterのパラメータ
spatial = rs.spatial_filter()
spatial.set_option(rs.option.filter_magnitude, 1)
spatial.set_option(rs.option.filter_smooth_alpha, 0.25)
spatial.set_option(rs.option.filter_smooth_delta, 50)
# hole_filling_filterのパラメータ
hole_filling = rs.hole_filling_filter()
# disparity
depth_to_disparity = rs.disparity_transform(True)
disparity_to_depth = rs.disparity_transform(False)

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
            # フレームセットを待機
            frames = pipeline.wait_for_frames()

            # カラーフレームとデプスフレームを取得
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            # フレームがない場合はスキップ
            if not color_frame or not depth_frame:
                continue

            # Numpy配列に変換
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            # デプス画像をカラーマップに変換
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            corners, ids, rejectedCandidates = aruco.detectMarkers(color_image, dictionary, parameters=parameters)
            filter_frame = decimate.process(depth_frame)
            filter_frame = depth_to_disparity.process(filter_frame)
            filter_frame = spatial.process(filter_frame)
            filter_frame = disparity_to_depth.process(filter_frame)
            filter_frame = hole_filling.process(filter_frame)
            result_frame = filter_frame.as_depth_frame()
            # 距離情報の取得
            if ids is not None:
                depth_data = result_frame.get_distance(corners[0][0][0][0], corners[0][0][0][1])
                print(depth_data)
            # カラーとデプス画像を並べて表示
            aruco.drawDetectedMarkers(color_image, corners, ids)
            images = np.hstack((color_image, depth_colormap))
            cv2.imshow('RealSense', images)

            # 'q'を押してウィンドウを閉じる
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # ストリーミング停止
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()