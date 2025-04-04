import pyrealsense2 as rs

align_to = rs.stream.color
align = rs.align(align_to)

def rs_align(frame):
    aligned_frames = align.process(frame)
    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()
    return depth_frame, color_frame