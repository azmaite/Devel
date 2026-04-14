import av
import cv2
import numpy as np
import h5py

def select_frames_manual(video_path):

    output_h5 = video_path.replace('.mkv', '.h5')

    # --- 1. Load First Frame (Greyscale) ---
    container = av.open(video_path)
    
    # Extract the very first frame
    first_packet = next(container.decode(video=0))
    # Convert to rgba64le and take first channel for greyscale as per snippet
    frame_0_raw = first_packet.to_ndarray(format='rgba64le')[..., 0]
    
    # Normalize to 8-bit for OpenCV display
    frame_0_8bit = cv2.normalize(frame_0_raw, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

    # --- 2. Select Rectangle Crop ---
    print("Select rectangle crop and press SPACE or ENTER. Press 'c' to cancel.")
    r = cv2.selectROI("Select Crop", frame_0_8bit, fromCenter=False)
    cv2.destroyWindow("Select Crop")
    x, y, w, h = int(r[0]), int(r[1]), int(r[2]), int(r[3])
    cropped_frame_0 = frame_0_8bit[y:y+h, x:x+w]

    # --- 3. Free-hand ROI Selection ---
    pts = []
    drawing = False

    def draw_roi(event, ix, iy, flags, param):
        nonlocal drawing, pts
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            pts.append((ix, iy))
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                pts.append((ix, iy))
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False

    cv2.namedWindow("Draw Free-hand ROI")
    cv2.setMouseCallback("Draw Free-hand ROI", draw_roi)

    print("Draw your free-hand ROI. Press any key when finished.")
    while True:
        img_copy = cropped_frame_0.copy()
        if len(pts) > 1:
            cv2.polylines(img_copy, [np.array(pts)], False, (255), 2)
        cv2.imshow("Draw Free-hand ROI", img_copy)
        if cv2.waitKey(1) != -1:
            break
    cv2.destroyWindow("Draw Free-hand ROI")

    # Create Binary Mask from ROI
    mask = np.zeros((h, w), dtype=np.uint8)
    if len(pts) > 0:
        cv2.fillPoly(mask, [np.array(pts)], 255)

    # --- 4. Load & Process All Frames ---
    print("Loading and masking all frames... please wait.")
    container.seek(0) # Reset stream to start
    all_frames = []
    
    for frame in container.decode(video=0):
        # Apply crop and greyscale extraction
        arr = frame.to_ndarray(format='rgba64le')
        for j in range(arr.shape[2]):
            img = arr[...,j]
            cropped = img[y:y+h, x:x+w]
            # Apply mask (pixels outside ROI become 0/black)
            masked = np.where(mask == 255, cropped, 0)
            all_frames.append(masked)
    
    container.close()
    video_data = np.array(all_frames)

    # --- 5. Interactive Frame Selection ---
    selected_indices = set()
    current_idx = 0
    total_frames = len(video_data)

    print("\n[CONTROLS]")
    print("  'd' : Next Frame")
    print("  'a' : Previous Frame")
    print("  'w' : SELECT Frame (Red Border)")
    print("  's' : DESELECT Frame")
    print("  ENTER             : Save and Exit\n")

    while True:
        # Normalize for display (since rgba64le has a high dynamic range)
        display_frame = cv2.normalize(video_data[current_idx], None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
        display_frame = cv2.cvtColor(display_frame, cv2.COLOR_GRAY2BGR)

        # Draw red border if index is in our "keep" set
        if current_idx in selected_indices:
            cv2.rectangle(display_frame, (0,0), (w-1, h-1), (0, 0, 255), 15)
            cv2.putText(display_frame, "SELECTED", (10, h-20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.putText(display_frame, f"Frame: {current_idx}/{total_frames-1}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow("Frame Selector", display_frame)
        
        # waitKey(0) waits indefinitely for a key press
        # .waitKeyEx() is better for arrows, but we use a bitmask for compatibility
        key = cv2.waitKeyEx(0)
        
        # Logic for platform-agnostic key codes
        # Standardize key (handling different OS/OpenCV versions)
        if key == 13 or key == 10: # Enter keys
            break
            
        # Navigation: Right ('d') / Left ('a')
        if key == ord('d'): 
            current_idx = min(current_idx + 1, total_frames - 1)
        elif key == ord('a'):
            current_idx = max(current_idx - 1, 0)
            
        # Selection: ('w') / Delete: ('s')
        elif key == ord('w'):
            selected_indices.add(current_idx)
        elif key == ord('s'):
            selected_indices.discard(current_idx)

    cv2.destroyAllWindows()
    

    # --- 6. Save to H5 ---
    keep_list = sorted(list(selected_indices))
    selected_frames_data = video_data[keep_list]

    # also save average of 3 frames around each selected frame (to reduce noise)
    selected_frames_smooth = []
    for idx in keep_list:
        start = max(0, idx - 1)
        end = min(len(video_data), idx + 2) # exclusive
        avg_frame = np.mean(video_data[start:end], axis=0)
        selected_frames_smooth.append(avg_frame)

    # also save min_projection image
    min_proj = np.min(video_data, axis=0)

    with h5py.File(output_h5, 'w') as f:
        f.create_dataset('crop_coords', data=np.array([x, y, w, h]))
        f.create_dataset('roi_points', data=np.array(pts))
        f.create_dataset('selected_indices', data=np.array(keep_list))
        f.create_dataset('processed_frames', data=selected_frames_data)
        f.create_dataset('smoothed_frames', data=np.array(selected_frames_smooth))
        f.create_dataset('min_projection', data=min_proj)
    
    print(f"Saved {len(keep_list)} frames to {output_h5}")

# Run usage
# select_frames('your_video.mkv')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Select frames from a video and save to H5.')
    parser.add_argument('video_path', type=str, help='Path to the input video file (e.g., .mkv)')
    args = parser.parse_args()
    
    select_frames_manual(args.video_path)