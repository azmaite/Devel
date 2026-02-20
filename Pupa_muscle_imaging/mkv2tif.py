import os
import numpy as np
import av
import tifffile



def mkv2tif(video_path, resize_factor, chunk_dur_min, chunk_num=1, crop=None, hz=40):
    """
    Function to make a tiff file from the raw .mkv video (video_path)
    where each frame saved includes 4 real frames (4 channels)
    Used for processing with CaImAn for ROI detection
    If video_path is empty '', it will let you pick a file using a file dialog
    The resolution of movie will be reduced by a factor of resize_factor 
    The video will be saved in chunks of chunk_dur_min minutes (to not overload memory)
    If chunk_num is 1, save a single tiff file of length chunk_dur_min minutes
    If chunk_num is 'all', save the whole video in chunks of chunk_dur_min minutes (max 100 chunks)
    (the last chunk can be shorter)
    If crop is not None, it should be a tuple of (x1, y1, w, h) and will crop the frames 
    before resizing and saving
    Return the list of tiff file paths that were saved
    """

    # if video_path is empty, open file dialog to select video
    if video_path == '':
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        video_path = filedialog.askopenfilename(title='Select video file', filetypes=[('Video files', '*.mkv')])

    # check that all variables are valid
    assert os.path.exists(video_path), f'Video path {video_path} does not exist'
    assert video_path.endswith('.mkv'), 'Video path must be in .mkv format'
    assert type(resize_factor) == int & resize_factor > 0, 'Resize factor must be and integer greater than 0'
    assert type(chunk_dur_min) == float & chunk_dur_min > 0, 'Chunk duration must be a number greater than 0'
    assert (type(chunk_num) == int & chunk_num > 0) or chunk_num == 'all', 'Chunk number must be an integer greater than 0 or "all"'
    assert crop is None or (type(crop) == tuple and len(crop) == 4), 'Crop must be None or a tuple of (x1, y1, w, h)'
    assert type(hz) == int & hz > 0, 'Hz must be an integer greater than 0'

    frames_per_frame = 4 # each frame in the video includes 4 real frames (4 channels)
    
    # set tif path (general, will add _chunk# for each chunk)
    if crop is None:
        tif_path = video_path.replace('.mkv', f'_resX{resize_factor}_{chunk_dur_min}min.tif')
        print(f'Saving {chunk_num} tiff files of length {chunk_dur_min} minutes each at:')
    else:
        tif_path = video_path.replace('.mkv', f'_resX{resize_factor}_{chunk_dur_min}min_cropped.tif')
        print(f'Saving {chunk_num} cropped tiff files of length {chunk_dur_min} minutes each at:')
    
    print(f'   {tif_path.replace(".tif", f"_chunkXX.tif")}')
    
    # initialize list of tiff file paths
    tif_paths = []

    # get the number of frames per chunk
    chunk_frames = chunk_dur_min * 60 * hz / 4 # divide by 4 cause each frame includes 4 real frames
    # get the closest multiple of frames_per_frame
    chunk_frames = int(np.round(chunk_frames / frames_per_frame) * frames_per_frame)

    # set the edges of the chunks
    if chunk_num == 'all':
        chunk_n = 100
    else:
        chunk_n = chunk_num
    chunk_edges = np.arange(chunk_frames, chunk_frames*chunk_n + 1, chunk_frames).astype(int)

    # initialize frames list    
    frames = []
    frame_n = 0

    # open video
    container = av.open(video_path)
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    
    for frame in container.decode(stream):
        
        # extract channels (each frame includes 4 real frames)
        img = frame.to_ndarray(format='rgba64le')
        for j in range(img.shape[2]):

            img_i = img[...,j]

            if crop is not None:
                x1, y1, w, h = crop
                img_cropped = img_i[y1:y1+h, x1:x1+w]
            else:
                img_cropped = img_i

            if resize_factor > 1:
                # reduce resolution by factor of resize_factor using cv2.resize
                from skimage.transform import resize
                Y, X = img_cropped.shape
                img_resized = resize(
                    img_i,
                    (Y//resize_factor, X//resize_factor),
                    order=1,              # bilinear (good tradeoff)
                    preserve_range=True,
                    anti_aliasing=True
                ).astype(np.uint16)
            else:
                img_resized = img_cropped.astype(np.uint16)

            frames.append(img_resized)

        # update frame number
        frame_n += 1

        # if we have reached the end of a chunk, save the frames as a tiff file and reset the frames list
        if frame_n in chunk_edges:

            # save frames as tif file
            frames_uint16 = np.stack([f.astype(np.uint16) for f in frames])
            save_path = tif_path.replace('.tif', f'_chunk{frame_n//chunk_frames}.tif')
            tifffile.imwrite(save_path, frames_uint16, bigtiff=True, imagej=False)
            print(f'Saved chunk # {frame_n//chunk_frames} / {chunk_num}')

            # add tiff path to list
            tif_paths.append(save_path)

            # reset frames list
            frames = []
        
        # if we have saved the desired number of chunks, break the loop
        if len(tif_paths) >= chunk_num:
            break  

    # if chunk=='all' and there are any remaining frames after the loop, save them as a final chunk  
    if chunk_num >= chunk_num and len(frames) > 0:
        frames_uint16 = np.stack([f.astype(np.uint16) for f in frames])
        save_path = tif_path.replace('.tif', f'_chunk{len(tif_paths)+1}.tif')
        tifffile.imwrite(save_path, frames_uint16, bigtiff=True, imagej=False)
        print(f'Saved final chunk # {len(tif_paths)+1}')
        tif_paths.append(save_path)

    container.close()

    return tif_paths
    