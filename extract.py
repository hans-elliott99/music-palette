#!usr/bin/env python
import time
import os
from pytube import YouTube, Playlist
import moviepy.editor as mpe
import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

def download_youtube(url:str, dir="."):
    yt = YouTube(url)
    vid = yt.streams.get_lowest_resolution()
    vid.download(output_path=dir)
    return dir+"/"+vid.default_filename


def get_video_frames(video:mpe.VideoClip, start_time, length_sec:int=5):
    """Get array of frames (1 fps) from start_time to start_time + length_sec.
    """
    frames = []
    for t in range(start_time, start_time+length_sec):
        frames.append( video.get_frame(t) )
    return frames

def resize_img_arr(img:np.array, basewidth:int=300) -> np.array:
  """Resize image array, maintain aspect ratio 
  """
  wperc  = basewidth / img.shape[0]
  height = int( img.shape[1]*wperc )
  return cv2.resize(img, (height, basewidth), cv2.INTER_AREA)

def resize_frames(frames:list, basewidth:int=300):
    return [resize_img_arr(ar, basewidth=basewidth) for ar in frames]


def save_audio_clip(video:mpe.VideoClip, save_name, start_time, length_sec:int=5) -> None:
    """Save audio clip as WAV file.
    length_sec determines the length of each audio clip, and should match get_video_frames.
    """
    clip = video.subclip(start_time, start_time+length_sec)
    clip.audio.write_audiofile(save_name, verbose=False, logger=None)

def get_clip_centroids(frames:list, n_clusters:int=5):
    """Get fitted KMeans centroids across all frames in the clip.
    """
    stack = np.vstack(frames)
    stack = np.reshape(stack, (stack.shape[0]*stack.shape[1], 3))
    kmeans = KMeans(n_clusters=n_clusters, random_state=123).fit(stack)
    return kmeans.cluster_centers_


def video_to_data(video:mpe.VideoFileClip, video_id:int, audio_path:str, n_clusters:int=5, clip_length:int=5, verbose:int=0):
    """Convert one video into data.
    
    The video is broken into `clip_length` second clips, from which a frame-per-second is extracted.
    Then `n_clusters` RGB colors are extracted across all the frames from each clip, using KMeans.
    The `clip_length` seconds of audio is saved as a .WAV file.
    Returns a dataframe which contains, for each audio clip, the path to the audio clip and the RGB values
    of the `n_clusters` colors which were extracted.
    """
    st = time.time()
    output = {"audio_clip" : []}
    for i in range(n_clusters):
        output[f'rgb_clust_{i}'] = []

    clipn = 0
    while True:
        start_t = (clipn+1)*clip_length ##dont start at 0sec, since we tend to have fade in from black

        if start_t + clip_length > video.duration:
            break

        # SAVE AUDIO
        audio_name = f"{video_id}_audioclip_{clipn}.wav"
        save_audio_clip(video, audio_path+"/"+audio_name, start_t, clip_length)

        # MAKE COLOR PALETTE
        clip_frames = get_video_frames(video, start_t, clip_length)
        clip_frames = resize_frames(clip_frames, basewidth=100)
        clusters    = get_clip_centroids(clip_frames, n_clusters=n_clusters)

        # SAVE METADATA
        output['audio_clip'].append(audio_name)
        for i in range(n_clusters):
            output[f'rgb_clust_{i}'].append( ' '.join(clusters[i].astype(int).astype(str).tolist()) )


        if verbose>1: print(f"clip {clipn} done. {(start_t/video.duration)*100 :.2f}%")
        clipn += 1

    if verbose: 
        et = time.time()-st
        print(f"Video {video_id}: duration={video.duration :.2f}s. et={et :.2f}s",
              f"({video.duration / et :.1f} video-secs processed per sec)")

    return pd.DataFrame(output)



if __name__=="__main__":

    clip_length = 5 ##seconds
    n_clusters  = 5 ##rgb colors per palette
    temp_path   = "./temp"
    audio_path  = "./audio_wav"
    meta_path   = "."
    playlist    = Playlist("https://www.youtube.com/playlist?list=PLMC9KNkIncKtGvr2kFRuXBVmBev6cAJ2u")
    merge_all   = True

    # create/load playlist metadata to track completed videos and maintain video ids
    if os.path.isfile(f"{meta_path}/playlist_metadata.csv"):
        playlist_df = pd.read_csv(f"{meta_path}/playlist_metadata.csv")
    else:
        playlist_df = pd.DataFrame({
            "url"       : playlist.video_urls,
            "id"        : range(0, len(playlist.videos)),
            "completed" : [0 for _ in range(len(playlist.videos))],
            "failed"    : [0 for _ in range(len(playlist.videos))]
        })
        playlist_df.to_csv(f"{meta_path}/playlist_metadata.csv", index = False)


    # Begin extraction process
    for i, df_row in playlist_df.iterrows():
        
        if df_row['completed'] == 1:
            continue
        vid_id = df_row['id']
        url    = df_row['url']

        vid_file = ""
        video    = None
        try:
            vid_file = download_youtube(url, dir=temp_path)
            video = mpe.VideoFileClip(vid_file)

            output = video_to_data(
                video, 
                video_id=vid_id, 
                audio_path=audio_path,
                n_clusters=n_clusters, 
                clip_length=clip_length,
                verbose=1
                )
            # save output to folder, to be combined at end of playlist processing
            output.to_csv(f"{temp_path}/{vid_id}_md.csv", index=False)
            playlist_df.loc[i, 'completed'] = 1

            # close video and delete file
            video.close()
            os.remove(vid_file)
        
        except Exception as e:
            print(f"iter {i} exception:", e)
            playlist_df.loc[i, 'failed'] = 1

            if video: video.close()
            if os.path.isfile(vid_file): os.remove(vid_file)

    # save updated metadata            
    playlist_df.to_csv(f"{meta_path}/playlist_metadata.csv", index = False)


    # merge all output dataframes into one final dataset
    if merge_all:
        dfs = []
        for file in os.listdir(temp_path):
            if not file.endswith("_md.csv"):
                continue
            dfs.append( pd.read_csv(f"{temp_path}/{file}") )

        final = pd.concat(dfs)
        final.to_csv("audio_color_mapping.csv", index=False)

