

### 주의사항 : moviePy 모듈 문제로 파이썬 실행 시, 반드시 python3 composite.py로 실행할 것 ### 


### 1. Setting Library ------

import numpy as np
import pandas as pd

import cv2
from PIL import Image, ImageDraw, ImageFont

import shutil # copy 생성
from tqdm import tqdm
import time
import moviepy.editor as mp
import subprocess
import os


### 2. Load script (Time stamp, Texts, Labels) ------

textdata = pd.read_csv("./input_text/script.csv")

# Texts & Labels
Labels, Texts= [], []
for T, L in zip(textdata["Text"], textdata["Label"]) :
    Texts.append(T)
    Labels.append(L)


# Time stamp
textdataList = textdata.values.tolist()

start_times, end_times = [], []
for times in textdataList :
    start_times.append(times[0])
    end_times.append(times[1])

# Time stamp → Convert to seconds
def convert_to_seconds(time):
    h, m, s, ms = map(int, time.split(':'))
    return 3600 * h + 60 * m + s + ms / 100.0

start_times = [convert_to_seconds(time) for time in start_times]
end_times = [convert_to_seconds(time) for time in end_times]


### 3. Function to composite subtitles according to a Time stamp ------

def add_subtitles_within_time_range(text, Label, input_video_path, start_time, end_time):
    # 비디오 로드
    video = cv2.VideoCapture(input_video_path)

    # Get the dimensions of the video
    frame_width = int(video.get(3))
    frame_height = int(video.get(4))

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v") 
    out = cv2.VideoWriter("./output_video/output.mp4", fourcc, 30.0, (frame_width, frame_height))

    # Define the text font and color
    if Label == 'neutral':
        font = ImageFont.truetype("./fonts/Cafe24Classictype.ttf", 42)
        color = (255, 255, 255)
    elif Label == 'happiness':
        font = ImageFont.truetype("./fonts/HSSantokki-Regular.ttf", 70)
        color = (255, 255, 255)
    elif Label == 'angry':
        font = ImageFont.truetype("./fonts/BMEuljiro10yearslater.ttf", 70)
        color = (128, 0, 16)
    elif Label == 'sadness':
        font = ImageFont.truetype("./fonts/SF함박눈TTF.ttf", 52)
        color = (255, 255, 255)
    elif Label == 'surprise':
        font = ImageFont.truetype("./fonts/개봉박두체.ttf", 68)
        color = (254, 226, 52)    

    # Read and process each frame of the video
    start_frame = int(start_time * 30)
    end_frame = int(end_time * 30)
    current_frame = 0
    while True:
        ret, frame = video.read()
        if not ret:
            break
        if current_frame >= start_frame and current_frame <= end_frame:
            # Convert the frame to a PIL image
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            
            # Create a draw object and add text to the image
            draw = ImageDraw.Draw(frame)
            text = text
            textwidth, textheight = font.getsize(text)
            text_position = (frame_width / 2 - textwidth / 2, frame_height - textheight - 50)
            draw.text(text_position, text, font=font, fill=color)   
            
            # Convert the PIL image back to a OpenCV frame
            frame = np.array(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
        # Write the frame to the output video
        out.write(frame)
        current_frame += 1

    # Release the video and VideoWriter objects
    out.release()
    video.release()


### 4. Run 'add_subtitles_within_time_range' function ------

input_video_path = "./input_video/test2.mp4"
cnt = 0

for i in tqdm(range(len(Texts)), desc=f'{len(Texts)}개의 자막과 동영상을 합성 중 입니다.'):
    cnt += 1
    text = Texts[i]
    start_time = start_times[i]
    end_time = end_times[i]
    Label = Labels[i]
    if cnt == 1 :
        add_subtitles_within_time_range(text, Label, input_video_path, start_time, end_time)
        input_file = "./output_video/output.mp4"
        output_file = "./output_video/final.mp4"
        shutil.copy(input_file, output_file)

    else :
        input_video_path = "./output_video/final.mp4"
        add_subtitles_within_time_range(text, Label, input_video_path, start_time, end_time)
        input_file = "./output_video/output.mp4"
        output_file = "./output_video/final.mp4"
        shutil.copy(input_file, output_file)


### 5. Recomposite audio and video ------

# 원본 테스트 영상에서 오디오 추출
clip = mp.VideoFileClip("./input_video/test2.mp4")
clip.audio.write_audiofile("./input_video/audio.mp3")

# output 동영상과 오디오 재합성하여 final_output 저장
video_file = "./output_video/final.mp4"
audio_file = "./input_video/audio.mp3"
output_file = "./output_video/final_output.mp4"

command = (f"ffmpeg -i {video_file} -i {audio_file} -c:v copy -c:a aac -strict experimental {output_file}")
subprocess.call(command, shell=True)