import os
import subprocess

files = os.listdir("videos")

for file in files:
    tutorial_num = file.split(" [")[0].split(" #")[1]
    file_name = file.split("｜")[0]
    print(tutorial_num, file_name)
    subprocess.run(["ffmpeg", "-i", f"Videos/{file}", f"audios/{tutorial_num}_{file_name}.mp3"])        