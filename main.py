from .src.utils import process_video
import os

if __name__ == "__main__":
    # input_path = input("Enter the input video file path (e.g., .mp4, .avi, etc.): ") or CONFIG["default_input_video"]
    # if not os.path.isfile(input_path):
    #     print("File does not exist. Please check the path.")
    #     exit(1)

    input_path = '/home/dawid/projekt/MOT16-11-raw.webm'
    
    input_dir, input_filename = os.path.split(input_path)
    filename_without_extension, file_extension = os.path.splitext(input_filename)
    output_filename = f"{filename_without_extension}_output.avi"

    output_path = os.path.join(input_dir, output_filename)
    
    # Uruchomienie procesu obróbki wideo
    process_video(input_path, output_path)