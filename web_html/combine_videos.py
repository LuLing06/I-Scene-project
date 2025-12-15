"""
Combine multiple videos SIDE BY SIDE with text labels.
All videos play simultaneously in a horizontal row.

Usage:
    python combine_videos.py

Configure the VIDEO_CONFIG list below to specify:
- video_path: path to the video file
- label: text to display at the bottom

Output: 5 videos side by side = 5120 x 1024 pixels
"""

import os
import subprocess
import sys

# -------- CONFIG ----------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Get video folder from command line argument (default: video1)
VIDEO_FOLDER = sys.argv[1] if len(sys.argv) > 1 else "video1"

INPUT_DIR = os.path.join(SCRIPT_DIR, VIDEO_FOLDER, "bpyrenderer_output")
OUTPUT_FILE = os.path.join(SCRIPT_DIR, VIDEO_FOLDER, "bpyrenderer_output/combined_sidebyside.mp4")

# Video configuration: (filename_pattern, label)
# Maps filename patterns to display labels
# Order matters - videos will be arranged left to right
VIDEO_LABELS = {
    "PartCrafter": "PartCrafter",
    "Gen3DSR": "Gen3DSR", 
    "MIDI": "MIDI",
    "SceneGen": "SceneGen",
    "nov-04-5block": "GSLR (Ours)",
}

# Order for display (left to right)
VIDEO_ORDER = ["PartCrafter", "Gen3DSR", "MIDI", "SceneGen", "nov-04-5block"]


def get_video_config(input_dir):
    """Auto-detect RGB videos and create config based on VIDEO_LABELS."""
    from glob import glob as globfunc
    
    rgb_files = globfunc(os.path.join(input_dir, "*_rgb.mp4"))
    
    print(f"\nFound {len(rgb_files)} RGB videos in {input_dir}:")
    for f in rgb_files:
        print(f"  - {os.path.basename(f)}")
    
    config = []
    
    for pattern in VIDEO_ORDER:
        for f in rgb_files:
            filename = os.path.basename(f)
            if pattern in filename:
                label = VIDEO_LABELS.get(pattern, pattern)
                config.append((filename, label))
                break
    
    return config

# Text styling
FONT_FILE = "/System/Library/Fonts/Supplemental/Times New Roman.ttf"
FONT_SIZE = 64
FONT_COLOR = "black"
BORDER_WIDTH = 0
BORDER_COLOR = "black"
TEXT_Y_OFFSET = 80  # pixels from bottom

# Bottom padding (percentage of video height)
# 0.1 = 10% of height added at bottom
BOTTOM_PADDING_PERCENT = 0.15
PADDING_COLOR = "white"
# --------------------------


def check_ffmpeg():
    """Check if ffmpeg is available."""
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("ERROR: ffmpeg not found. Please install ffmpeg first.")
        return False


def combine_side_by_side(video_configs, input_dir, output_path):
    """
    Combine videos side by side with text labels.
    Uses ffmpeg's hstack filter with drawtext.
    """
    n = len(video_configs)
    
    # Build ffmpeg command
    cmd = ["ffmpeg"]
    
    # Add all input files
    for filename, _ in video_configs:
        input_path = os.path.join(input_dir, filename)
        cmd.extend(["-i", input_path])
    
    # Build filter complex
    # First add text to each video, then stack them horizontally
    filter_parts = []
    
    # Add text labels to each video
    for i, (_, label) in enumerate(video_configs):
        # Escape parentheses for ffmpeg
        escaped_label = label.replace("(", "\\(").replace(")", "\\)")
        drawtext = (
            f"[{i}:v]drawtext=text='{escaped_label}':"
            f"fontfile='{FONT_FILE}':"
            f"fontsize={FONT_SIZE}:"
            f"fontcolor={FONT_COLOR}:"
            f"borderw={BORDER_WIDTH}:"
            f"bordercolor={BORDER_COLOR}:"
            f"x=(w-text_w)/2:"
            f"y=h-{TEXT_Y_OFFSET}[v{i}]"
        )
        filter_parts.append(drawtext)
    
    # Stack all videos horizontally
    inputs = "".join([f"[v{i}]" for i in range(n)])
    
    if BOTTOM_PADDING_PERCENT > 0:
        # hstack then add bottom padding only
        filter_parts.append(f"{inputs}hstack=inputs={n}[stacked]")
        # Add padding at bottom only: pad=width:height+padding:x:y
        # y=0 means video at top, padding at bottom
        pad_expr = f"[stacked]pad=iw:ih*(1+{BOTTOM_PADDING_PERCENT}):0:0:color={PADDING_COLOR}[out]"
        filter_parts.append(pad_expr)
    else:
        filter_parts.append(f"{inputs}hstack=inputs={n}[out]")
    
    filter_complex = ";".join(filter_parts)
    
    cmd.extend([
        "-filter_complex", filter_complex,
        "-map", "[out]",
        "-c:v", "libx264",
        "-crf", "18",
        "-y", output_path
    ])
    
    print(f"  Running ffmpeg with {n} videos side by side...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"ERROR: {result.stderr}")
        return False
    return True


def main():
    if not check_ffmpeg():
        return
    
    print("=" * 60)
    print(f"Video Combiner - Side by Side with Labels")
    print(f"Folder: {VIDEO_FOLDER}")
    print("=" * 60)
    
    # Auto-detect videos
    VIDEO_CONFIG = get_video_config(INPUT_DIR)
    
    if not VIDEO_CONFIG:
        print(f"\nERROR: No *_rgb.mp4 files found in {INPUT_DIR}")
        return
    
    # Validate input files
    print("\nChecking input files...")
    all_found = True
    for filename, label in VIDEO_CONFIG:
        path = os.path.join(INPUT_DIR, filename)
        if os.path.exists(path):
            print(f"  ✓ {label}: {filename}")
        else:
            print(f"  ✗ {label}: {filename} NOT FOUND")
            all_found = False
    
    if not all_found:
        print("\nERROR: Some input files not found. Exiting.")
        return
    
    # Combine videos side by side
    print("\nCombining videos side by side...")
    if combine_side_by_side(VIDEO_CONFIG, INPUT_DIR, OUTPUT_FILE):
        print(f"\n✓ Combined video saved to:")
        print(f"  {OUTPUT_FILE}")
        
        # Get video info
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", 
             "stream=width,height", "-of", "csv=p=0",
             OUTPUT_FILE],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            info = result.stdout.strip().split(',')
            if len(info) >= 2:
                print(f"  Resolution: {info[0]} x {info[1]}")
            
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", 
             "format=duration", "-of", "default=noprint_wrappers=1:nokey=1",
             OUTPUT_FILE],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            duration = float(result.stdout.strip())
            print(f"  Duration: {duration:.1f} seconds")
    else:
        print("Failed to combine videos")
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
