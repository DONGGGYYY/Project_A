"""Flip already-saved videos vertically (fix MuJoCo upside-down rendering)."""
from pathlib import Path
import imageio
import numpy as np

video_root = Path("outputs/videos")
mp4_files = list(video_root.rglob("*.mp4"))
print(f"Found {len(mp4_files)} videos")

for vid_path in mp4_files:
    reader = imageio.get_reader(str(vid_path))
    fps = reader.get_meta_data().get('fps', 20)
    frames = [frame[::-1].copy() for frame in reader]
    reader.close()

    # Overwrite in place
    imageio.mimsave(str(vid_path), frames, fps=fps)
    print(f"  flipped: {vid_path.relative_to(video_root.parent)}")

print("Done.")
