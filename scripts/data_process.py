from data_utils import fix_tracking_data, extract_frames, merge_and_convert_to_coco, split_coco_data, print_dataset_info
import json


import os
import glob

# Base dataset directory
dataset_dir = 'dataset/'

# Automatically generate file paths
video_paths = glob.glob(dataset_dir + '**/*.mp4', recursive=True)  # Finds all .mp4 files
tracking_data_paths = [path.replace('.mp4', '_gt.json') for path in video_paths]  # Assumes annotation file naming convention

# Filter out videos without corresponding annotation files
video_paths_with_annotations = [path for path, annotation_path in zip(video_paths, tracking_data_paths) if os.path.exists(annotation_path)]
tracking_data_paths = [path for path in tracking_data_paths if os.path.exists(path)]

# Generate paths for fixed and merged data
fixed_data_paths = [path.replace('.mp4', '_gt_fixed.json') for path in video_paths_with_annotations]
output_path = os.path.join(dataset_dir, 'merged_dataset/annotations/coco_format_detection.json')
images_dir = os.path.join(dataset_dir, 'merged_dataset/images')
skip_frames = 5

# Example print to verify paths
for video_path, fixed_data_path in zip(video_paths_with_annotations, fixed_data_paths):
    print(f"Video Path: {video_path}, Fixed Data Path: {fixed_data_path}")


# Fix tracking data and extract frames for each video
#for video_path, tracking_data_path, fixed_data_path in zip(video_paths_with_annotations, tracking_data_paths, fixed_data_paths):
#    fix_tracking_data(tracking_data_path, fixed_data_path)
#    extract_frames(video_path, images_dir, skip_frames)

# Merge fixed annotations and convert to COCO format
merge_and_convert_to_coco(fixed_data_paths, output_path, images_dir, skip_frames)

# Split the data and save
coco_data_path = output_path
split_coco_data(coco_data_path)

# Print the dataset information
coco_data_path = output_path
print_dataset_info(coco_data_path, language='english')  # For English output
