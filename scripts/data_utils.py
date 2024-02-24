"""
This script is designed to process and merge tracking annotation data from videos for object detection model training. It performs several key tasks:

1. Fixes tracking data to ensure each annotation includes the track_id.
2. Extracts frames from video files at a specified interval.
3. Merges annotations from multiple sources into a unified dataset.
4. Converts the merged tracking annotations into the COCO format for object detection tasks.

The script assumes that:
- Tracking data schema:
    {
        track_id: [
            [track_id, frame_number, x1, y1, x2, y2, 1, object_class],
            [track_id, frame_number, x1, y1, x2, y2, 1, object_class],
            [track_id, frame_number, x1, y1, x2, y2, 1, object_class]
        ],
        track_id: [
            [track_id, frame_number, x1, y1, x2, y2, 1, object_class],
            [track_id, frame_number, x1, y1, x2, y2, 1, object_class],
            [track_id, frame_number, x1, y1, x2, y2, 1, object_class]
        ]
    }

"""

import cv2
import json
import os
from tqdm import tqdm
import random
import glob
from sklearn.model_selection import train_test_split


def fix_tracking_data(tracking_data_path, fixed_data_path, debug = False):
    """
    Fixes tracking annotations by ensuring each annotation includes a track_id.
    If an annotation is missing the track_id (detected by having only 7 elements in the list),
    this function will prepend the missing track_id using the track_id from the outer dictionary.
    Parameters:
    - tracking_data_path: Path to the original tracking data JSON file.
    - fixed_data_path: Path where the fixed tracking data will be saved.
    """
    with open(tracking_data_path, 'r') as f:
        tracking_data = json.load(f)

    fixed_data = {}
    for track_id, track_annotations in tqdm(tracking_data.items(), desc="Fixing tracking data", mininterval=1):
        # Ensure each annotation has 8 elements, prepending track_id if missing.
        for i, ann in enumerate(track_annotations):
            if len(ann) != 8:
                if len(ann) == 7:
                    if ann[0] != track_id:
                        if debug:
                            print(f"     Annotation {ann} is missing track_id {track_id}. Prepending track_id.")
                            if i > 0:
                                print(f"Last annotation {track_annotations[i-1]} and track_id is {track_id}.")
                        ann.insert(0, track_id)
                    else:
                        raise ValueError(f"Annotation {ann} is missing something but does not miss track_id {track_id}.")
                else:
                    raise ValueError(f"Annotation {ann} does not have 7 or 8 elements. it has {len(ann)} elements.")
        fixed_annotations = [ann for ann in track_annotations]                
        fixed_data[track_id] = fixed_annotations

    with open(fixed_data_path, 'w') as f:
        json.dump(fixed_data, f, indent=4)


def extract_frames(video_path, output_dir, skip_frames):
    """
    Extracts frames from a video at a specified interval, adding a video identifier
    to each frame's filename to prevent name clashes between different videos.
    Parameters:
    - video_path: Path to the video file.
    - output_dir: Directory where extracted frames will be saved.
    - skip_frames: Interval at which frames are extracted.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Extract a unique identifier from the video filename (e.g., '867691e6' from '212/867691e6.mp4')
    video_identifier = os.path.splitext(os.path.basename(video_path))[0]
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in tqdm(range(total_frames), desc=f"Extracting frames from {os.path.basename(video_path)}", mininterval=1):
        ret, frame = cap.read()
        if not ret:
            break
        if i % skip_frames == 0:
            # Include the video identifier in the frame filename
            frame_filename = f"{output_dir}/{video_identifier}_{i:04d}.jpg"
            cv2.imwrite(frame_filename, frame)
    cap.release()


def merge_and_convert_to_coco(fixed_data_paths, output_path, images_dir, skip_frames):
    """
    Merges fixed tracking annotations from multiple sources and directly converts them into COCO format.
    This function incorporates video identifiers into frame filenames for unique referencing
    and considers the frame skipping interval used during frame extraction.
    
    Parameters:
    - fixed_data_paths: List of paths to the fixed tracking data JSON files.
    - output_path: Path where the COCO format data will be saved.
    - images_dir: Base directory containing extracted frames organized by video identifiers.
    - skip_frames: Interval at which frames were extracted, to ensure consistency with frame extraction logic.
    """
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    category_id_map = {}
    annotation_id_counter = 1
    frame_file_name_image_id_map = {}
    
    print("Starting the process of merging and converting to COCO format...")
    for fixed_data_path in fixed_data_paths:
        video_identifier = os.path.splitext(os.path.basename(fixed_data_path))[0].replace('_gt_fixed', '')
        print(f"Processing video: {video_identifier}")
        
        # Note: Direct construction of frame file names assumes frames are named with the video identifier and frame number.
        with open(fixed_data_path, 'r') as f:
            tracking_data = json.load(f)
        
        for track_id, annotations in tqdm(tracking_data.items(), desc="Processing annotations", mininterval=1):
            for ann in annotations:
                frame_number = ann[1]
                # Apply the skip_frames logic to ensure only the frames extracted are processed.
                if frame_number % skip_frames == 0:
                    frame_file_name = f"{video_identifier}_{frame_number:04d}.jpg"
                    frame_file_path = os.path.join(images_dir, frame_file_name)
                    if os.path.exists(frame_file_path):
                        if not any(d['file_name'] == frame_file_name for d in coco_data["images"]):
                            frame_file_name_image_id_map[frame_file_name] = len(frame_file_name_image_id_map) + 1
                            img = cv2.imread(frame_file_path)
                            height, width = img.shape[:2]
                            coco_data["images"].append({
                                "id": frame_file_name_image_id_map[frame_file_name],
                                "width": width,
                                "height": height,
                                "file_name": frame_file_name
                            })
                        
                        category = ann[-1]
                        if category not in category_id_map:
                            category_id_map[category] = len(category_id_map) + 1
                            coco_data["categories"].append({"id": category_id_map[category], "name": category})
                        
                        x1, y1, x2, y2 = ann[2:6]
                        coco_data["annotations"].append({
                            "id": annotation_id_counter,
                            "image_id": frame_file_name_image_id_map[frame_file_name],
                            "category_id": category_id_map[category],
                            "bbox": [x1, y1, x2 - x1, y2 - y1],
                            "area": (x2 - x1) * (y2 - y1),
                            "iscrowd": 0
                        })
                        annotation_id_counter += 1
                    else:
                        print(f"Frame file {frame_file_name} not found.")
                
    print("COCO conversion process completed. Saving data...")
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    with open(output_path, 'w') as f:
        json.dump(coco_data, f, indent=4)
    print(f"Converted COCO data saved to {output_path}")


def split_coco_data(coco_data_path, train_ratio=0.8):
    """
    Splits COCO data into training and validation sets and saves them in the same directory as the original COCO data.
    
    Parameters:
    - coco_data_path: Path to the COCO format JSON file.
    - train_ratio: Ratio of data to be used for training (default 0.8).
    """
    # Load the COCO data
    with open(coco_data_path, 'r') as f:
        coco_data = json.load(f)
    
    # Split images into training and validation sets
    images = coco_data['images']
    train_images, val_images = train_test_split(images, train_size=train_ratio, random_state=42)
    
    def filter_annotations_by_images(annotations, images):
        """Filter annotations by image_ids."""
        image_ids = set([img['id'] for img in images])
        return [ann for ann in annotations if ann['image_id'] in image_ids]
    
    # Filter annotations for training and validation sets
    annotations = coco_data['annotations']
    train_annotations = filter_annotations_by_images(annotations, train_images)
    val_annotations = filter_annotations_by_images(annotations, val_images)
    
    # Prepare train and val datasets
    train_data = {
        "images": train_images,
        "annotations": train_annotations,
        "categories": coco_data['categories']
    }
    val_data = {
        "images": val_images,
        "annotations": val_annotations,
        "categories": coco_data['categories']
    }
    
    # Determine the directory of coco_data_path to save the split data
    coco_data_dir = os.path.dirname(coco_data_path)
    train_data_path = os.path.join(coco_data_dir, 'coco_train.json')
    val_data_path = os.path.join(coco_data_dir, 'coco_val.json')
    
    # Save the split data to new JSON files
    with open(train_data_path, 'w') as f:
        json.dump(train_data, f, indent=4)
    with open(val_data_path, 'w') as f:
        json.dump(val_data, f, indent=4)
    
    print(f"Training data saved to {train_data_path}")
    print(f"Validation data saved to {val_data_path}")
    print(f"Data split into {len(train_images)} training and {len(val_images)} validation images.")



def print_dataset_info(coco_data_path, language='english'):
    """Prints information about the COCO dataset in the specified language.

    Parameters:
    - coco_data_path: Path to the COCO format JSON file.
    - language: Language for the output information ('english' or 'chinese').
    """
    try:
        with open(coco_data_path, 'r') as f:
            dataset = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file {coco_data_path} was not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: Failed to decode {coco_data_path} as JSON.")
        return

    keys = '\nkeys:' + ', '.join(dataset.keys())
    categories = '\nCategories:' if language == 'english' else '\n物体类别:'
    categories += str(dataset['categories'])
    num_images = '\nNumber of images:' if language == 'english' else '\n图像数量：'
    num_images += str(len(dataset['images']))
    num_annotations = '\nNumber of annotations:' if language == 'english' else '\n标注物体数量：'
    num_annotations += str(len(dataset['annotations']))
    sample_annotation = '\nSample annotation:' if language == 'english' else '\n查看一条目标物体标注信息：'
    sample_annotation += str(dataset['annotations'][0])

    print(keys + categories + num_images + num_annotations + sample_annotation)
