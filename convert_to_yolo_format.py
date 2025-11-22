"""
Convert classification dataset to YOLO object detection format
For entire image labeling (each image = one damage type)
"""

import os
import shutil
from pathlib import Path
import yaml

def convert_classification_to_yolo(source_dir='data', output_dir='yolo_data'):
    """
    Convert classification dataset to YOLO format
    Each image gets a label file with bounding box covering entire image
    """
    
    # Class mapping
    classes = ['Crack', 'Pothole', 'Severe_Damage']
    
    # Create YOLO directory structure
    for split in ['train', 'val']:
        os.makedirs(f'{output_dir}/images/{split}', exist_ok=True)
        os.makedirs(f'{output_dir}/labels/{split}', exist_ok=True)
    
    print("Converting dataset to YOLO format...")
    
    for split in ['train', 'val']:
        split_name = split if split == 'train' else 'validation'
        source_split = f'{source_dir}/{split_name}'
        
        if not os.path.exists(source_split):
            print(f"Warning: {source_split} not found")
            continue
        
        total_converted = 0
        
        for class_idx, class_name in enumerate(classes):
            class_dir = os.path.join(source_split, class_name)
            
            if not os.path.exists(class_dir):
                print(f"Warning: {class_dir} not found")
                continue
            
            images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            for img_name in images:
                # Copy image
                src_img = os.path.join(class_dir, img_name)
                dst_img = os.path.join(f'{output_dir}/images/{split}', img_name)
                shutil.copy2(src_img, dst_img)
                
                # Create YOLO label (entire image as bounding box)
                # Format: class_id center_x center_y width height (normalized 0-1)
                label_name = os.path.splitext(img_name)[0] + '.txt'
                label_path = os.path.join(f'{output_dir}/labels/{split}', label_name)
                
                with open(label_path, 'w') as f:
                    # Entire image: center at 0.5, 0.5, width and height = 1.0
                    f.write(f"{class_idx} 0.5 0.5 1.0 1.0\n")
                
                total_converted += 1
            
            print(f"  {split}/{class_name}: {len(images)} images")
        
        print(f"Total {split}: {total_converted} images converted")
    
    # Create data.yaml for YOLOv8
    data_yaml = {
        'path': str(Path(output_dir).absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'names': {i: name for i, name in enumerate(classes)},
        'nc': len(classes)
    }
    
    yaml_path = f'{output_dir}/data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    
    print(f"\n✓ Conversion complete!")
    print(f"✓ YOLO dataset created at: {output_dir}")
    print(f"✓ Configuration saved: {yaml_path}")
    print(f"\nClasses: {classes}")
    print(f"Total classes: {len(classes)}")
    
    return output_dir

if __name__ == '__main__':
    convert_classification_to_yolo()
