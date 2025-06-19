import os
import shutil
import xml.etree.ElementTree as ET

ANNOTATIONS_DIR = 'data/archive/Train/Train/Annotations'
IMAGES_DIR = 'data/archive/Train/Train/JPEGImages'

OUTPUT_DIR = 'data/train'
PEDESTRIAN_DIR = os.path.join(OUTPUT_DIR, 'pedestrian')
NON_PEDESTRIAN_DIR = os.path.join(OUTPUT_DIR, 'non_pedestrian')

os.makedirs(PEDESTRIAN_DIR, exist_ok=True)
os.makedirs(NON_PEDESTRIAN_DIR, exist_ok=True)

for xml_file in os.listdir(ANNOTATIONS_DIR):
    if not xml_file.endswith('.xml'):
        continue

    xml_path = os.path.join(ANNOTATIONS_DIR, xml_file)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    has_pedestrian = False
    for obj in root.findall('object'):
        label = obj.find('name').text.lower()
        if label in ['person', 'pedestrian']:
            has_pedestrian = True
            break

    image_filename = xml_file.replace('.xml', '.jpg')
    image_path = os.path.join(IMAGES_DIR, image_filename)

    if os.path.exists(image_path):
        target_dir = PEDESTRIAN_DIR if has_pedestrian else NON_PEDESTRIAN_DIR
        shutil.copy(image_path, os.path.join(target_dir, image_filename))
    else:
        print(f"[!] Missing image for: {xml_file}")