import xml.etree.ElementTree as ET

def xml_to_yolo(bbox, width, height):
    xmin, ymin, xmax, ymax = bbox
    x_center = ((xmax + xmin) / 2) / width
    y_center = ((ymax + ymin) / 2) / height
    bbox_width = (xmax - xmin) / width
    bbox_height = (ymax - ymin) / height
    return [x_center, y_center, bbox_width, bbox_height]

def parse_annotation(xml_file, class_mapping):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    width = int(root.find("size/width").text)
    height = int(root.find("size/height").text)

    objects = []
    for obj in root.findall("object"):
        label = obj.find("name").text
        class_id = class_mapping[label]
        bndbox = obj.find("bndbox")
        xmin, xmax = int(bndbox.find("xmin").text), int(bndbox.find("xmax").text)
        ymin, ymax = int(bndbox.find("ymin").text), int(bndbox.find("ymax").text)
        yolo_bbox = xml_to_yolo([xmin, ymin, xmax, ymax], width, height)
        objects.append([class_id] + yolo_bbox)

    return objects

def write_label(objects, filename):
    with open(filename, 'w') as f:
        for obj in objects:
            f.write(" ".join(str(x) for x in obj) + "\n")

