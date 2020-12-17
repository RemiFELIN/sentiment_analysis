import tools.read_xml as read_xml
import os


def generate_dataset_from_file(xml_path):
    # Format file
    # Text | class_term | class_category | y:polarity(class_term, class_category)
    xml_path = os.getcwd().replace("\\", "/") + xml_path
    nodes = []
    dataset = read_xml.get_elem_from_file_data(xml_path)
    for node in dataset:
        nodes.append(node)
    return nodes
