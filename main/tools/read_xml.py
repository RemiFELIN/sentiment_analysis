# Nos fonctions pour les fichiers xml
import xml.etree.ElementTree as ET


def get_elem_from_file_data(path):
    # On crée notre dataset
    dataset = []
    # Structure : ["text", "aspectTerm", "aspectCategory"]
    # On récupère nos data
    data = ET.parse(path)
    root = data.getroot()
    for node in root.iter('sentence'):
        # On stock les 'aspectTerm' et 'aspectCategory'
        # aspectTerms = []
        # Structure : ["term", "polarity"]
        # print(node)
        # On récupère la phrase :
        question = node[0].text
        # On récupère les 'aspectTerm'
        for aspectTerm in node.iter('aspectTerm'):
            term = aspectTerm.attrib.get('term')
            polarity = aspectTerm.attrib.get('polarity')
            # On push nos données dans notre dataset
            if polarity != 'conflict':
                dataset.append([question, term, polarity])
    return dataset
