# A file to import configuration information for HAB training form XML file
#
# Copyright: (c) 2019 Paul Hill
import xml.etree.ElementTree as ET
import numpy

class inputXMLConfig():
    def __init__(self, xmlName):

        tree = ET.parse(xmlName)
        root = tree.getroot()

        for child in root:
            thisTag = child.tag
            thisText = child.text
            if thisTag == 'inDir':
                self.inDir = thisText
            elif thisTag == 'dataDir':
                self.dataDir = thisText
            elif thisTag == 'seqName':
                self.seqName = thisText
            elif thisTag == 'model':
                self.model = thisText
            elif thisTag == 'featureLength':
                self.featureLength = int(thisText)
            elif thisTag == 'seqLength':
                self.seqLength = int(thisText)
            elif thisTag == 'batchSize':
                self.batchSize = int(thisText)
            elif thisTag == 'epochNumber':
                self.epochNumber = int(thisText)
            elif thisTag == 'SVDFeatLen':
                self.SVDFeatLen = int(thisText)




