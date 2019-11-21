#!/bin/bash
python3 trainHAB_CV.py ./cnfgXMLs/NASNet11_svmGULF.xml
python3 trainHAB_CV.py ./cnfgXMLs/NASNet33_RFGULF.xml
python3 trainHAB_CV.py ./cnfgXMLs/NASNet33_lstm0GULF.xml
python3 trainHAB_CV.py ./cnfgXMLs/NASNet33_lstm1GULF.xml
python3 trainHAB_CV.py ./cnfgXMLs/NASNet33_lstm2GULF.xml
python3 trainHAB_CV.py ./cnfgXMLs/NASNet33_lstm3GULF.xml
python3 trainHAB_CV.py ./cnfgXMLs/NASNet33_lstm4GULF.xml
python3 trainHAB_CV.py ./cnfgXMLs/NASNet33_mlp0GULF.xml
python3 trainHAB_CV.py ./cnfgXMLs/NASNet33_mlp1GULF.xml
python3 trainHAB_CV.py ./cnfgXMLs/NASNet33_mlp2GULF.xml
