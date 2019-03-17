function genSingleH5sWrapper(lat, lon, sample_date, h5nameOut)
%% Top level wrapper code to generate H5 file from lat, lon and date 
%% (inputs all config info and then calls genSingleH5s)
% USAGE:
%   genSingleH5sWrapper(lat, lon, sample_date, h5name)
% INPUT:
%   lat: lattitude for centre of datacube
%   lon: lattitude for centre of datacube
%   sample_date: lattitude for centre of datacube
%   h5nameOut: lattitude for centre of datacube
% OUTPUT:
%   -
% THE UNIVERSITY OF BRISTOL: HAB PROJECT
% Author Dr Paul Hill 26th June 2018
% Updated March 2019 PRH
% Updates for WIN compatibility: JVillegas 21 Feb 2019, Khalifa University

lat = str2double(lat); lon = str2double(lon);
sample_date = str2int(sample_date);

addpath('../extractData');
[~, pythonStr, tmpStruct] = getHABConfig;

%% load all config from XML file
confgData.inputFilename = tmpStruct.confgData.inputFilename.Text;
confgData.gebcoFilename = tmpStruct.confgData.gebcoFilename.Text;
confgData.wgetStringBase = tmpStruct.confgData.wgetStringBase.Text;
confgData.downloadDir = tmpStruct.confgData.downloadFolder.Text;
confgData.distance1 = str2double(tmpStruct.confgData.distance1.Text);
confgData.resolution = str2double(tmpStruct.confgData.resolution.Text);
confgData.numberOfDaysInPast = str2double(tmpStruct.confgData.numberOfDaysInPast.Text);
confgData.numberOfSamples = str2double(tmpStruct.confgData.numberOfSamples.Text);
confgData.mods = tmpStruct.confgData.Modality;
confgData.pythonStr = pythonStr;

inStruc.ii = 0;
inStruc.thisLat = lat;
inStruc.thisLon = lon;
inStruc.dayEnd = sample_date;
inStruc.thisCount = 0;

inStruc.h5name = h5nameOut;

genSingleH5s(inStruc, confgData);
quit()

