function CropIms;

addpath('../BM3D');
addpath('../LASIP');
addpath('../DNCNN/model');
addpath('../DNCNN/utilities');

if ismac
    baseDir = '/Users/csprh/tmp/Huawei/Train/';
else
    baseDir = '/mnt/storage/home/csprh/scratch/HAB/Huwei/Train/';
end

run /mnt/storage/home/csprh/code/matconvnet-1.0-beta25/matlab/vl_setupnn
load theseIndices

theseHO = inds.NHO;
classesA{1} = 'Buildings/';
classesA{2} = 'Foliage/';
classesA{3} = 'Text/';
for ii = 1:length(theseHO)
    thisInd = theseHO(ii);

    imName = ['Image_' num2str(thisInd) '.png']; 
    filepaths{ii}.Clean = getThisPath(baseDir,classesA,imName,'Clean/');
    filepaths{ii}.Noisy = getThisPath(baseDir,classesA,imName,'Noisy/');
end


cropImsDir(filepaths);


function cropImsDir(filepaths)


load(fullfile('/mnt/storage/home/csprh/code/HUAWEI/DNCNN/TrainingCodes/DnCNNHuawei/data/model_Huwei_All/model_Huwei_All-epoch-19.mat'));
net = vl_simplenn_tidy(net);
net.layers = net.layers(1:end-1);

%%%
%%%
net = vl_simplenn_tidy(net);


for i = 1 : length(filepaths)
    
    zRGB = im2double(imread(filepaths{i}.Noisy)); % uint8
    yRGB = im2double(imread(filepaths{i}.Clean)); % uint8
    
    %zRGB = im2double(imread([noisyDir thisImName]));
    %yRGB = im2double(imread([cleanDir thisImName]));
    greyzRGB = rgb2gray(zRGB);
    greyyRGB = rgb2gray(yRGB);
    thisSigma = estimate_noise(greyzRGB*255);
    thisSigma2 = function_stdEst(greyzRGB*255);
    clean = single(yRGB);
    input = single(zRGB);
    res    = vl_simplenn(net,input,[],[],'conserveMemory',true,'mode','test');
    %res = simplenn_matlab(net, input); %%% use this if you did not install matconvnet.i
    size(res(end).x)
    output = input - res(end).x;
    
    [PSNRNN(i), SSIMNN(i)] = Cal_PSNRSSIM(255*clean,255*output,0,0);
    
    %thisSigma = 7;
    [PSNR, yRGB_est] = CBM3D(clean, zRGB, thisSigma,'high',0);
    [PSNRB(i), SSIMB(i)] = Cal_PSNRSSIM(255*clean,255*yRGB_est,0,0);
    PSNRNN
    SSIMNN
    PSNRB
    SSIMB
    pause(0.1);
end









function output = getThisPath(baseDir, classesA,imName,cleanNoisy)

for ii = 1: length(classesA)
    thisClass = classesA{ii};
    thisPath = dir ([baseDir thisClass cleanNoisy imName]);
    if length(thisPath) == 1 
        output = [baseDir thisClass cleanNoisy imName];
    end
end



