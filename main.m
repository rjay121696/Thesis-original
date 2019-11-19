function varargout = main(varargin)
% MAIN MATLAB code for main.fig
%      MAIN, by itself, creates a new MAIN or raises the existing
%      singleton*.
%
%      H = MAIN returns the handle to a new MAIN or the handle to
%      the existing singleton*.
%
%      MAIN('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in MAIN.M with the given input arguments.
%
%      MAIN('Property','Value',...) creates a new MAIN or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before main_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to main_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help main

% Last Modified by GUIDE v2.5 19-Nov-2019 09:11:12

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @main_OpeningFcn, ...
                   'gui_OutputFcn',  @main_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before main is made visible.
function main_OpeningFcn(hObject, ~, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to main (see VARARGIN)
ah = axes('unit', 'normalized', 'position', [0 0 1 1]);
bg = imread('back2.jpg'); imagesc(bg);

set(ah,'handlevisibility','off','visible','off')

% making sure the background is behind all the other uicontrols
uistack(ah, 'bottom');


% Choose default command line output for main
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes main wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = main_OutputFcn(~, ~, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on button press in pushbutton1.
function pushbutton1_Callback(~, ~, handles)
% hObject    handle to pushbutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
clc

camera = webcam();
nnet = alexnet;
banana_rejected = false;

%%
% 
% Brightness.camera = 60;
while true
    picture = camera.snapshot;
    picture = imresize(picture, [227,227]);
    
    label = classify(nnet, picture);
    
    image(picture);
   
     if label == 'banana'
         banana_rejected = true; 
         set(handles.edit1, 'ForegroundColor', 'green', 'string', 'YES Banana');
     set(handles.uipanel1, 'highlightcolor', 'g')
    
    
    % For Removing Background
            % 1
        Agray = colouredToGray(picture);
     imageSize = size(picture);
     numRows = imageSize(1);
    numCols = imageSize(2);


    wavelengthMin = 4/sqrt(2);
    wavelengthMax = hypot(numRows,numCols);
    n = floor(log2(wavelengthMax/wavelengthMin));
    wavelength = 2.^(0:(n-2)) * wavelengthMin;

    deltaTheta = 45;
    orientation = 0:deltaTheta:(180-deltaTheta);

    % gabor draw the limit removing background
    g = gabor(wavelength,orientation);
    % 1
    gabormag = imgaborfilt(Agray,g);
    %2
    parfor i = 1:length(g)
    sigma = 0.5*g(i).Wavelength;
    K = 3;
    gabormag(:,:,i) = imgaussfilt(gabormag(:,:,i),K*sigma); 
    end
    %2
    %*
    X = 1:numCols;
    Y = 1:numRows;
    [X,Y] = meshgrid(X,Y);
    featureSet = cat(3,gabormag,X);
    featureSet = cat(3,featureSet,Y);
    %*
    %*1
    numPoints = numRows*numCols;
    X = reshape(featureSet,numRows*numCols,[]);
    %*1
    %*2
    X = bsxfun(@minus, X, mean(X));
    X = bsxfun(@rdivide,X,std(X));
    %*2
    %*3
    coeff = pca(X);
    feature2DImage = reshape(X*coeff(:,1),numRows,numCols);
    %*3
    %*3-1
    L = kmeans(X,2,'Replicates',5);
    %*3-1
    %3-2
	L = reshape(L,[numRows numCols]);
    %3-2
    
    %*3-3
    
    Aseg1 = zeros(size(picture),'like',picture);
%    Aseg2 = zeros(size(picture),'like',picture);
    BW = L == 2;
    BW = repmat(BW,[1 1 3]);
    Aseg1(BW) = picture(BW);
%      Aseg2(~BW) = picture(~BW);
    
    imshow(Aseg1);
    drawnow;
    
    %classify cavendish
outputFolder = fullfile('Banana');
outputFolder1 = fullfile('Classes');

rootFolder = fullfile(outputFolder, 'verify');
rootFolder1 = fullfile(outputFolder1, 'verify'); 

categories = {'cavendish', 'other'};
categories1 = {'A', 'B', 'other'};

imds = imageDatastore(fullfile(rootFolder,categories), 'LabelSource', 'foldernames');
imds1 = imageDatastore(fullfile(rootFolder1, categories1), 'LabelSource', 'foldernames');

tbl = countEachLabel(imds);
tbl1 = countEachlabel(imds1);

minSetCount = min(tbl{:,2});
minSetCount1 = min(tbl1{:,2});

imds = splitEachLabel(imds, minSetCount,'randomize');
imds1 = splitEachLabel(imds, minSetCount, 'randomize');

countEachLabel(imds);
countEachLabel(imds1);
% 
 cavendish = find(imds.Labels == 'cavendish', 1);
 other = find(imds.Labels == 'other', 1);
 
 A = find(imds1.Labels =='A', 1);
 B = find(imds1.Labels == 'B', 1);
 other1 = find(imds1.Labels == 'other', 1);
 
 net = resnet50();
 net1 = resnet50();
 
 net.Layers(1);
 net1.Layers(1);
 
 net.Layers(end);
 net1.Layers(end);
 
 
% 
 numel(net.Layers(end).ClassNames);
 numel(net1.Layers(end).ClassNames);
 
 [trainingSet, testSet] = splitEachLabel(imds, 0.3, 'randomize');
 [trainingSet1, testSet1] = splitEachLabel(imds1, 0.3, 'randomize');
% 
 imageSize = net.Layers(1).InputSize;
 imageSize1 = net.Layers(1).InputSize;
 
 augmentedTrainingSet = augmentedImageDatastore(imageSize, ...
      trainingSet, 'ColorPreprocessing', 'gray2rgb');
  augmentedTrainingSet1 = augmentedImageDatastore(imageSize1, ...
      trainingSet1, 'ColorPreprocessing', 'gray2rgb');
  
 augmentedTestSet = augmentedImageDatastore(imageSize, ...
     testSet,'ColorPreprocessing', 'gray2rgb');
 augmentedTestSet1 = augmentedImageDatastore(imageSize1, ...
     testSet1,'ColorPreprocessing', 'gray2rgb');
% 
 w1 = net.Layers(2).Weights;
 w1 = mat2gray(w1);

 w2 = net1.Layers(2).Weights;
 w2 = mat2gray(w1);
 
 featureLayer = 'fc1000';
 featureLayer1 = 'fc1000';
 
 trainingFeatures1 = activations(net,...
     augmentedTrainingSet, featureLayer, 'MiniBatchSize', 32, 'OutputAs', 'columns');
 trainingFeatures2 = activations(net1,...
     augmentedTrainingSet1, featureLayer1, 'MiniBatchSize', 32, 'OutputAs', 'columns');
 
trainingLables = trainingSet.Labels;
trainingLables1 = trainingSet1.Labels;

classifier = fitcecoc(trainingFeatures, trainingLables, 'Learner', 'Linear', 'Coding', 'onevsall', 'ObservationsIn', 'columns');
classifier1 = fitcecoc(trainingFeatures1, trainingLables1, 'Learner', 'Linear', 'Coding', 'onevsall', 'ObservationsIn', 'columns');

testFeatures = activations(net, augmentedTestSet, featureLayer, 'MiniBatchSize', 32, 'OutputAs', 'columns');
testFeatures1 = activations(net1, augmentedTestSet1, featureLayer1, 'MiniBatchSize', 32, 'OutputAs', 'columns');

predictLabels = predict(classifier, testFeatures, 'ObservationsIn', 'columns');
predictLabels1 = predict(classifier1, testFeatures1, 'ObservationsIn', 'columns');

testLables = testSet.Labels;
testLables1 = testSet1.Labels;

confMat = confusionmat(testLables, predictLabels);
confMat = bsxfun(@rdivide, confMat, sum(confMat,2));

confMat1 = confusionmat(testLables1, predictLabels1);
confMat1 = bsxfun(@rdivide, confMat1, sum(confMat1,2));

mean(diag(confMat));
mean(diag(confMat1));


ds = augmentedImageDatastore(imageSize, Aseg1, 'ColorPreprocessing', 'gray2rgb');
ds1 = augmentedImageDatastore(imageSize1, Aseg1, 'ColorPreprocessing', 'gray2rgb');

imageFeatures = activations(net, ds, featureLayer, 'MiniBatchSize', 32, 'OutputAs', 'columns');
imageFeatures1 = activations(net1, ds1, featureLayer1, 'MiniBatchSize', 32, 'OutputAs', 'columns');

labels = predict(classifier, imageFeatures, 'ObservationsIn', 'columns');
labels1 = predict(classifier, imageFeatures, 'ObsevationIn', 'columns');

if labels == 'cavendish'
set(handles.edit6, 'ForegroundColor', 'g', 'string', 'Cavendish');
       
else
set(handles.edit6, 'ForegroundColor', 'g', 'string', 'this is not an cavendish');
set(handles.edit7, 'ForegroundColor', 'r', 'string', 'this is Not Cavendish Type');
end

    %Detect Color Grren
   diff_im = imsubtract(Aseg1(:,:,2), rgb2gray(Aseg1)); 
      diff_im = medfilt2(diff_im, [3 3]);
      diff_im = imbinarize(diff_im,0.18);
      diff_im = bwareaopen(diff_im,300);
      bw = bwlabel(diff_im, 8);
      stats = regionprops(bw, 'BoundingBox', 'Centroid');
      
        imshow(Aseg1)
      hold on
      for object = 1:length(stats)
          bb = stats(object).BoundingBox;
          bc = stats(object).Centroid;
          rectangle('Position',bb,'EdgeColor','g','LineWidth',2)
          plot(bc(1),bc(2), '-m+')
          a=text(bc(1)+15,bc(2), strcat('X: ', num2str(round(bc(1))), '    Y: ', num2str(round(bc(2)))));
          set(a, 'FontName', 'Arial', 'FontWeight', 'bold', 'FontSize', 12, 'Color', 'yellow');
      end
      hold off
       drawnow;
       
 grayImage = Aseg1;
[rows, columns, numberOfColorChannels] = size(grayImage)
if numberOfColorChannels > 1
  grayImage = rgb2gray(grayImage);
end
hp = impixelinfo();
binaryImage = grayImage == 0;
binaryImage = imclearborder(binaryImage);
[labeledImage, numBlobs] = bwlabel(binaryImage);
coloredLabels = label2rgb (labeledImage, 'hsv', 'k', 'shuffle');
props = regionprops(labeledImage, 'BoundingBox', 'Centroid');

imshow(picture);
hold on;
for k = 1 : numBlobs
   bb = props(k).BoundingBox;
   bc = props(k).Centroid;
   rectangle('Position',bb,'EdgeColor','c','LineWidth',2);
end
drawnow;
     if  ~isempty(props)
          set(handles.edit4, 'ForegroundColor', 'r', 'string', 'YES');
          set(handles.edit5, 'ForegroundColor', 'r', 'string', 'REJECT');
      else 
          set(handles.edit4, 'ForegroundColor', 'g', 'string', 'NO spots');
          set(handles.edit5, 'ForegroundColor', 'r', 'string', 'ACCEPT');
     end 
     
        if  ~isempty(stats)
          set(handles.edit2, 'ForegroundColor', 'g', 'string', 'YES');
          set(handles.edit3, 'ForegroundColor', 'g', 'string', 'Accept');
      else isempty(stats)
          set(handles.edit2, 'ForegroundColor', 'red', 'string', 'NO');
          set(handles.edit3, 'ForegroundColor', 'g', 'string', 'REJECT');
      end
      
  if isempty(props) && ~isempty(stats) && labels == 'cavendish' && labels1 == 'A' || ...
      labels1 == 'B'
  title('ACCEPT');
  else
      title('REJECT');
      msgbox('Banana');
  end
  
     else
        
      set(handles.edit1, 'ForegroundColor', 'red', 'string', 'this is not an banana');
     set(handles.uipanel1, 'highlightcolor', 'r');
     %Detect Color Grren
    diff_im = imsubtract(picture(:,:,2), rgb2gray(picture)); 
      diff_im = medfilt2(diff_im, [3 3]);
      diff_im = imbinarize(diff_im,0.18);
      diff_im = bwareaopen(diff_im,300);
      bw = bwlabel(diff_im, 8);
      stats = regionprops(bw, 'BoundingBox', 'Centroid');
      
        imshow(picture)
         
      hold on
      for object = 1:length(stats)
          bb = stats(object).BoundingBox;
          bc = stats(object).Centroid;
          rectangle('Position',bb,'EdgeColor','g','LineWidth',2)
          plot(bc(1),bc(2), '-m+')
          a=text(bc(1)+15,bc(2), strcat('X: ', num2str(round(bc(1))), '    Y: ', num2str(round(bc(2)))));
          set(a, 'FontName', 'Arial', 'FontWeight', 'bold', 'FontSize', 12, 'Color', 'yellow');
         
      end
      hold off
       drawnow;
          
      if  ~isempty(stats)
          set(handles.edit2, 'ForegroundColor', 'g', 'string', 'YES');
          set(handles.edit3, 'ForegroundColor', 'red', 'string', 'But this is not an banana');
      elseif isempty(stats)
          set(handles.edit2, 'ForegroundColor', 'red', 'string', 'NO');
          set(handles.edit3, 'ForegroundColor', 'red', 'string', 'this is not an bananaa');
      end
%         im_gray = rgb2gray(picture);
%          holes = im_gray < 5;
%         [~,count] = bwlabel(holes);
%     im_gray = rgb2gray(picture);
%         holes = im_gray < 5;
% 
% 
%             %// Further processing
%             se = strel('square', 5);
%         holes_process = imopen(holes, se);
% 
% 
%             [~,count] = bwlabel(holes_process);
%         [labeledImage, numberOfBlobs] = bwlabel(~binaryImage);
 grayImage = picture;
[rows, columns, numberOfColorChannels] = size(grayImage)
if numberOfColorChannels > 1
  grayImage = rgb2gray(grayImage);
end
hp = impixelinfo();
binaryImage = grayImage == 0;
binaryImage = imclearborder(binaryImage);
[labeledImage, numBlobs] = bwlabel(binaryImage);
coloredLabels = label2rgb (labeledImage, 'hsv', 'k', 'shuffle');
props = regionprops(labeledImage, 'BoundingBox', 'Centroid');

imshow(picture);
hold on;
for k = 1 : numBlobs
   bb = props(k).BoundingBox;
   bc = props(k).Centroid;
   rectangle('Position',bb,'EdgeColor','c','LineWidth',2);
end
drawnow;
     if  ~isempty(props)
          set(handles.edit4, 'ForegroundColor', 'r', 'string', numBlobs);
          set(handles.edit5, 'ForegroundColor', 'r', 'string', 'This is not an banana');
      else 
          set(handles.edit4, 'ForegroundColor', 'g', 'string', 'NO spots');
          set(handles.edit5, 'ForegroundColor', 'r', 'string', 'But this is not an banana');
     end       
        
     set(handles.edit6, 'ForegroundColor', 'r', 'string', '-----');
     set(handles.edit7, 'ForegroundColor', 'r', 'string', 'This is not an banana');
   end
   
   
   if banana_rejected == false
       msgbox('Please put a banana');
       n = 5;
       pause(n);
   end
end
% --- Executes on button press in pushbutton2.
function pushbutton2_Callback(~, ~, ~)
% hObject    handle to pushbutton2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
clear all; close all; clc;



function edit1_Callback(~, ~, ~)
% hObject    handle to edit1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit1 as text
%        str2double(get(hObject,'String')) returns contents of edit1 as a double


% --- Executes during object creation, after setting all properties.
function edit1_CreateFcn(hObject, ~, ~)
% hObject    handle to edit1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit2_Callback(~, ~, ~)
% hObject    handle to edit2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit2 as text
%        str2double(get(hObject,'String')) returns contents of edit2 as a double


% --- Executes during object creation, after setting all properties.
function edit2_CreateFcn(hObject, ~, ~)
% hObject    handle to edit2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit3_Callback(hObject, eventdata, handles)
% hObject    handle to edit3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit3 as text
%        str2double(get(hObject,'String')) returns contents of edit3 as a double


% --- Executes during object creation, after setting all properties.
function edit3_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit4_Callback(hObject, eventdata, handles)
% hObject    handle to edit4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit4 as text
%        str2double(get(hObject,'String')) returns contents of edit4 as a double


% --- Executes during object creation, after setting all properties.
function edit4_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit5_Callback(hObject, eventdata, handles)
% hObject    handle to edit5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit5 as text
%        str2double(get(hObject,'String')) returns contents of edit5 as a double


% --- Executes during object creation, after setting all properties.
function edit5_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in popupmenu1.
function popupmenu1_Callback(hObject, eventdata, handles)
% hObject    handle to popupmenu1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns popupmenu1 contents as cell array
%        contents{get(hObject,'Value')} returns selected item from popupmenu1


% --- Executes during object creation, after setting all properties.
function popupmenu1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to popupmenu1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit6_Callback(hObject, eventdata, handles)
% hObject    handle to edit6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit6 as text
%        str2double(get(hObject,'String')) returns contents of edit6 as a double


% --- Executes during object creation, after setting all properties.
function edit6_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit7_Callback(hObject, eventdata, handles)
% hObject    handle to edit7 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit7 as text
%        str2double(get(hObject,'String')) returns contents of edit7 as a double


% --- Executes during object creation, after setting all properties.
function edit7_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit7 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on slider movement.
function slider1_Callback(hObject, eventdata, handles)
% hObject    handle to slider1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider


% --- Executes during object creation, after setting all properties.
function slider1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to slider1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end
