% This script contains functions to perform image segmentation, image evaluation, and comparison of
% different segmentations. The main function, segment_image, performs the segmentation process. The
% compare_segmentations function compares a predicted image segmentation to human segmentations of the
% same image.
%
% Source for Fuzzy Logic Edge Detection: https://uk.mathworks.com/help/fuzzy/fuzzy-logic-image-processing.html

% Function to perform image segmentation
function [seg] = segment_image(I)
    % Convert the image to grayscale
    Ia = rgb2gray(I);

    % Convolve the image with a Gaussian mask
    Ia = conv(Ia);

    % Perform Fuzzy Logic Edge Detection
    Gx = [-1 1];
    Gy = Gx';
    Ix = conv2(Ia, Gx, 'same');
    Iy = conv2(Ia, Gy, 'same');

    % Set up Fuzzy Inference System
    edgeFIS = setupFIS(Ix, Iy);

    % Evaluate the Fuzzy Inference System
    Ieval = evalFIS(Ix, Iy, edgeFIS, Ia);

    % Display the result of edge detection
    displayEdgeDetectionResult(Ieval);

    % Apply Canny edge detection to the result
    seg = applyCannyEdgeDetection(Ieval);
end

% Helper function to set up Fuzzy Inference System
function edgeFIS = setupFIS(Ix, Iy)
    edgeFIS = mamfis('Name', 'edgeDetection');
    edgeFIS = addInput(edgeFIS, [-1 1], 'Name', 'Ix');
    edgeFIS = addInput(edgeFIS, [-1 1], 'Name', 'Iy');

    edgeFIS = addMF(edgeFIS, 'Ix', 'gaussmf', [0.1 0], 'Name', 'zero');
    edgeFIS = addMF(edgeFIS, 'Iy', 'gaussmf', [0.1 0], 'Name', 'zero');
    edgeFIS = addOutput(edgeFIS, [0 1], 'Name', 'Iout');

    edgeFIS = addMF(edgeFIS, 'Iout', 'trimf', [0.1 1 1], 'Name', 'white');
    edgeFIS = addMF(edgeFIS, 'Iout', 'trimf', [0 0 0.7], 'Name', 'black');

    r1 = "If Ix is zero and Iy is zero then Iout is white";
    r2 = "If Ix is not zero or Iy is not zero then Iout is black";
    edgeFIS = addRule(edgeFIS, [r1 r2]);
    edgeFIS.Rules;
    return edgeFIS;
end

% Helper function to evaluate Fuzzy Inference System
function Ieval = evalFIS(Ix, Iy, edgeFIS, Ia)
    Ieval = zeros(size(Ia));
    for ii = 1:size(Ia, 1)
        Ieval(ii, :) = evalfis(edgeFIS, [(Ix(ii, :)); (Iy(ii, :))]');
    end
    return Ieval;
end

% Helper function to display edge detection result
function displayEdgeDetectionResult(Ieval)
    figure
    image(Ieval, 'CDataMapping', 'scaled')
    colormap('gray')
    title('Edge Detection Using Fuzzy Logic')
end

% Helper function to apply Canny edge detection
function seg = applyCannyEdgeDetection(Ieval)
    input = Ieval;
    result = edges(input);
    seg = result;
end


function [img_conv]=conv(img)

    % Convolve with a Guassian mask of 2 standard deviation and hsize 12 

    g=fspecial('gaussian',12,2);
    img_conv = conv2(img,g,'same');

end


function [seg]=edges(img)
    
   % CANNY EDGES

   gray  = im2gray(img);
   BW = edge(gray,'canny');
   imshow(BW);
   seg = BW;

end


%%%%%%%%%%%%%%%

function [f1score,TP,FP,FN]=evaluate(boundariesPred,boundariesHuman)
    
    %Returns the f1score quantifying the quality of the match between predicted
    %and human image segmentations.
    %
    %Note both inputs are assumed to show the boundaries between image regions.
    
    r=3; %set tolerance for boundary matching
    neighbourhood=strel('disk',r,0); 
    
    %make dilated and thinned versions of boundaries
    boundariesPredThin = boundariesPred.*bwmorph(boundariesPred,'thin',inf);
    boundariesHumanThin = prod(imdilate(boundariesHuman,neighbourhood),3);
    boundariesHumanThin = boundariesHumanThin.*bwmorph(boundariesHumanThin,'thin',inf);
    boundariesPredThick = imdilate(boundariesPred,neighbourhood);
    boundariesHumanThick = max(imdilate(boundariesHuman,neighbourhood),[],3);
    
    %Calculate statistics
    %true positives: pixels from predicted boundary that match pixels from any human boundary 
    %(human boundaries dilated to allow tolerance to match location)
    TP=boundariesPredThin.*boundariesHumanThick;
    %false positives: pixels that are predicted but do not match any human boundary
    FP=max(0,boundariesPred-boundariesHumanThick);
    %false negatives: human boundary pixels that do not match predicted boundary 
    %(predicted boundaries dilated to allow tolerance to match location)
    FN=max(0,boundariesHumanThin-boundariesPredThick);
    
    numTP=sum(TP(:));
    numFP=sum(FP(:));
    numFN=sum(FN(:));
    
    f1score=2*numTP/(2*numTP+numFP+numFN);
end



function b=convert_seg_to_boundaries(seg)
    %Performs conversion from an array containing region labels (seg) 
    %to one containing the boundaries between the regions (b)
    
    seg=padarray(seg,[1,1],'post','replicate');
    
    b=abs(conv2(seg,[-1,1],'same'))+abs(conv2(seg,[-1;1],'same'))+abs(conv2(seg,[-1,0;0,1],'same'))+abs(conv2(seg,[0,-1;1,0],'same'));
    
    b=im2bw(b(1:end-1,1:end-1),0);
end



function show_results(boundariesPred,boundariesHuman,f1score,TP,FP,FN)
    %Function used to show comparison between predicted and human image segmentations.
        
    maxsubplot(2,2,3); imagescc(boundariesPred); title('Predicted Boundaries')
    [a,b]=size(boundariesPred);
    if a>b
        ylabel(['f1score=',num2str(f1score,2)]);
    else
        xlabel(['f1score=',num2str(f1score,2)]);
    end
    maxsubplot(2,2,4); imagescc(mean(boundariesHuman,3)); title('Human Boundaries')
    
    maxsubplot(2,3,1); imagescc(TP); title('True Positives')
    maxsubplot(2,3,2); imagescc(FP); title('False Positives')
    maxsubplot(2,3,3); imagescc(FN); title('False Negatives')
    colormap('gray'); 
    drawnow;
    
    
    
    function imagescc(I)
    %Combines imagesc with some other commands to improve appearance of images
    imagesc(I,[0,1]); 
    axis('equal','tight'); 
    set(gca,'XTick',[],'YTick',[]);
    
    
    function position=maxsubplot(rows,cols,ind,fac)
    %Create subplots that are larger than those produced by the standard subplot command.
    %Good for plots with no axis labels, tick labels or titles.
    %*NOTE*, unlike subplot new axes are drawn on top of old ones; use clf first
    %if you don't want this to happen.
    %*NOTE*, unlike subplot the first axes are drawn at the bottom-left of the
    %window.
    
    if nargin<4, fac=0.075; end
    position=[(fac/2)/cols+rem(min(ind)-1,cols)/cols,...
              (fac/2)/rows+fix((min(ind)-1)/cols)/rows,...
              (length(ind)-fac)/cols,(1-fac)/rows];
    axes('Position',position); 
    end
    end
end


function compare_segmentations(imNum)
    %Compares a predicted image segmentation to human segmentations of the same image. 
    %The number of the image used is defined by the input parameter "imNum".
    %
    %Note, this function assumes that images and their corresponding human segmentations 
    %are stored in a sub-directory "Images" of the current working directory. If they are 
    %stored elsewhere, change the following to point to the correct location:
    ImDir='Images/';
    
    %load image 
    imFile=[ImDir,'im',int2str(imNum),'.jpg'];
    I=im2double(imread(imFile));
    
    %segment image
    segPred=segment_image(I); %<<<<<< calls your method for image segmentation
    
    %convert segmentation to a boundary map, if necessary
    segPred=round(segPred);
    inseg=unique(segPred(:));
    if min(inseg)==0 & max(inseg)==1
        %result is a boundary map
        boundariesPred=double(segPred);
    else
        %result is a segmentation map
        boundariesPred=double(convert_seg_to_boundaries(segPred)); %convert segmentation map to boundary map
    end
        
    %load human segmentations
    humanFiles=[ImDir,'im',int2str(imNum),'seg*.png'];
    numFiles=length(dir(humanFiles));
    for i=1:numFiles
        humanFile=['Images/im',int2str(imNum),'seg',int2str(i),'.png'];
        boundariesHuman(:,:,i)=im2double(imread(humanFile));
    end
    
    %evaluate and display results
    [f1score,TP,FP,FN]=evaluate(boundariesPred,boundariesHuman);
    figure(1), clf
    show_results(boundariesPred,boundariesHuman,f1score,TP,FP,FN);
end





