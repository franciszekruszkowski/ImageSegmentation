%{

The segment_image() function takes an image I as an input which is already read in the “double” format 
from integer format, meaning it is scaled and ready for all the transformations. 
Next the image is transformed into a grayscale format, making the third dimension equal to 1 instead of 3 
(which is the case for colourful images in the RGB format). Next image gets convolved with a Gaussian mask 
with a standard deviation of 2 and a mask size of 12 for initial smoothing using the function “conv(img)”. 
Afterward the Image undergoes Fuzzy Logic Edge Detection, which is an effective edge detection technique 
that uses membership functions to define specific degrees regarding whether a pixel belongs to an edge or a uniform region. 
This method uses the image gradient to detect local breaks. Next, the original image is convolved 
with the defined gradients and together with a set of other parameters (such as sx or sy, 
which resemble the standard deviation for the zero membership function), all the relevant matrices 
and parameters are added to the Fuzzy Inference System (FIS). Next, the Fuzzy Logic Edge Detection method 
returns an image with highly detailed edges. Because the returned images have so much detail, the output
 of the aforementioned method goes through another edge detection function "edges(img)" - which uses the
 Canny Edge detection method on our image with very detailed images, making them slightly less detailed
 and making them more look like the edges where drawn with "A thick line”. The reason for this
 additional edge detection is that the training sets images with edges drawn by humans were reasonably thick, 
therefore in order to achieve images resembling the ones that humans have drawn, the last step of the 
segment_image(I) function was essential.

Source : https://uk.mathworks.com/help/fuzzy/fuzzy-logic-image-processing.html
%}

function [seg]=segment_image(I)

   Ia = rgb2gray(I);
   
   % Convolving the image with a Gaussian mask 

   Ia = conv(Ia);
   
   % Fuzzy Logic Edge Detection 
   % Reference https://uk.mathworks.com/help/fuzzy/fuzzy-logic-image-processing.html

   % Gradient 
   
   Gx = [-1 1];
   Gy = Gx';
   Ix = conv2(Ia,Gx,'same');
   Iy = conv2(Ia,Gy,'same');

   % Fuzzy inference system 
   
   edgeFIS = mamfis('Name','edgeDetection');
   edgeFIS = addInput(edgeFIS,[-1 1],'Name','Ix');
   edgeFIS = addInput(edgeFIS,[-1 1],'Name','Iy');

   edgeFIS = addMF(edgeFIS,'Ix','gaussmf',[0.1 0],'Name','zero');
   edgeFIS = addMF(edgeFIS,'Iy','gaussmf',[0.1 0],'Name','zero');
   edgeFIS = addOutput(edgeFIS,[0 1],'Name','Iout');

   edgeFIS = addMF(edgeFIS,'Iout','trimf',[0.1 1 1],'Name','white');
   edgeFIS = addMF(edgeFIS,'Iout','trimf',[0 0 0.7],'Name','black');

   %RULES

   r1 = "If Ix is zero and Iy is zero then Iout is white";
   r2 = "If Ix is not zero or Iy is not zero then Iout is black";
   edgeFIS = addRule(edgeFIS,[r1 r2]);
   edgeFIS.Rules

   Ieval = zeros(size(Ia));
   for ii = 1:size(Ia,1)
        Ieval(ii,:) = evalfis(edgeFIS,[(Ix(ii,:));(Iy(ii,:))]');
   end

   figure
   image(Ieval,'CDataMapping','scaled')
   colormap('gray')
   title('Edge Detection Using Fuzzy Logic')

   % Images segmented with too many details, therefore apply our edge detection "canny" function

   input=Ieval;
   result=edges(input);
   seg=result;
   
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





