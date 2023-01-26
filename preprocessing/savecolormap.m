function [done] = savecolormap()

    set(0,'DefaultFigureVisible','off');
    set(gcf,'visible','off');
    myFolder = 'D:\workspace-extention\datasets\Data\images';
    if ~isfolder(myFolder)
        sprintf('Error: folder does not exist.')
    end
    
    filePattern = fullfile(myFolder, '*.jpg'); 
    theFiles = dir(filePattern);
    numberofImages = length(theFiles);
    slow = ceil(numberofImages/50);

    for limit = 1 : 50
        for k = (slow*(limit-1) + 1) : slow*limit
            baseFileName = theFiles(k).name;
            fullFileName = fullfile(theFiles(k).folder, baseFileName);
            image = imread(fullFileName);
            im = image(:, :, 3); % take blue channel
            im = double(im);
            
            figure;
            imagesc(im);
            map=jet(256);
            colormap(map);
            
            axis off;
            axis tight;
            myPath = 'D:\workspace-extention\datasets\Data\colormap';
            [fileName, ~] = strtok(baseFileName, '.');
            newname = sprintf('%s\%s_colormap.png', myPath, fileName);
            exportgraphics(gcf, newname);
        end
        clc;
        close all;
        pause(1); %so my laptop would not through up
    end
    done = 1;
end