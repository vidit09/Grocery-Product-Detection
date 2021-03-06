%%
% Get Finer Labels to Faster RCNN big and small boxes with labels
function match_des

    num_clusters = 500;
    % To choose small or big box. inter=1 for small and inter=0 for big
    inter = 1;

    % Set path to final matched output
    match_dir = 'final/output';
    mkdir(match_dir);

    % Set path to Grocery Product directory training and testing images
    train_path = 'Training/';
    test_path = 'Testing/';

    cats = textread([train_path 'cat_mapping.txt'],'%s');
    all_images = textread([train_path 'TrainingFiles.txt'],'%s');

    % Load the computed VLAD descriptors
    all_centers = {};
    all_enc = {};
    all_kdtree = {};

    disp('Loading Class Desc...')

    for cls = 1:27

        load(['vlad_phsift_out_' num2str(num_clusters) '/vlad_kmeans' num2str(cls) '.mat']);
        all_enc{cls} = enc;
        all_centers{cls} = centers;
        all_kdtree{cls} = vl_kdtreebuild(centers) ;

    end


    % Set output for Faster RCNN boxes
    % Crop needed to be resized for better matching
    if inter == 1
        files = dir('path/to/small/box/results');
        crop_dim = [200,300];
    else
        files = dir('path/to/big/box/results');
        crop_dim = [400,600];
    end

    % Process test images one at a time
    for i = 1:length(files)
        if strcmp('.',files(i).name) || strcmp('..',files(i).name)
            continue;
        end

        fname = files(i).name;

        disp(['For file:' fname]);
        ffname = [files(i).folder '/' fname];
        % output file
        ff = fopen([match_dir fname],'w');

        % read the Faster RCNN labels and bounding boxes
        if inter == 1
            [bboxs,labels,ids] = interBBox(ffname);
        else
            [bboxs,labels,ids] = bigBBox(ffname);
        end

        s = split(fname,'_');
        folder = s{5};
        imname = replace(s{6},'.txt','.jpg');

        % read the corresponding test images
        impath = [test_path folder '/images/' imname];
        im = imread(impath);

        % for all the predictions corresponding to this test image
        for ii = 1:length(labels)
            bbox = bboxs(ii,:);
            label = labels{ii};
            id = ids(ii);

            % take the crop using predicted bounding box
            crop = im(bbox(2):bbox(4),bbox(1):bbox(3),:);

            %resize the crop image
            if size(crop,1) > size(crop,2)
                crop = imresize(crop,[crop_dim(2),crop_dim(1)]);
            else
                crop = imresize(crop,crop_dim);
            end
            
            % compute SIFTs for crop
            [f,sf] = phow_sift(crop);

            ss = split(label,'_');
            class = ss{2};
 
            % get cluster centers corresponding to predicted label
            kdtree = all_kdtree{str2num(class)};
            centers = all_centers{str2num(class)};
            enc = all_enc{str2num(class)};

            d = single(sf);
            nn = vl_kdtreequery(kdtree, centers, d) ;

            % assign SIFTs to different centers
            assignments = zeros(num_clusters,size(d,2));
            assignments(sub2ind(size(assignments), nn, 1:length(nn))) = 1;
            
            % get VLAD corresponding to the crop
            des = vl_vlad(d,centers,single(assignments),'NormalizeComponents');

            % match descriptor to training images
            enc = cell2mat(enc);
            dist = pdist2(des',enc');
            [sortV,sortI] = sort(dist);
            [minval,ind] = min(dist);

            % get the corresponding final image label
            category = cats(str2num(class));
            index = find(contains(all_images,category));
            [p,name,ext] = fileparts(all_images{index(ind)});

            final_id = offset(str2num(class))+str2num(name);
            fprintf(ff,'%d %d\n',id,final_id);
        end
        fclose(ff);
    end
end

%%
% Read the results of small boxes
function [bboxs,labels,ids] = interBBox(fname)
    [bbox1, bbox2, bbox3, bbox4, labels, ids] = textread(fname,'%d %d %d %d %s %d');
    bboxs = cat(2,bbox1,bbox2,bbox3,bbox4);
end

%%
% Read the results of big boxes
function [bboxs,labels,ids] = bigBBox(fname)
    [labels, conf, bbox1, bbox2, bbox3, bbox4] = textread(fname,'%s %f %f %f %f %f');
    bboxs = cat(2,bbox1,bbox2,bbox3,bbox4);
    ids = [];
    if size(bboxs,1) > 0
        for i=1:size(bboxs,1)
            ids = cat(1,ids,i-1);

        end
    end
end

%%
%Compute dense color SIFT
function [f,sf] = phow_sift(im)
    im = single(im);
    [f,sf] = vl_phow(im,'sizes',[8,10,16,32],'step',6,'Color','rgb');
end
                                                                                                                                                                                                                                                                                                                                        