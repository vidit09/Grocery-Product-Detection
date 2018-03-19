%%
% Compute VLAD descriptor out of dense color SIFT
function vlad_sift()
    
    % Number of Vocabulary Words
    num_clusters = 500;
    % Set path to Grocery Product directory
    train_image_dir = '';

    % Output Directory
    outdirname = ['vlad_phsift_out_' num2str(num_clusters) extra];
    mkdir(outdirname);
    
    cat = textread([train_image_dir 'cat_mapping.txt'],'%s');
    all_images = textread([train_image_dir 'TrainingFiles.txt'],'%s');
    
    % Calculate cluster per class and vlad
    for i = 1:length(cat)
       disp(['For category:' cat{i}]);
       index = find(contains(all_images,cat{i}));
       [enc_,centers_] = descriptor(index,all_images,num_clusters,train_image_dir);
       
       % Save the descriptor and kmeans centers
       enc = enc_;
       centers = centers_;
       save([outdirname '/vlad_kmeans' num2str(i) '.mat'],'enc','centers');
    end    
        
end

%%
% Get vlad descriptor for all images in a category
function [enc,centers] = descriptor(index,all_images,num_clusters,dir_path)
    
    disp('Extract SIFT ...');
    for i = 1:length(index)
        im_name = all_images{index(i)};
        [path,name,ext] = fileparts(im_name);
        im_name = [dir_path path '/' name '_bkg_reduced.jpg'];
        [c,sift{i}] = phow_sift(im_name);
    end
    
    %Cluster the SIFTs
    disp('Get Kmeans Centers ...');
    all_sift = single(cell2mat(sift));
    centers = vl_kmeans(all_sift, num_clusters,'Initialization', 'plusplus');
    
    kdtree = vl_kdtreebuild(centers) ;
    
    %Computer VLAD per image
    disp('Get VLAD ...');
    for i = 1:length(index)
        d = single(sift{i});
        nn = vl_kdtreequery(kdtree, centers, d) ;

        assignments = zeros(num_clusters,size(d,2));
        assignments(sub2ind(size(assignments), nn, 1:length(nn))) = 1;

        enc{i} = vl_vlad(d,centers,single(assignments),'NormalizeComponents');
    end
end
%%
%Compute dense color SIFT
function [f,sf] = phow_sift(im_name)
    im = single(imread(im_name)); 
    [f,sf] = vl_phow(im,'sizes',[8,16,24,32],'step',6,'Color','rgb');
    
end
