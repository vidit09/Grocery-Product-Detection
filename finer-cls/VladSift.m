function VladSift()
    phow = 1;
    num_clusters = 500;

    dirname = ['vlad_phsift_out_' num2str(num_clusters) extra];
    mkdir(dirname);
    
    cat = textread('cat_mapping.txt','%s');
    all_images = textread('TrainingFiles.txt','%s');
    
    for i = 1:length(cat)
       disp(['For category:' cat{i}]);
       index = find(contains(all_images,cat{i}));
       [enc_,centers_] = descriptor(index,all_images,num_clusters,phow);
       enc = enc_;
       centers = centers_;
       save([dirname '/vlad_kmeans' num2str(i) '.mat'],'enc','centers');
    end    
        
end

function [enc,centers] = descriptor(index,all_images,num_clusters,phow)
    disp('Extract SIFT ...');
    for i = 1:length(index)
        im_name = all_images{index(i)}
        [path,name,ext] = fileparts(im_name);
        im_name = [path '/' name '_bkg_reduced.jpg'];
        [c,sift{i}] = phow_sift(im_name);
    end
    
    disp('Get Kmeans Centers ...');
    all_sift = single(cell2mat(sift));
    centers = vl_kmeans(all_sift, num_clusters,'Initialization', 'plusplus');
    
    kdtree = vl_kdtreebuild(centers) ;
    
    disp('Get VLAD ...');
    for i = 1:length(index)
        d = single(sift{i});
        nn = vl_kdtreequery(kdtree, centers, d) ;

        assignments = zeros(num_clusters,size(d,2));
        assignments(sub2ind(size(assignments), nn, 1:length(nn))) = 1;

        enc{i} = vl_vlad(d,centers,single(assignments),'NormalizeComponents');
    end
end

function [f,sf] = phow_sift(im_name)
    im = single(imread(im_name)); 
    [f,sf] = vl_phow(im,'sizes',[8,16,24,32],'step',6,'Color','rgb');
    
end
