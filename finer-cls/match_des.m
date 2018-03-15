function match_des
phow = 1;
num_clusters = 500;
inter = 1;


test_path = 'Testing/';

cats = textread('cat_mapping.txt','%s');
all_images = textread('TrainingFiles.txt','%s');

if inter == 1
    files = dir('/Users/vidit/Thesis/Grocery_products/intersection_nmaj_70ov_nms_higher_recall/');
    crop_dim = [200,300]; 
else
    files = dir('/Users/vidit/Thesis/Grocery_products/results_org_scale_higher_recall/');
    crop_dim = [400,600];
end


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


if inter == 1
    match_dir = ['out_sift_match_inters_nms_200_300_' num2str(num_clusters) 'c_nmaj' extra '/'];
else
    match_dir = ['out_sift_match_400_600_' num2str(num_clusters) 'c_nmaj' extra '/'];      
end
mkdir(match_dir);


for i = 1:length(files)
    if strcmp('.',files(i).name) || strcmp('..',files(i).name)
        continue;
    end

    fname = files(i).name;
    
    disp(['For file:' fname]);
    ffname = [files(i).folder '/' fname];
    ff = fopen([match_dir fname],'w');

    if inter == 1
        [bboxs,labels,ids] = interBBox(ffname);
    else
        [bboxs,labels,ids] = bigBBox(ffname);
    end
    
    s = split(fname,'_');
    folder = s{5};
    imname = replace(s{6},'.txt','.jpg');
    
    impath = [test_path folder '/images/' imname];
    im = imread(impath);
        

    for ii = 1:length(labels)
        bbox = bboxs(ii,:);
        label = labels{ii};
        id = ids(ii);
        
        crop = im(bbox(2):bbox(4),bbox(1):bbox(3),:);
        
        if size(crop,1) > size(crop,2)
            crop = imresize(crop,[crop_dim(2),crop_dim(1)]);
        else
            crop = imresize(crop,crop_dim);
        end

       [f,sf] = phow_sift(crop);

        ss = split(label,'_');
        class = ss{2};
            
        kdtree = all_kdtree{str2num(class)};
        centers = all_centers{str2num(class)};
        enc = all_enc{str2num(class)};

        d = single(sf);
        nn = vl_kdtreequery(kdtree, centers, d) ;   

        assignments = zeros(num_clusters,size(d,2));
        assignments(sub2ind(size(assignments), nn, 1:length(nn))) = 1;

        des = vl_vlad(d,centers,single(assignments),'NormalizeComponents');


        enc = cell2mat(enc);
        dist = pdist2(des',enc');
        [sortV,sortI] = sort(dist);
        [minval,ind] = min(dist);

        category = cats(str2num(class));
        index = find(contains(all_images,category));
        [p,name,ext] = fileparts(all_images{index(ind)});

        final_id = offset(str2num(class))+str2num(name);
        fprintf(ff,'%d %d\n',id,final_id);
    end
    fclose(ff);
end
end

function [bboxs,labels,ids] = interBBox(fname)
    [bbox1, bbox2, bbox3, bbox4, labels, ids] = textread(fname,'%d %d %d %d %s %d');
    bboxs = cat(2,bbox1,bbox2,bbox3,bbox4);
end

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

function [f,sf] = phow_sift(im)
    im = single(im);
    [f,sf] = vl_phow(im,'sizes',[8,10,16,32],'step',6,'Color','rgb');
end
                                                                                                                                                                                                                                                                                                                                        