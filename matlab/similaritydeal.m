wordnum = 16677;
Similarity = zeros(wordnum);
testnewPlace = importdata('testnewPlace.mat');
trainnewPlace = importdata('trainnewPlace.mat');
allnewPlace = importdata('allnewPlace.mat');
similarity_1 = importdata('similarity_1.mat');
city_flag = find(allnewPlace(:,3)==4);
cityRelation = allnewPlace(city_flag,:);
cityword = [];
for i = 1:size(cityRelation,1)
    cityword = [cityword;cityRelation(i,1)];
    cityword = [cityword;cityRelation(i,2)];
end
cityword = unique(cityword);
save cityword.mat cityword;
for i =1:size(similarity_1,1)
    Similarity(similarity_1(i,1),similarity_1(i,2))=similarity_1(i,3);
    Similarity(similarity_1(i,2),similarity_1(i,1))=similarity_1(i,3);
end
Similarity(cityword,:)=0;
Similarity(:,cityword)=0;
Similarity(cityword,cityword) = 1;
for i =1:size(Similarity,1)
    Similarity(i,i)=1;
end
k =1;
tensorSimilarity = zeros(wordnum*wordnum,3);
for i = 1:size(Similarity,1)
    for j = 1:size(Similarity,1)
        tensorSimilarity(k,1) = i;
        tensorSimilarity(k,2) = j;
        tensorSimilarity(k,3) = Similarity(i,j);
        k = k+1;
    end
end

save tensorSimilarity_1.mat tensorSimilarity;