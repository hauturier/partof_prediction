tensorall=importdata('allRelationNew.mat');
% save('tensorall.mat','tensorall','-v7.3');
disp('all tensor read end');
tensortrain = importdata('allRelationTrainNew.mat');
% save('tensortrain.mat','tensortrain','-v7.3');
disp('train tensor read end');
tensortest = importdata('allRelationTestNew.mat');
% save('tensortest.mat','tensortest','-v7.3');
disp('test tensor read end');

test1 = tensortest(find(tensortest(:,3)==1),:);
test2 = tensortest(find(tensortest(:,3)==2),:);
test3 = tensortest(find(tensortest(:,3)==3),:);
test4 = tensortest(find(tensortest(:,3)==4),:);

train1 = tensortrain(find(tensortrain(:,3)==1),:);
train2 = tensortrain(find(tensortrain(:,3)==2),:);
train3 = tensortrain(find(tensortrain(:,3)==3),:);
train4 = tensortrain(find(tensortrain(:,3)==4),:);

test_size1 = size(test1,1);
test_size2 = size(test2,1);
test_size3 = size(test3,1);
test_size4 = size(test4,1);

train_size1 = size(train1,1);
train_size2 = size(train2,1);
train_size3 = size(train3,1);
train_size4 = size(train4,1);

train_size = min([train_size1,train_size2,train_size3,train_size4]);
test_size = int32(train_size/3);

test1new_rand = randperm(size(test1,1));
test2new_rand = randperm(size(test2,1));
test3new_rand = randperm(size(test3,1));
test4new_rand = randperm(size(test4,1));

train1new_rand = randperm(size(train1,1));
train2new_rand = randperm(size(train2,1));
train3new_rand = randperm(size(train3,1));
train4new_rand = randperm(size(train4,1));

test1new = test1(test1new_rand(1:test_size),:);
test2new = test2(test2new_rand(1:test_size),:);
test3new = test3(test3new_rand(1:test_size),:);
test4new = test4(test4new_rand(1:test_size),:);

train1new = train1(train1new_rand(1:train_size),:);
train2new = train2(train2new_rand(1:train_size),:);
train3new = train3(train3new_rand(1:train_size),:);
train4new = train4(train4new_rand(1:train_size),:);

trainnew = [train1new;train2new;train3new;train4new];
testnew = [test1new;test2new;test3new;test4new];

save trainnew trainnew;
save testnew testnew;
allword = [];
trainword = [];
testword = [];
subjectwordnew = [];
objectwordnew = [];
for i =1:size(trainnew,1)
    allword = [allword;trainnew(i,1)];
    allword = [allword;trainnew(i,2)];
    subjectwordnew = [subjectwordnew;trainnew(i,1)];
    objectwordnew = [objectwordnew;trainnew(i,2)];
    trainword =[trainword;trainnew(i,1)];
    trainword =[trainword;trainnew(i,2)];
end

for i =1:size(testnew,1)
    allword = [allword;testnew(i,1)];
    allword = [allword;testnew(i,2)];
    subjectwordnew = [subjectwordnew;testnew(i,1)];
    objectwordnew = [objectwordnew;testnew(i,2)];
    testword =[testword;testnew(i,1)];
    testword =[testword;testnew(i,2)];
end
allwordnew = unique(allword);
trainwordnew = unique(trainword);
testwordnew = unique(testword);
subjectwordnew = unique(subjectwordnew);
objectwordnew = unique(objectwordnew);
save allwordnew allwordnew;

fp_train = fopen('trainnew.txt','wt');
for i =1:size(trainnew,1)
    fprintf(fp_train,'%d\t%d\t%d\t%d\n',trainnew(i,1),trainnew(i,2),trainnew(i,3),trainnew(i,4));
end
fclose(fp_train);
fp_test = fopen('testnew.txt','wt');
for i =1:size(testnew,1)
    fprintf(fp_test,'%d\t%d\t%d\t%d\n',testnew(i,1),testnew(i,2),testnew(i,3),testnew(i,4));
end
fclose(fp_test);
fp_word = fopen('allwordnew.txt','wt');
for i =1:size(allwordnew,1)
    fprintf(fp_word,'%d\n',allwordnew(i));
end
fclose(fp_word);
fp_testword = fopen('testwordnew.txt','wt');
for i =1:size(testwordnew,1)
    fprintf(fp_testword,'%d\n',testwordnew(i));
end
fclose(fp_testword);
fp_trainword = fopen('trainwordnew.txt','wt');
for i =1:size(trainwordnew,1)
    fprintf(fp_trainword,'%d\n',trainwordnew(i));
end
fclose(fp_trainword);

fp_subjectwordnew = fopen('subjectwordnew.txt','wt');
for i =1:size(subjectwordnew,1)
    fprintf(fp_subjectwordnew,'%d\n',subjectwordnew(i));
end
fclose(fp_subjectwordnew);
fp_objectwordnew = fopen('objectwordnew.txt','wt');
for i =1:size(objectwordnew,1)
    fprintf(fp_objectwordnew,'%d\n',objectwordnew(i));
end
fclose(fp_objectwordnew);