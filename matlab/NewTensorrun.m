% tensorall=importdata('tensorall_new.mat');
% save('tensorall.mat','tensorall','-v7.3');
% disp('all tensor read end');
tensortrain = importdata('trainnewPlace.mat');
% save('tensortrain.mat','tensortrain','-v7.3');
disp('train tensor read end');
tensortest = importdata('testnewPlace.mat');
% save('tensortest.mat','tensortest','-v7.3');
disp('test tensor read end');
similarity = importdata('tensorSimilarity_1.mat');
% save('similarity.mat','similarity','-v7.3');
disp('similarity read end');
tensorall = [tensortest;tensortrain];
ADMM_find4(tensorall,tensortrain,tensortest,similarity,10);