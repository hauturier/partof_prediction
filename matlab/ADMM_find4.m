function fit = ADMM_find4(R_all,R_train,R_test,P,rank)
% ----------initialize diary----------
diary_path = './log/' + datestr(now) + '.txt';
diary(diary_path);
diary on;
% ----------Parameters----------
singlewordnum = 16677;
class_num = 4;

best_A1 = 0;
best_A2 = 0;
best_B1 = 0;
best_B2 = 0;
best_C = 0;

iiindex = 0;
max_acc = 0;
max_filename = '';
max_iiiii = 0;
max_mse = 1000;
max_num = 0;
MaxIter = 30;
MSE = 0;

convt = 1e-5; % Convergence tolerance
fit = zeros(1, MaxIter);
oldfit = 0;
test_num = [1699 1699 1699 1699];

% ----------Initialize---------
rank = 10;
lambda1 = 1.55;
lambda2 = 1e-5;
rho = 0.16;

I = singlewordnum;
J = singlewordnum;
K = class_num;
A1=abs(randn(I,rank));A2=abs(randn(I,rank));
B1=abs(randn(J,rank));B2=abs(randn(J,rank));
C=abs(randn(K,rank));
thetaA1=zeros(I,rank);thetaA2=zeros(I,rank);
thetaB1=zeros(J,rank);thetaB2=zeros(J,rank);
A_bar=A1;B_bar=B1;
X=R_train;
Ik=eye(rank);
% ----------Computing----------
tic
for iter=1:MaxIter
    % ---------Update A1----------
    for i=1:max(X(:,1))% 遍历关系
        omega=find(X(:,1)==i);% 查找主语为i的关系
        if length(omega) > 0
            X1=X(omega,4)';% 查找主语为i的关系的所有评分
            CoB=C(X(omega,3),:).*B1(X(omega,2),:);% 计算COB，
            A1(i,:)=(X1*CoB+rho*A_bar(i,:)-thetaA1(i,:))*...
                (CoB'*CoB+(lambda1+rho)*Ik)^-1;
        end
    end
    % ----------Update A2----------
    for i=1:singlewordnum
        omega=similaritysubject(P,i,singlewordnum);
        if length(omega)>0
            
            p1=P(omega,3)';
            B21=B2(P(omega,2),:);
            A2(i,:)=(p1*B21+rho*A_bar(i,:)-thetaA2(i,:))*(B21'*B21+(rho+lambda2)*Ik)^-1;
        end
    end
    % ----------Update B1----------
    for i=1:max(X(:,2))% 遍历用户
        omega=find(X(:,2)==i);% 查找id为i的用户
        if length(omega)>0
            X2=X(omega,4)';% 查找id为i的用户的所有评分
            CoA=C(X(omega,3),:).*A1(X(omega,1),:);% 计算COB，id为i的用户的车型和
            B1(i,:)=(X2*CoA+rho*B_bar(i,:)-thetaB1(i,:))*...
                (CoA'*CoA+(lambda1+rho)*Ik)^-1;
        end
    end
    % ----------Update B2----------
    for i=1:singlewordnum
        omega = similarityobject(P,i,singlewordnum);
        if length(omega)>0
            p2=P(omega,3)';
            A21=A2(P(omega,1),:);
            B2(i,:)=(p2*A21+rho*B_bar(i,:)-thetaB2(i,:))*(A21'*A21+(rho+lambda2)*Ik)^-1;
        end
    end
    % ----------Update C----------
    for k=1:max(X(:,3))
        omega=find(X(:,3)==k);
        if length(omega)>0
            X3=X(omega,4)';
            BoA=B1(X(omega,2),:).*A1(X(omega,1),:);
            C(k,:)=(X3*BoA)*(BoA'*BoA+lambda1*Ik)^-1;
        end
    end
    % ----------Update theta----------
    thetaA1=thetaA1+rho*(A1-A_bar);
    thetaA2=thetaA2+rho*(A2-A_bar);
    thetaB1=thetaB1+rho*(B1-B_bar);
    thetaB2=thetaB2+rho*(B2-B_bar);
    % ----------Update A_bar B_bar----------
    A_bar=(A1+A2)/2;
    B_bar=(B1+B2)/2;
    % ----------Computing fit----------
    fit(iter) = sqrt(sum((sum(A_bar(R_train(:,1),:).*B_bar(R_train(:,2),:).*C(R_train(:,3),:),2)...
                -R_train(:,4)).^2)/length(R_train));
    disp(fit);

    if (oldfit - fit(iter)) < convt
        break;
    end
    oldfit = fit;
end

Result = zeros(size(R_test,1),4+2+2);
Result(:,1:2) = R_test(:,1:2);
Result(:,4+2+2) = R_test(:,3);
for i=1:size(R_test,1)
    for j=1:class_num
        Result(i,j+2)=sum(A_bar(R_test(i,1),:).*B_bar(R_test(i,2),:).*C(j,:),2);
    end
end

filename = ['finalresult',';',num2str(rank),';',num2str(MaxIter),';', ...
           num2str(lambda1),';',num2str(lambda2),';',num2str(rho),'.txt'];
fidresult_out = fopen(filename,'w');

for i=1:size(Result,1)
    [~,index]=max(Result(i,3:6));
    Result(i,4+2+1)=index;
    for j=1:size(Result(i,:),2)-1
        fprintf(fidresult_out,'%s\t',num2str(Result(i,j)));
    end
    fprintf(fidresult_out,'%s\n',num2str(Result(i,size(Result(i,:),2))));
end
fclose(fidresult_out);

accuracynum = 0;
RelationAccs = zeros(1,4);
for i =1:size(Result,1)
    if Result(i,4+2+1)==Result(i,4+2+2)
        accuracynum = accuracynum+1;
        RelationAccs(Result(i,4+2+2))=RelationAccs(Result(i,4+2+2))+1;
    end
end

acc_each = accuracynum ./ test_num;
acc_val = sum(accuracynum) ./ sum(test_num);

disp(['the number of right relations is: ' num2str(accuracynum)]);
disp(['Accuracy:  ' num2str(acc_val)]);
disp('the number of right relations is:');
disp(RelationAccs);
disp(max_filename);
disp(max_iiiii);
disp('Accuracy of each class:  ');
disp(acc_each)
disp('-----------------------------------------------------------------------------------------');
save ResultTest Result;
disp('end');

end
