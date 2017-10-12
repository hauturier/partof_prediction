function fit = ADMM_findnew(R_all,R_train,R_test,P,rank)
MaxIter = 10;
max_filename = '';
max_num = 0;
max_iiiii = 0;
iiindex = 0;
fit = importdata('fit.mat');
test_num = [1699 1699 1699 1699];
MSE = 0;
max_mse = 1000;
max_acc = 0;
for MaxIter = 10;
    for rank = 15:30;
        %     for lambda1 = [2:0.5:3.5];
        %         for lambda2 = [2:0.5:3.5];
        %             for rho = [0.1:0.1:0.5];
        for lambda1 = 1.55;
            for lambda2 = 0.0001;
                for rho = 0.16;
                    ci_max = 0;
                    for cishu =41:50;
                        iiindex=iiindex+1;
                        % disp('-----------------------------------------------------------------------------------------');
                        singlewordnum = 16677;
                        % I = max(R_all(:,1));
                        % J = max(R_all(:,2));
                        % I = max(I,J);
                        I = singlewordnum;
                        J = singlewordnum;
                        K = max(R_all(:,3));
                        % A_bar=zeros(I,rank);B_bar=zeros(J,rank);
                        A1=abs(randn(I,rank));A2=abs(randn(I,rank));
                        B1=abs(randn(J,rank));B2=abs(randn(J,rank));
                        C=abs(randn(K,rank));
                        thetaA1=zeros(I,rank);thetaA2=zeros(I,rank);
                        thetaB1=zeros(J,rank);thetaB2=zeros(J,rank);
                        A_bar=A1;B_bar=B1;
                        X=R_train;
                        Ik=eye(rank);
                        tic
                        for iter=1:MaxIter
%                             fprintf('%d\n',iter);
                            for i=1:max(X(:,1))%遍历关系
                                omega=find(X(:,1)==i);%查找主语为i的关系
                                if length(omega)>0
                                    X1=X(omega,4)';%查找主语为i的关系的所有评分
                                    CoB=C(X(omega,3),:).*B1(X(omega,2),:);%计算COB，
                                    A1(i,:)=(X1*CoB+rho*A_bar(i,:)-thetaA1(i,:))*...
                                        (CoB'*CoB+(lambda1+rho)*Ik)^-1;
                                end
                            end
                            %                     disp('A1 end');
                            %                     disp(datestr(now));
                            % for i=1:max(P(:,1))
                            for i=1:singlewordnum
                                %         omega=find(P(:,1)==i);
                                omega=similaritysubject(P,i,singlewordnum);
                                if length(omega)>0
                                    
                                    p1=P(omega,3)';
                                    B21=B2(P(omega,2),:);
                                    A2(i,:)=(p1*B21+rho*A_bar(i,:)-thetaA2(i,:))*(B21'*B21+(rho+lambda2)*Ik)^-1;
                                end
                                %                                 save omega omega;
                            end
                            %                     disp('A2 end');
                            %                     disp(datestr(now));
                            for i=1:max(X(:,2))%遍历用户
                                omega=find(X(:,2)==i);%查找id为i的用户
                                if length(omega)>0
                                    X2=X(omega,4)';%查找id为i的用户的所有评分
                                    CoA=C(X(omega,3),:).*A1(X(omega,1),:);%计算COB，id为i的用户的车型和
                                    B1(i,:)=(X2*CoA+rho*B_bar(i,:)-thetaB1(i,:))*...
                                        (CoA'*CoA+(lambda1+rho)*Ik)^-1;
                                end
                            end
                            %                     disp('B1 end');
                            %                     disp(datestr(now));
                            %                     for i=1:max(P(:,2))
                            for i=1:singlewordnum
                                %         omega=find(P(:,2)==i);
                                omega = similarityobject(P,i,singlewordnum);
                                if length(omega)>0
                                    p2=P(omega,3)';
                                    A21=A2(P(omega,1),:);
                                    B2(i,:)=(p2*A21+rho*B_bar(i,:)-thetaB2(i,:))*(A21'*A21+(rho+lambda2)*Ik)^-1;
                                end
                                %                                 save omega1 omega;
                            end
                            %                     disp('B2 end');
                            %                     disp(datestr(now));
                            for k=1:max(X(:,3))
                                omega=find(X(:,3)==k);
                                if length(omega)>0
                                    X3=X(omega,4)';
                                    BoA=B1(X(omega,2),:).*A1(X(omega,1),:);
                                    C(k,:)=(X3*BoA)*(BoA'*BoA+lambda1*Ik)^-1;
                                end
                            end
                            %                     disp('C end');
                            %                     disp(datestr(now));
                            thetaA1=thetaA1+rho*(A1-A_bar);
                            thetaA2=thetaA2+rho*(A2-A_bar);
                            thetaB1=thetaB1+rho*(B1-B_bar);
                            thetaB2=thetaB2+rho*(B2-B_bar);
                            A_bar=(A1+A2)/2;
                            B_bar=(B1+B2)/2;
                            %#####    RMSE  #####
                            %   fit(iter) =sqrt(sum((sum(A_bar(R_train(:,1),:).*B_bar(R_train(:,2),:).*C_bar(R_train(:,3),:),2)...
                            %                                                        -R_train(:,4)).^2)/length(R_train));
                            %
                            %
                            %  disp('fit');
                            %  size(A_bar(R_test(:,1),:).*B_bar(R_test(:,2),:).*C_bar(R_test(:,3),:))
                            %  cons=sum(A_bar(R_test(:,1),:).*B_bar(R_test(:,2),:).*C_bar(R_test(:,3),:),2);
                            %  size(cons)
                            % %  cons(find(cons>5))=5*ones(length(find(cons>5)),1);
                            % %  cons(find(cons<0.5))=0.5*ones(length(find(cons<0.5)),1);
                            %  test_err(iter)=sqrt(sum((cons-R_test(:,4)).^2)/length(R_test));
                            disp(i);
                        end
                        
                        Result = zeros(size(R_test,1),4+2+2);
                        Result(:,1:2) = R_test(:,1:2);
                        Result(:,4+2+2) = R_test(:,3);
                        for i=1:size(R_test,1)
                            for j=1:4
                                %     Result(i,j+2)=scoreweight(j)*sum(A_bar(R_test(i,1),:).*B_bar(R_test(i,2),:).*C_bar(j,:),2);
                                Result(i,j+2)=sum(A_bar(R_test(i,1),:).*B_bar(R_test(i,2),:).*C(j,:),2);
                            end
                        end
                        
                        % for k = 1:9
                        %     Result(:,k+2) = scoreweight(k).*Result(:,k+2);
                        % end
                        % toc
                        %
                        % fit
                        % test_err
                        %
                        % figure(1)
                        % plot(fit)
                        % figure(2)
                        % plot(test_err)
                        
                        filename = ['finalresult_withtest_allnew4',';',num2str(rank),';',num2str(MaxIter),...
                            ';',num2str(lambda1),';',num2str(lambda2),';',num2str(rho),';',num2str(cishu),'.txt'];
                        fidresult_out = fopen(filename,'w');
                        for i=1:size(Result,1)
                            [maxr,index]=max(Result(i,3:6));
                            Result(i,4+2+1)=index;
                            for j=1:size(Result(i,:),2)-1
                                fprintf(fidresult_out,'%s\t',num2str(Result(i,j)));
                            end
                            fprintf(fidresult_out,'%s\n',num2str(Result(i,size(Result(i,:),2))));
                            %     fprintf(fidresult_out,'%s\n',num2str(index));
                        end
                        accuracynum = 0;
                        RelationAccs = zeros(1,4);
                        for i =1:size(Result,1)
                            if Result(i,4+2+1)==Result(i,4+2+2)
                                accuracynum = accuracynum+1;
                                RelationAccs(Result(i,4+2+2))=RelationAccs(Result(i,4+2+2))+1;
                            end
                        end
                        
                        %                         MSE = sum(((test_num - RelationAccs)./test_num).^2)./length(test_num);
                        % Accuracy = sum(RelationAccs./test_num)./length(test_num);
                        Accuracy = min(RelationAccs./test_num);
                        %                 if(MSE<max_mse)
                        %                     max_num = accuracynum;
                        %                     max_filename = filename;
                        %                     max_iiiii=RelationAccs;
                        %                     max_mse = MSE;
                        %                 end
                        if(Accuracy > ci_max)
                            ci_max_num = accuracynum;
                            ci_max_filename = filename;
                            ci_max_iiiii=RelationAccs;
                            ci_max = Accuracy;
                        end
                        fit_row = [MaxIter, rank, lambda1, lambda2, rho, cishu, RelationAccs, Accuracy];
                        fit = [fit;fit_row];
                    end
                    if(ci_max > max_acc)
                        max_num = ci_max_num;
                        max_filename = ci_max_filename;
                        max_iiiii=ci_max_iiiii;
                        max_acc = ci_max;
                    end
                    
                    disp('the number of right relations is:');
                    disp(ci_max_filename);
                    disp(ci_max_num);
                    disp(ci_max_iiiii);
                    disp(['Accuracy:  ' num2str(ci_max)]);
                    disp('the number of right relations is:');
                    disp(max_num);
                    disp(max_filename);
                    disp(max_iiiii);
                    disp(['Accuracy:  ' num2str(max_acc)]);
                    disp('-----------------------------------------------------------------------------------------');
                    save ResultTest Result;
                    disp('end');
                    fclose(fidresult_out);
                    
                end
            end
        end
    end
end
save fit fit
end
