function omega = similaritysubject(P,subjectplace,singlewordnum) 
    omega = zeros(singlewordnum,1);
    for i =1:singlewordnum
        omega(i)=(subjectplace-1)*singlewordnum+i;
    end
    omega = int64(omega);
end