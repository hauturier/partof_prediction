function omega = similarityobject(P,objectplace,singlewordnum)
    omega = zeros(singlewordnum,1);
    for i =1:singlewordnum
        omega(i)=(i-1)*singlewordnum+objectplace;
    end
    omega = int64(omega);
end