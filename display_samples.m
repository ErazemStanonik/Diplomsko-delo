function display_samples(X,Y)

m = length(Y);
d = ndims(X)-1;
for i = 1:m
    idx = repmat({':'},1,d);
    imshow(X(idx{:},i).data);
    pause(0.5);
end

end