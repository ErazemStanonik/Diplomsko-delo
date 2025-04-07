% This is the main setup for the testing on the systhetic Data
% We will generate squares, triangles and circles.

sample_num = 15;    % each class has this many observations
img_size = 32;      % img will be a matrix of size x size

X = zeros(img_size, img_size, 3*sample_num);
Y = zeros(2*sample_num,1);

% Square
% for i = 1:sample_num
%     img = zeros(img_size);
%     % square sizes will vary from 1/16 to 9/16 of img
%     square_size = randi([img_size/4,img_size/2+img_size/4]);
%     % we also choose x and y position as random
%     x = randi([1,img_size-square_size+1]);
%     y = randi([1,img_size-square_size+1]);
%     % fill the square
%     img(y:y+square_size-1, x:x+square_size-1) = 1;
%     X(:,:,i) = img;
%     Y(i) = 1;
% end

% Triangle
for j = 1:sample_num %j = sample_num+1:2*sample_num
    img = zeros(img_size);
    % triangle sizes will vary from 1/16 to 1/4 of img
    triangle_size = randi([img_size/4,img_size/2+img_size/4]);
    % we also choose x and y position as random
    x = randi([1,img_size-triangle_size+1]);
    y = randi([1,img_size-triangle_size+1]);
    % we now fill the triangle. Triangle will be lower left.
    for k = 1:triangle_size
        img(y+k-1,x:x+k-1) = 1;
    end
    X(:,:,j) = img;
    Y(j) = -1;
end

% Circle
for l = sample_num+1:2*sample_num%2*sample_num+1:3*sample_num
    img = zeros(img_size);
    radius = randi([img_size/4,3*img_size/8]);
    x = randi([radius+1,img_size-radius-1]);
    y = randi([radius+1,img_size-radius-1]);
    % we draw a circle
    for yi = y-radius:y+radius
        % we go from top to bottom and in each step the x-length is
        % sin(alpha). We get alpha as arccos(percentage_of_r)
        p = abs(yi-y)/radius;
        alpha = acos(p);
        p = round(sin(alpha) * radius);
        for xi = x-p:x+p
            img(yi,xi) = 1;
        end
    end
    X(:,:,l) = img;
    Y(l) = 1;%0;
end
    
% now we remix the order in X and Y so we can use cross-validation.
order = randperm(2*sample_num);
X = X(:,:,order);
Y = Y(order);

% now we check classification accuracy
k = 3;       % NOTE: PLEASE CHOOSE K, SUCH THAT IT CAN DEVIDE THE NUMBER OF ALL SAMPLES
accuracySVM = testSVM(X,Y,k);
accuracySTM = testSTM(X,Y,k);
accuracyHoSTM = testHoSTM(X,Y,k);
accuracySTuM = testSTuM(X,Y,k);