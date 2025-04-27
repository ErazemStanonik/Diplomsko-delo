function [X,Y] = synthetic_init_RGB(sample_num, img_size, background, random)
% We will generate RGB squares, triangles and circles.
% 
% sample num tells us number of observations
%
% img_size tells us that each sample will be a matrix of this size
%
% mode is either 'black' or 'white' and tells us the color of background
%
% random tells us if sizes nad positions of shapes should be random or not

if strcmp(background, 'black')
    bg = zeros(img_size, img_size);
    val = 1;
elseif strcmp(background, 'white')
    bg = ones(img_size, img_size);
    val = -1;
else
    assert(0, 'mode should be either "black" or "white"');
end

assert(mod(img_size,8) == 0, 'Parameter img_size should be an divisible by 8');

X = tensor(zeros(img_size, img_size, 3, 2*sample_num));
Y = zeros(2*sample_num,1);

% Square
% for i = 1:sample_num
%     img = zeros(img_size);
%     if random
%         % square sizes will vary from 4/16 to 9/16 of img
%         square_size = randi([img_size/2,img_size/2+img_size/4]);
%         % we also choose x and y position as random
%         x = randi([1,img_size-square_size+1]);
%         y = randi([1,img_size-square_size+1]);
%     else
%         square_size = 3*img_size/4;
%         x = img_size / 8;
%         y = img_size / 8;
%     end
%     % fill the square
%     img(y:y+square_size-1, x:x+square_size-1) = 1;
%     for c=1:3
%         X(:,:,c,i) = bg + val*(img * rand);
%     end
%     Y(i) = 1;
% end

% Triangle
for j = 1:sample_num %j = sample_num+1:2*sample_num
    img = zeros(img_size);
    if random
        % triangle sizes will vary from 1/16 to 1/4 of img
        triangle_size = randi([img_size/4,img_size/2+img_size/4]);
        % we also choose x and y position as random
        x = randi([1,img_size-triangle_size+1]);
        y = randi([1,img_size-triangle_size+1]);
    % we now fill the triangle. Triangle will be lower left.
    else
        triangle_size = 3*img_size/4;
        x = img_size / 8;
        y = img_size / 8;
    end
    for k = 1:triangle_size
        img(y+k-1,x:x+k-1) = 1;
    end
    for c=1:3
        X(:,:,c,j) = bg + val*(img * rand);
    end
    Y(j) = 1;
end

% Circle
for l = sample_num+1:2*sample_num%2*sample_num+1:3*sample_num
    img = zeros(img_size);
    if random
        radius = randi([2*img_size/8,3*img_size/8]);
        x = randi([radius+1,img_size-radius-1]);
        y = randi([radius+1,img_size-radius-1]);
    else
        radius = 3*img_size/8;
        x = img_size / 2;
        y = img_size / 2;
    end
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
    for c=1:3
        X(:,:,c,l) = bg + val*(img * rand);
    end
    Y(l) = -1;%0;
end
    
% now we remix the order in X and Y so we can use cross-validation.
order = randperm(2*sample_num);
X = X(:,:,:,order);
Y = Y(order);