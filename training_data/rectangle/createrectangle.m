imsize = 100;
for i = 1:50
    img = ones(imsize) * 255;
   
    w = randi([20 50]);
    h = randi([20 50]);
    
    x = randi([20 50]);
    y = randi([20 50]);
    
    img(x:x+h, y:y+w) = 0;
    
    imwrite(img, strcat('rectangle-', num2str(i), '.png'));
end