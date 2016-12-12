imsize = 100;
for i = 1:200
    img = ones(imsize) * 255;
   
    w = randi(imsize - 1);
    h = randi(imsize - 1);
    
    x = randi(imsize-h);
    y = randi(imsize-w);
    
    img(x:x+h, y:y+w) = 0;
    
    imwrite(img, strcat('rectangle-', num2str(i), '.bmp'));
end