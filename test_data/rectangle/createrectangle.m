imsize = 100;
for i = 1:12
    img = ones(imsize) * 255;
   
    w = randi([30 40]);
    h = randi([30 40]);
    
    x = 30;
    y = 30;
    
    img(x:x+h, y:y+w) = 0;
    
    imwrite(img, strcat('rectangle-', num2str(i), '.png'));
end