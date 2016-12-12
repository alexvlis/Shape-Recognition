size = 100;
for i = 1:200
   img = ones(size) * 255;
   R = randi(size/2);
   x = randi(size-R);
   y = randi(size-R);
   
   for r = 1:R
       for theta = 0:0.01:2*pi
          p = round(x + r * cos(theta));
          n = round(y + r * sin(theta));

          if p <= 0
              p = 1;
          end
          if n <= 0
              n = 1;
          end
          img(p, n) = 0;
       end
   end
   img(x, y) = 0;
   imwrite(img, strcat('circle-', num2str(i), '.bmp'))
end