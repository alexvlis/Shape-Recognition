size = 100;
for i = 1:12
   R = randi([20 40]);
   x = randi([40 60]);
   y = randi([40 60]);
   
   %thetas = randi([1 12], [1, 3]);
   thetas = [0, pi/2, pi];
   p = round(x + R * cos(thetas));
   n = round(y + R * sin(thetas));
   
   figure('Position', [100, 100, 100, 100], 'Color', 'w', 'Visible', 'off');
   fill(p, n, 'k')
   axis off
   f = getframe(gcf);
   
   imwrite(f.cdata, strcat('triangle-', num2str(i), '.png'))
   close all
end