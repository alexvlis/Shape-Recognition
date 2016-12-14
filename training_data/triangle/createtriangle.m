size = 100;
for i = 1:50
   R = randi([30 40]);
   x = randi([45 55]);
   y = randi([45 55]);
   
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