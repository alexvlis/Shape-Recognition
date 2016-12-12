size = 100;
for i = 1:200
   R = randi(size/2);
   x = randi(size-R);
   y = randi(size-R);
   
   points = zeros(3, 2);
   thetas = rand(1, 3) * (2*pi);
   idx = 1;
   for theta = thetas
        p = round(x + R * cos(theta));
        n = round(y + R * sin(theta));
        points(idx, :) = [p, n];
        idx = idx + 1;
   end
   
   figure('Position', [100, 100, 100, 100], 'Color', 'w', 'Visible', 'off');
   fill(points(:, 1), points(:, 2), 'k')
   axis off
   f = getframe(gcf);
   
   imwrite(f.cdata, strcat('triangle-', num2str(i), '.png'))
   close all
end