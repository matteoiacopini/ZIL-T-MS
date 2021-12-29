function newmap = bwr(m)
%% bwr Returns a M-by-3 matrix containing red-white-blue colormap (corresponding to (+ 0 -) values).
%
% Useful for images and surface plots with positive and negative values. Scale:
%   x > 0 --> red
%   x = 0 --> white
%   x < 0 --> blue
%
% Examples:
%   figure;   imagesc(peaks(250));  colormap(bluewhitered(256));   colorbar
%   figure;   imagesc(peaks(250), [0 8]);  colormap(bluewhitered);   colorbar
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   if nargin < 1
      m = size(get(gcf,'colormap'),1);
   end

   bottom = [0 0 0.5];
   botmiddle = [0 0.5 1];
   middle = [1 1 1];
   topmiddle = [1 0 0];
   top = [0.5 0 0];

   % Find middle
   lims = get(gca,'CLim');

   % Find ratio of negative to positive
   if (lims(1) < 0) && (lims(2) > 0)
       % It has both negative and positive --> Find ratio of negative to positive
       ratio = abs(lims(1)) / (abs(lims(1)) + lims(2));
       neglen = round(m*ratio);
       poslen = m - neglen;

       % Just negative
       new = [bottom; botmiddle; middle];
       len = length(new);
       oldsteps = linspace(0, 1, len);
       newsteps = linspace(0, 1, neglen);
       newmap1 = zeros(neglen, 3);

       for i=1:3
           % Interpolate over RGB spaces of colormap
           newmap1(:,i) = min(max(interp1(oldsteps, new(:,i), newsteps)', 0), 1);
       end

       % Just positive
       new = [middle; topmiddle; top];
       len = length(new);
       oldsteps = linspace(0, 1, len);
       newsteps = linspace(0, 1, poslen);
       newmap = zeros(poslen, 3);

       for i=1:3
           % Interpolate over RGB spaces of colormap
           newmap(:,i) = min(max(interp1(oldsteps, new(:,i), newsteps)', 0), 1);
       end

       % And put them together
       newmap = [newmap1; newmap];

   elseif lims(1) >= 0
       % Just positive
       new = [middle; topmiddle; top];
       len = length(new);
       oldsteps = linspace(0, 1, len);
       newsteps = linspace(0, 1, m);
       newmap = zeros(m, 3);

       for i=1:3
           % Interpolate over RGB spaces of colormap
           newmap(:,i) = min(max(interp1(oldsteps, new(:,i), newsteps)', 0), 1);
       end

   else
       % Just negative
       new = [bottom; botmiddle; middle];
       len = length(new);
       oldsteps = linspace(0, 1, len);
       newsteps = linspace(0, 1, m);
       newmap = zeros(m, 3);

       for i=1:3
           % Interpolate over RGB spaces of colormap
           newmap(:,i) = min(max(interp1(oldsteps, new(:,i), newsteps)', 0), 1);
       end

   end
end

% m = 64;
% new = [bottom; botmiddle; middle; topmiddle; top];
% % x = 1:m;
% 
% oldsteps = linspace(0, 1, 5);
% newsteps = linspace(0, 1, m);
% newmap = zeros(m, 3);
% 
% for i=1:3
%     % Interpolate over RGB spaces of colormap
%     newmap(:,i) = min(max(interp1(oldsteps, new(:,i), newsteps)', 0), 1);
% end
% 
% % set(gcf, 'colormap', newmap), colorbar
