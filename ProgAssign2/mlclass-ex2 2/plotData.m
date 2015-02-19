function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure


% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%


figure; hold on;
m = length(y);
positives = [];
negatives = [];
for i = 1:m,
     if y(i)==1,
         positives = [positives; X(i,:)];
     else
         negatives = [negatives; X(i,:)];
     end
end

plot(positives(:,1), positives(:,2), 'k+');
plot(negatives(:,1), negatives(:,2), 'ko');

% =========================================================================



hold off;

end
