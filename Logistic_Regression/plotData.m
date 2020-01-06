function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%

%Since the output values are discrete and taking only values 1 or 0 :
pos = find(y==1);%find the array index form which y == 1
neg = find(y==0);%find the array index for which y == 0

%plot the examples : 
%first plot the positives
%X(pos,1) means row = pos and column = 1 i.e the index where y == 1 
%... as described above i.e pos = 4 and 1 is first column of X and the pair
%is 4,60.18259938620976

plot(X(pos,1),X(pos,2), 'k+', 'LineWidth',2, 'MarkerSize', 7);
%the line above plots a cordinate as the x point and another cordinate as
%the y point i.e X(pos,1) is the x cordinate, X(pos,2) is the y cordinate
%to form plot(x,y)
plot(X(neg,1),X(neg,2),'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 7);







% =========================================================================



hold off;

end
