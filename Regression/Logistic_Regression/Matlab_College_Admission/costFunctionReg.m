function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
 z = X*theta;
 h = sigmoid(z);
 THETA = theta(2:end);%exclude the first element in the theta vector
 R = (lambda/(2*m))*sum(THETA.^2);
 %regularized cost function:
 J = (-(1/m))*sum((y.*log(h)) + ((1-y).*log(1-h))) + R;
    
 grad(1) = (1/m)*sum((h-y));
 for i=2:size(grad)
     grad(i) = ((1/m)*sum((h-y)'*X(:,i))) ;%+ ((lambda/m)*(THETA(1:end)))) ;%formular found on page 5 of ex pdf
 end
 
 %make new grad i which has been regularized:
 
 gtheta = zeros(size(theta));
 gtheta = [0;(lambda/m).*(THETA)];
 grad = grad + gtheta;
 
% =============================================================

end
