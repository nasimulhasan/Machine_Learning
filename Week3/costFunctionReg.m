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

 %find the logrithm of the hypothesis
    log_h = log(sigmoid(X * theta));
    log_h_1 = log(1 - (sigmoid(X * theta)));
    
    %regularization 
    %reg = lambda / 2. / m * ( theta' * theta - theta(1)^2 );
    
    %cost of the particular choice of theta
    J = (1 / m) * sum((-((y') * log_h)) - ((1 - (y')) * log_h_1)) + ...
        lambda / 2. / m * ( theta' * theta - theta(1)^2 );

    %gradient 
    mask = ones(size(theta));
    mask(1) = 0;
    
    %reg_grad = lambda * (theta .* mask)/ m;
    
    grad = (X' * (sigmoid(X * theta) - y)) * (1 / m) + lambda * (theta .* mask)/ m;

% 
%     J = 1./m * ( -y' * log( sigmoid(X * theta) ) - ( 1 - y' ) * log ( 1 - sigmoid( X * theta)) ) + ...
%         lambda / 2. / m * ( theta' * theta - theta(1)^2 );
%     mask = ones(size(theta));
%     mask(1) = 0;
%     grad = 1./m * X' * (sigmoid(X * theta) - y) + lambda * (theta .* mask)/ m;

% =============================================================

end
