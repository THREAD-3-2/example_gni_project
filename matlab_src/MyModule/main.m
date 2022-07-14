% An example for Forward-Euler scheme
clear all;
clc;

% parameters
duration = 0.3;
initial_condition = [1 0];
h = 0.1;

vector_field = @(x) [x(2); -x(1)];

% solve x using Forward-Euler scheme
[t, x] = Forward_Euler(vector_field, initial_condition, h, duration);

% plot the results
plot(t, x(1, :));