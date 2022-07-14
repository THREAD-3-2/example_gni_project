% An example for Forward-Euler scheme
clear all;
clc;

% parameters
T = 1.5;
x0=2.0/3.0;
dt = 0.1;

% solve x using Forward-Euler scheme
[t, x]=Forward_Euler(x0, dt, T);

% plot the results
plot(t,x);