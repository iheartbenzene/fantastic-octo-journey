clear; clc; close all;
format short

avg_work_hours_year = 40 * 52;

% load work_hours.txt
hours = load('work_hours.txt');
hr_per_week = load('work_week.txt');

%set initial working hours
% A = [zeros(size(hours)), hours];
a = sum(hours);

%set hourly rate
rates = load('hourly_rate.txt');
% rate = [zeros(size(rates)), rates];

%load deductions
deduction = load('deductions.txt');

% compute gross income
% set an initial value to match most recent paycheck

% scaled income to show the available funds
% as a value which is lagging behind by a week

wage = (hours + hr_per_week - 40) .* rates;

total_gross_income = sum(wage) - 40 * rates;

%projected annual income
proj_ann_income = avg_work_hours_year .* rates;

%TODO add tax bracket vector
%TODO at some point turn the vector into a function

proj_ann_income = proj_ann_income - 0.22 * proj_ann_income

%compute net income
net_income1 = total_gross_income - 0.22 * total_gross_income;

%compute remaining income
bills_1 = load('bills_1.txt');
expenses = sum(bills_1);
rem_income = net_income1 - expenses

%compute annual income remaining
rem_ann_income = proj_ann_income - rem_income

i = [1:length(bills_1)]';
m = sum((i .- mean(i)) .* bills_1) / sum((i .- mean(i)) .^ 2);
c = mean(bills_1) - m * mean(i);

% Disparity
despare_ity = (proj_ann_income - rem_income) / proj_ann_income

% plotData
hold on;
 plot(m * i + c);
 plot(i, bills_1, 'rx', 'MarkerSize', 5);
 plot(i, bills_1);
 xlabel('day number');
 ylabel('amount spent per day');
hold off;
