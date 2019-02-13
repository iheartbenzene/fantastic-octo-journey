clear; clc; close all;

avg_work_hours_year = 40 * 52;

% load work_hours.txt
hours = load('work_hours.txt');

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
% gross_income_initial = rates * 2 * (5*unique(hours(2,:))); 

% scaled income to show the available funds
% as a value which is lagging behind by a week
if gcd(21, length(hours)) == 21,
  total_gross_income = rates * (2 * (5*unique(hours(2,1))) + (a-5*unique(hours(2,1))));
else,
  total_gross_income = rates * 2 * (5*unique(hours(2,1)));
endif

% possible comparison to display for the total span of 120 hours worked
% before being paid for prior 80 via bi-weekly pay periods
% total_gross_income = gross_income0 + rates * (a - 80);

%compute future paychecks based on hours worked.


%projected annual income
proj_ann_income = avg_work_hours_year .* rates;

%TODO add tax bracket vector
%TODO at some point turn the vector into a function
proj_ann_income = proj_ann_income - 0.22 * proj_ann_income

%compute net income
% net_income0 = (gross_income0 - (0.22 * gross_income0));
net_income1 = total_gross_income - 0.22 * total_gross_income;

%compute remaining income
bills_1 = load('bills_1.txt');
expenses = sum(bills_1);
% rem_income0 = net_income0 - 247 - 165.69;
rem_income = net_income1 - expenses

%compute annual income remaining
rem_ann_income = proj_ann_income - rem_income

% Disparity
disparity = (proj_ann_income - rem_income) / proj_ann_income