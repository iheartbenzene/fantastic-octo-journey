avg_work_hours_year = 40 * 52;

% load work_hours.txt
hours = load('work_hours.txt');

%set initial working hours
% A = [zeros(size(hours)), hours];
a = sum(hours);

%set hourly rate
rates = load('hourly_rate.txt');

%load deductions
deduction = load('deductions.txt');

%compute gross income
gross_income0 = rates * 80;
total_gross_income = gross_income0 + rates * a;

%projected annual income
proj_ann_income = avg_work_hours_year * rates;
proj_ann_income = proj_ann_income - 0.22 * proj_ann_income

%compute net income
net_income1 = total_gross_income - 0.22 * total_gross_income;

%compute remaining income
data = load('bills_1.txt');
expenses = sum(data);
rem_income = net_income1 - expenses

%compute annual income remaining
rem_ann_income = proj_ann_income - rem_income
