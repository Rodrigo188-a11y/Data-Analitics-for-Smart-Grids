%Lab1 DASG
clear all;
format compact;

%Comsumption Data (Given)
 X = [0.332 0.064 0.084 0.12 
 0.236 0.164 0.276 0.064
 0.224 0.708 1.572 0.072
 0.36  3.44  1.188 0.18 
 1.332 2.176 0.484 1.464
 1.516 3.02  0.316 0.624
 0.92  0.916 0.404 2.772
 0.752 0.64  0.396 1.464
 1.828 0.684 0.576 0.576
 3.568 0.564 0.828 0.428
 0.78  0.356 0.728 0.348
 0.856 0.22  0.308 0.12];

  X12 = [0.332 0.064 0.084 0.12  1.0609 0.1520 0.9477 0.9274 0.5812 0.1346 0.4788 0.2631  
         0.236 0.164 0.276 0.064 0.6849 2.5335 0.8298 0.8098 0.9927 1.0933 1.0244 0.4628
         0.224 0.708 1.572 0.072 0.9509 1.1034 3.2552 0.7877 0.9883 2.0290 0.5621 0.1456
         0.36  3.44  1.188 0.18  0.5942 0.3003 0.1526 0.7399 0.7549 1.8059 1.0372 0.9655
         1.332 2.176 0.484 1.464 1.1496 1.1724 0.9721 0.9783 0.8662 1.1529 0.4204 0.4464
         1.516 3.02  0.316 0.624 1.8755 1.5012 0.6790 2.3537 2.2031 0.6837 1.4170 1.4787
         0.92  0.916 0.404 2.772 0.8226 0.7777 0.8662 0.7023 0.8426 0.3046 0.4072 0.7127
         0.752 0.64  0.396 1.464 0.8972 1.0445 0.2508 0.8662 0.8661 0.4205 0.8533 1.1063
         1.828 0.684 0.576 0.576 1.4815 1.7483 1.6594 1.9279 2.3501 2.0913 1.5790 1.6951
         3.568 0.564 0.828 0.428 1.6831 0.6445 0.8788 1.7808 1.8698 1.9777 1.6707 2.3301
         0.78  0.356 0.728 0.348 3.9789 0.4418 0.4646 0.1779 0.4116 0.5831 0.8912 0.8763
         0.856 0.22  0.308 0.12  0.5107 2.8695 0.2129 0.3998 0.7125 0.8951 1.4301 0.8411];

 %Atribute comsumers to phase: abca
  beta_orig = [1 0 0
              0 1 0
              0 0 1
              1 0 0];

 beta_orig12 = [1 0 0
                0 1 0
                0 0 1
                1 0 0
                1 0 0
                0 1 0
                0 0 1
                0 1 0
                1 0 0
                0 1 0
                0 0 1
                1 0 0];
 

 %Consumers aggregation by phase and noise inclusion
 Y = zeros(3,1);
 Y12 = zeros(3,1);
 var = 0.25^2;

 for k = 1:12
    Y(k,1) = X(k,1) + X(k,4) + normrnd(0,var);
    Y(k,2) = X(k,2) + normrnd(0,var);
    Y(k,3) = X(k,3) + normrnd(0,var);
 end

  for k = 1:12
    Y12(k,1) = X12(k,1) + X12(k,4) + X12(k,5) + X12(k,9) + X12(k,12) + normrnd(0,var);
    Y12(k,2) = X12(k,2) + X12(k,6) + X12(k,8) + X12(k,10) + normrnd(0,var);
    Y12(k,3) = X12(k,3) + X12(k,7) + X12(k,11) + normrnd(0,var);
 end

 %Multivariate Regression
 B = (X.'*X)^-1*X.'*Y;
 B12 = (X12.'*X12)^-1*X12.'*Y12;

 %Mapping
 beta = zeros(4,3);
 beta12 = zeros(12,3);

 for k = 1:4
    if B(k,1)>B(k,2)
        if B(k,1)>B(k,3)
            beta(k,1) = 1;
        else
            beta(k,3) = 1;
        end
    elseif B(k,2)>B(k,3)
            beta(k,2) = 1;
        else
            beta(k,3) = 1;
    end
 end

%Error count
err = 0;

for k = 1:4
    if beta(k,1) ~= beta_orig(k,1) || beta(k,2) ~= beta_orig(k,2) || beta(k,3) ~= beta_orig(k,3)
        err = err + 1;
    end
end


%Plot LV/MV meter information
figure()
ep=0:11;
stairs(ep,Y12)
xlabel("Epoch - 15 min time stamp")
ylabel("Power comsumption - kwh")
axis([0 11 0 10.5])
legend({'phase a','phase b','phase c'},'Location','northwest')

%Plot smartmeter information
figure()
ep=0:11;
stairs(ep,X12)
xlabel("Epoch - 15 min time stamp")
ylabel("Power comsumption - kwh")
axis([0 11 0 4.3])

