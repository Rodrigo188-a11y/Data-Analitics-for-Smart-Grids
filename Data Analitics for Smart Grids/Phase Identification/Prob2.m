%Comsumption Data (Given)
a=1;
b=1;
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

 %Atribute comsumers to phase: abca
 beta_orig = [1 0 0
              0 1 0
              0 0 1
              1 0 0];
 

 %Consumers aggregation by phase and noise inclusion
 Y = zeros(3,1);
 Y2 = zeros(3,1);
 Y3 = zeros(3,1);
 Y4 = zeros(3,1);
 var = 0.25^2;
 trifasico = 20;
 for kkk = 1:100 
        err_aux = 0;
        
        for kk = 1:80
             for k = 1:12
                Y(k,1) = X(k,1) + X(k,4) + normrnd(0,var) + trifasico*0.2 - trifasico/3;
                Y(k,2) = X(k,2) + normrnd(0,var) + trifasico*0.4 - trifasico/3;
                Y(k,3) = X(k,3) + normrnd(0,var) + trifasico*0.4 - trifasico/3;
             end
            
             %Multivariate Regression
             B = (X.'*X)^-1*X.'*Y;
             e = cov(B);
             %Mapping
             beta = zeros(4,3);
            
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

            if(err>0)
            err_aux = err_aux+1;
            end
        end

        err_record(kkk) = err_aux/80;
        var_record(kkk) = a;
        a = a*1.0001;
        b=b+1;
        disp('aaaaaaaaaaaa')
        disp(X)
 end

 for k =1:100
compare(1,k) = err_record(k)
 end


%Plot probability info
figure()
scatter(var_record,compare)
xlabel("Variance")
ylabel("Error probability")

legend({'o=0.25'},'Location','southwest')
 


