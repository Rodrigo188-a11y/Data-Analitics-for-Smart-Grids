%Comsumption Data (Given)
%{
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
%}
%{
 X = [0.332 0.064 0.084 0.084
 0.236 0.164 0.276 0.276
 0.224 0.708 1.572 1.572
 0.36  3.44  1.188 1.188
 1.332 2.176 0.484 0.484
 1.516 3.02  0.316 0.316
 0.92  0.916 0.404 0.404
 0.752 0.64  0.396 0.396
 1.828 0.684 0.576 0.576
 3.568 0.564 0.828 0.828
 0.78  0.356 0.728 0.728
 0.856 0.22  0.308 0.308];
%}
b=1;
a=1;
%{
X = [0.332 0.064 0.084 0.333*a 
 0.236 0.164 0.276 0.237*a
 0.224 0.708 1.572 0.223*a
 0.36  3.44  1.188 0.34*a 
 1.332 2.176 0.484 1.335*a
 1.516 3.02  0.316 1.517*a
 0.92  0.916 0.404 0.90*a
 0.752 0.64  0.396 0.748*a
 1.828 0.684 0.576 1.824*a
 3.568 0.564 0.828 3.571*a
 0.78  0.356 0.728 0.80*a
 0.856 0.22  0.308 0.856*a];
%}
X = [0.332 0.064 0.084 0.332 
 0.236 0.164 0.276 0.236
 0.224 0.708 1.572 0.224
 0.36  3.44  1.188 0.36
 1.332 2.176 0.484 1.332
 1.516 3.02  0.316 1.516
 0.92  0.916 0.404 0.92
 0.752 0.64  0.396 0.752
 1.828 0.684 0.576 1.828
 3.568 0.564 0.828 3.568
 0.78  0.356 0.728 0.78
 0.856 0.22  0.308 0.856];
 %}
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
 var2 = 0.5^2;
 var3 = 0.75^2;
 var4 = 1;

 dist = sqrt(X(1,4)^2+X(2,4)^2+X(3,4)^2+X(5,4)^2+X(6,4)^2+X(7,4)^2+X(8,4)^2+X(9,4)^2+X(10,4)^2+X(11,4)^2+X(12,4)^2);
 
 for kkk = 1:100
        err_aux = 0;
        err_aux2 = 0;
        err_aux3 = 0;
        err_aux4 = 0;
        X(1:6,4) = X(1:6,4)*a;
        X(6:12,4) = X(6:12,4)*a*1.0002;
        for kk = 1:80
             for k = 1:12
                Y(k,1) = X(k,1) + X(k,4) + normrnd(0,var);
                Y(k,2) = X(k,2) + normrnd(0,var);
                Y(k,3) = X(k,3) + normrnd(0,var);
                Y2(k,1) = X(k,1) + X(k,4) + normrnd(0,var2);
                Y2(k,2) = X(k,2) + normrnd(0,var2);
                Y2(k,3) = X(k,3) + normrnd(0,var2);
                Y3(k,1) = X(k,1) + X(k,4) + normrnd(0,var3);
                Y3(k,2) = X(k,2) + normrnd(0,var3);
                Y3(k,3) = X(k,3) + normrnd(0,var3);
                Y4(k,1) = X(k,1) + X(k,4) + normrnd(0,var4);
                Y4(k,2) = X(k,2) + normrnd(0,var4);
                Y4(k,3) = X(k,3) + normrnd(0,var4);
             end
            
             %Multivariate Regression
             B = (X.'*X)^-1*X.'*Y;
             B2 = (X.'*X)^-1*X.'*Y2;
             B3 = (X.'*X)^-1*X.'*Y3;
             B4 = (X.'*X)^-1*X.'*Y4;

            
             %Mapping
             beta = zeros(4,3);
             beta2 = zeros(4,3);
             beta3 = zeros(4,3);
             beta4 = zeros(4,3);
            
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
             for k = 1:4
                if B2(k,1)>B2(k,2)
                    if B2(k,1)>B2(k,3)
                        beta2(k,1) = 1;
                    else
                        beta2(k,3) = 1;
                    end
                elseif B2(k,2)>B2(k,3)
                        beta2(k,2) = 1;
                    else
                        beta2(k,3) = 1;
                end 
             end
             for k = 1:4
                if B3(k,1)>B3(k,2)
                    if B3(k,1)>B3(k,3)
                        beta3(k,1) = 1;
                    else
                        beta3(k,3) = 1;
                    end
                elseif B3(k,2)>B3(k,3)
                        beta3(k,2) = 1;
                    else
                        beta3(k,3) = 1;
                end
             end
             for k = 1:4
                if B4(k,1)>B4(k,2)
                    if B4(k,1)>B4(k,3)
                        beta4(k,1) = 1;
                    else
                        beta4(k,3) = 1;
                    end
                elseif B4(k,2)>B4(k,3)
                        beta4(k,2) = 1;
                    else
                        beta4(k,3) = 1;
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

            %Error count 2
            err2 = 0;
            
            for k = 1:4
                if beta2(k,1) ~= beta_orig(k,1) || beta2(k,2) ~= beta_orig(k,2) || beta2(k,3) ~= beta_orig(k,3)
                    err2 = err2 + 1;
                end
            end

            if(err2>0)
            err_aux2 = err_aux2+1;
            end

            %Error count 3
            err3 = 0;
            
            for k = 1:4
                if beta3(k,1) ~= beta_orig(k,1) || beta3(k,2) ~= beta_orig(k,2) || beta3(k,3) ~= beta_orig(k,3)
                    err3 = err3 + 1;
                end
            end

            if(err3>0)
            err_aux3 = err_aux3+1;
            end

            %Error count 4
            err4 = 0;
            
            for k = 1:4
                if beta4(k,1) ~= beta_orig(k,1) || beta4(k,2) ~= beta_orig(k,2) || beta4(k,3) ~= beta_orig(k,3)
                    err4 = err4 + 1;
                end
            end

            if(err4>0)
            err_aux4 = err_aux4+1;
            end
        end

        err_record(kkk) = err_aux/80;
        err_record2(kkk) = err_aux2/80;
        err_record3(kkk) = err_aux3/80;
        err_record4(kkk) = err_aux4/80;
        var_record(kkk) = b;
        a = a*1.0001;
        b=b+1;
        disp('aaaaaaaaaaaa')
        disp(X)
 end

 for k =1:100
compare(1,k) = err_record(k)
compare(2,k) = err_record2(k)
compare(3,k) = err_record3(k)
compare(4,k) = err_record4(k)
end


%Plot probability info
figure()
scatter(var_record,compare)
xlabel("Epoch - 0.7% increase in meter readings")
ylabel("Error probability")

legend({'o=0.25','o=0.5', 'o=0.75', 'o=1'},'Location','southwest')
 


