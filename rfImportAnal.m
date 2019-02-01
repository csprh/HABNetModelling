load rfImports

L = length(rfImports);

L10 = L / 10;
L100 = L /100;
for ii = 1:10
    startL1 = (ii-1)*L10;
    for iii = 1:10
        startL = startL1 + (iii-1)*L100;
        stopL = startL + L100;
        p(ii,iii) = max(rfImports(startL+1:stopL))
    end
end

plot((p(1,:))); hold on;
plot((p(2,:)));
plot((p(3,:)));
plot((p(4,:)));
plot((p(5,:)));
plot((p(6,:)));
plot((p(7,:)));


