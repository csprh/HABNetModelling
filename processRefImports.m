clear all
close all
load refImports

refImports;
Days = 10; Mods = 12;
totSize = length(refImports);
lengthOfPortion = totSize/Days;

for ii = 1: Days
    thisPortion = refImports(1+lengthOfPortion*(ii-1): lengthOfPortion*(ii));
    portSize = length(thisPortion);
    lengthOfPortionP = portSize/Mods;
    for iii = 1: Mods
        thisPortionP(iii,ii) =  mean(thisPortion(1+lengthOfPortionP*(iii-1): lengthOfPortionP*(iii)));
    end
    hold on;
    
end
surface(thisPortionP);