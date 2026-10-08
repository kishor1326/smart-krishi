function out = simulateIrrigation(varargin)
% Daily soil-water bucket (FAO-56-style, simplified). SYNTHETIC weather unless you pass real series.
% Output = SIMULATION RESULT under stated assumptions; savings depend entirely on the baseline schedule.
p = inputParser;
addParameter(p,'days',90); addParameter(p,'rainMult',1); addParameter(p,'irrigMult',1);
addParameter(p,'seed',1); addParameter(p,'fixedDepth',12); addParameter(p,'fixedEvery',3);
addParameter(p,'forecastErr',0.3); addParameter(p,'makePlot',true);
parse(p,varargin{:}); o = p.Results; rng(o.seed); N = o.days; d = 1:N;
et0  = max(2, 4 + sin(2*pi*d/N) + 0.3*randn(1,N));            % mm/day (assumed)
rain = zeros(1,N); wet = rand(1,N) < 0.12;
rain(wet) = -log(rand(1,nnz(wet)))*12*o.rainMult;              % mm (assumed storm model)
Kc   = interp1([1 round(0.25*N) round(0.6*N) N],[0.6 0.8 1.15 0.8], d);  % assumed crop-stage curve
TAW  = 60; fcNoise = randn(1,N);                               % root-zone available water, mm (assumed)
out.smart = runPolicy('smart',N,et0,rain,Kc,TAW,o,fcNoise);
out.fixed = runPolicy('fixed',N,et0,rain,Kc,TAW,o,fcNoise);
out.waterSavingPct = 100*(1 - out.smart.waterApplied/max(out.fixed.waterApplied,eps));
fprintf('Smart: %.0f mm, %d events, %d stress days | Fixed: %.0f mm, %d events, %d stress days | saving vs fixed: %.1f%% (SIMULATION)\n', ...
    out.smart.waterApplied,out.smart.events,out.smart.stressDays,out.fixed.waterApplied,out.fixed.events,out.fixed.stressDays,out.waterSavingPct);
if o.makePlot
    figure('Position',[100 100 1000 500]); yyaxis left; bar(d,rain,'FaceColor',[0.6 0.8 1]); ylabel('Rain (mm)');
    yyaxis right; plot(d,out.smart.S,'g-',d,out.fixed.S,'r-',[1 N],[0.5 0.5]*TAW,'k--'); ylabel('Soil water (mm)');
    legend('Rain','Smart','Fixed','Stress threshold'); xlabel('Day'); title('Irrigation what-if (synthetic weather)'); grid on
end
end

function r = runPolicy(pol,N,et0,rain,Kc,TAW,o,fcNoise)
S = 0.8*TAW; r.S = zeros(1,N); r.irr = zeros(1,N); r.Ks = zeros(1,N); drained = 0;
for t = 1:N
    irr = 0;
    switch pol
        case 'smart'
            fc = rain(min(t+1,N))*max(0,1+o.forecastErr*fcNoise(t));   % noisy next-day rain forecast
            if S < 0.55*TAW && fc < 8, irr = min(25, 0.9*TAW - S)*o.irrigMult; end
        case 'fixed'
            if mod(t-1,o.fixedEvery)==0, irr = o.fixedDepth*o.irrigMult; end
    end
    Ks = min(1, S/(0.5*TAW));
    S  = S + rain(t) + irr - Kc(t)*et0(t)*Ks;
    drained = drained + max(0,S-TAW);
    S = min(max(S,0),TAW);
    r.S(t)=S; r.irr(t)=irr; r.Ks(t)=Ks;
end
r.waterApplied = sum(r.irr); r.drained = drained; r.events = nnz(r.irr);
r.stressDays = sum(r.Ks < 0.8); r.meanKs = mean(r.Ks);
end
