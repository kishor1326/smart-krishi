function h = healthScore(pHealthy, lesionFrac)
% PROTOTYPE "Leaf Health Index" (0-100). NOT an agronomic standard.
% Assumptions (all unvalidated, state them to judges):
%   * 30% visibly damaged leaf area is treated as "fully damaged" (damage = min(1, frac/0.30))
%   * weights 0.6 (visible damage) and 0.4 (model's non-healthy probability) are chosen, not fitted
damage = min(1, lesionFrac/0.30);
pDis   = 1 - pHealthy;
h.score = round(100*(1 - 0.6*damage - 0.4*pDis));
h.score = max(0, min(100, h.score));
if     h.score >= 80, h.band = "Good";
elseif h.score >= 60, h.band = "Watch";
elseif h.score >= 40, h.band = "Moderate risk";
else                  h.band = "High risk"; end
if     lesionFrac < 0.05, h.severity = "low";
elseif lesionFrac < 0.15, h.severity = "moderate";
else                      h.severity = "high"; end
h.damageFraction = lesionFrac; h.pDisease = pDis;
end
