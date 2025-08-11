from scipy.stats import nbinom, poisson
def poisson_sf(k, lam): return 1 - poisson.cdf(k-1, lam)
def nb_params_from_mean_var(m, v):
    if v<=m: v=m+1e-6
    p=m/v; r=m*p/(1-p) if p<1 else 1e6; return r,p
def nb_sf(k, m, v): r,p=nb_params_from_mean_var(m,v); return nbinom.sf(k-1, r, p)
def combine_strengths(team_for, league_for, opp_against, league_against, league_mean, home_factor=1.0):
    att=(team_for/max(1e-9,league_for)); de=(opp_against/max(1e-9,league_against))
    return home_factor*att*de*league_mean
def blended_var_factor(tv, ov, lv, floor=1.0, ceil=3.0):
    vals=[v for v in [tv,ov,lv] if v and v>0]
    if not vals: return 1.1
    g=1.0
    for v in vals: g*=v
    g**=(1.0/len(vals)); return min(max(g,floor),ceil)
def finalize_probability(k, lam, var_factor=1.0, prefer_nb=True):
    var=max(lam*var_factor,1e-6)
    if prefer_nb and var>lam*1.15: return nb_sf(k, lam, var)
    return poisson_sf(k, lam)