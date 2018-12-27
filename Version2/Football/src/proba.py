import numpy as np
from scipy.stats import poisson
from math import floor

def proba_score(score, average_hgoals, average_agoals):
    home_goals = score[0]
    away_goals = score[1]
    return poisson(average_hgoals).pmf(home_goals) * poisson(average_agoals).pmf(away_goals)

def gen_score(nb_goals):
    n = nb_goals
    if n == 0:
        return [(0, 0),]
    S = []
    t = floor(n/2.)
    for i in range(t + 1):
        S.append((n-i, i))
        if i != n-i:
            S.append((i, n-i))
    return S

def form(outcomes, n_last_games):
    l = len(outcomes)
    if l < n_last_games:
        n_last_games = l
    if n_last_games == 2:
        return (0.4 * float(outcomes[-2]) + 0.6 * float(outcomes[-1]))/2.
    elif n_last_games == 3:
        return (0.22 * float(outcomes[-3]) + 0.33*float(outcomes[-2]) + 0.44*float(outcomes[-1]))/2.
    elif n_last_games == 4:
        return (0.10 * float(outcomes[-4]) + 0.20 * float(outcomes[-3]) + 0.3 * float(outcomes[-2]) + 0.4 * float(outcomes[-1]))/2.
    elif n_last_games == 5:
        return (0.10 * float(outcomes[-5]) + 0.15 * float(outcomes[-4]) + 0.2 * float(outcomes[-3]) + 0.25 * float(outcomes[-2]) + 0.3*float(outcomes[-1]))/2.
    else:
        return 1

h = form([1., 1., 2., 2.], 5)
a = form([0., 1., 1.], 5)

def get_form_proba(h_form, o_form):
    p_draw = 1 - 0.5*(h_form + o_form)
    p_win = 0.5*(1 - p_draw)
    p_loose = 0.5*(1 - p_draw)
    d = 0.5*(1 - p_win)*abs(h_form - o_form)
    if h_form > o_form:
        p_win += d
        p_loose -= d
    else:
        p_win -= d
        p_loose += d
    return float(p_win), float(p_draw), float(p_loose)

print(get_form_proba(h, a))

def strength(average_team_goals, average_goals):
    return average_team_goals / average_goals

def goals(att_strength, def_strength, average_allgoals):
    return att_strength * def_strength * average_allgoals

def outcome(average_home_sgoals, average_home_allsgoals, average_away_sgoals, average_away_allsgoals,
            average_home_tgoals, average_away_tgoals):

    home_att_strength = strength(average_home_sgoals, average_home_allsgoals)
    away_att_strength = strength(average_away_sgoals, average_away_allsgoals)

    home_def_strength = strength(average_home_tgoals, average_away_allsgoals)
    away_def_strength = strength(average_away_tgoals, average_home_allsgoals)

    home_goals = goals(home_att_strength, away_def_strength, average_home_allsgoals)
    away_goals = goals(away_att_strength, home_def_strength, average_away_allsgoals)
    p_win = 0.
    p_draw = 0.
    p_lost = 0.
    for g in range(9):
        l = gen_score(g)
        for score in l:
            q = proba_score(score, home_goals, away_goals)
            if (score[0] > score[1]):
                p_win += q
            elif(score[0] == score[1]):
                p_draw += q
            else:
                p_lost += q
    return (round(p_win, 4), round(p_draw, 4), round(p_lost, 4))

