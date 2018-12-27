import pandas as pd
import numpy as np
from tqdm import tqdm
import proba
from championship import Championship
from team import Team


fields = ['homeTeam',
          'awayTeam',
          'homeGoals',
          'awayGoals',
          'month',
          'pHomeWin1',
          'pDraw1',
          'pAwayWin1',
          'pHomeWin2',
          'pDraw2',
          'pAwayWin2',
          'pHomeWin3',
          'pDraw3',
          'pAwayWin3',
          'homePoints',
          'awayPoints',
          'homeRank',
          'awayRank',
          'homeGlobalRank',
          'awayGlobalRank',
          'homeAccuracy',
          'awayAccuracy',
          'homeEfficiency', 
          'awayEfficiency',        
          'homeVulnerability',
          'awayVulnerability',
          'openningOddHome',
          'openningOddDraw',
          'openningOddAway',
          'closingOddHome',
          'closingOddDraw',
          'closingOddAway']

def features(csv_file, country):
    season = None
    dataframe = pd.read_csv(csv_file)
    champ = Championship(dataframe, season, country)
    meetings = champ.get_meetings()
    X = np.array([])
    for meeting in tqdm(meetings):
        hteam = meeting[0]
        ateam = meeting[1]
        month = int(meeting[2].split('/')[1])
        home_team = Team(hteam, dataframe, season, country)
        away_team = Team(ateam, dataframe, season, country)
        score = champ.get_score(meeting)
        oddHome = champ.get_home_odd(meeting, 'openning')
        oddDraw = champ.get_draw_odd(meeting, 'openning')
        oddAway = champ.get_away_odd(meeting, 'openning')
        cOddHome = champ.get_home_odd(meeting, 'closing')
        cOddDraw = champ.get_draw_odd(meeting, 'closing')
        cOddAway = champ.get_away_odd(meeting, 'closing')
        nb_h_games = home_team.get_nb_games('home', meeting)
        nb_a_games = away_team.get_nb_games('away', meeting)
        nb_games = champ.get_nb_games(meeting)
 
        home_accuracy = home_team.get_average_accuracy('home', 0, nb_h_games)
        away_accuracy = away_team.get_average_accuracy('away', 0, nb_a_games)
        home_efficiency = home_team.get_average_efficiency('home', 0, nb_h_games)
        away_efficiency = away_team.get_average_efficiency('away', 0, nb_a_games)
        home_vulnerability = home_team.get_average_vulnerability('home', 0, nb_h_games)
        away_vulnerability = away_team.get_average_vulnerability('away', 0, nb_a_games)
        average_home_spoints = home_team.get_average_spoints('home', 0, nb_h_games)
        average_home_tpoints = home_team.get_average_tpoints('home', 0, nb_h_games)
        average_away_spoints = away_team.get_average_spoints('away', 0, nb_a_games)
        average_away_tpoints = away_team.get_average_tpoints('away', 0, nb_a_games)
        average_home_allspoints = champ.average_home_allspoints(0, nb_games)
        average_away_allspoints = champ.average_away_allspoints(0, nb_games)

        home_outcomes = home_team.get_outcomes('both', meeting)
        away_outcomes = away_team.get_outcomes('both', meeting)
        home_last_outcomes = home_team.get_outcomes('home', meeting)
        away_last_outcomes = away_team.get_outcomes('away', meeting)
        h_form = proba.form(home_outcomes, 5)
        a_form = proba.form(away_outcomes, 5)
        h_last_form = proba.form(home_last_outcomes, 5)
        a_last_form = proba.form(away_last_outcomes, 5)
        w, d, l = proba.get_form_proba(h_form, a_form)
        wl, dl, ll = proba.get_form_proba(h_last_form, a_last_form)
        h_global_rank = home_team.get_rank('global', meeting)
        a_global_rank = away_team.get_rank('global', meeting)
        h_home_rank = home_team.get_rank('home', meeting)
        a_away_rank = away_team.get_rank('away', meeting)
        h_points = float(home_team.get_points('home', meeting)/((2*nb_h_games) + 1))
        a_points = float(away_team.get_points('away', meeting)/((2*nb_a_games) + 1))
        g = proba.outcome(average_home_spoints, average_home_allspoints, average_away_spoints, 
                            average_away_allspoints, average_home_tpoints, average_away_tpoints)
        
        
        X = np.append(X, np.array([str(hteam), str(ateam), int(score[0]), int(score[1]), int(month), float(g[0]), float(g[1]),
                                   float(g[2]), float(w), float(d), float(l), float(wl), float(dl), float(ll), 
                                   float(h_points),  float(a_points), int(h_home_rank), float(a_away_rank), int(h_global_rank), 
                                   int(a_global_rank), float(home_accuracy), float(away_accuracy), float(home_efficiency), 
                                   float(away_efficiency), float(home_vulnerability),  float(away_vulnerability), float(oddHome), 
                                   float(oddDraw), float(oddAway), float(cOddHome), float(cOddDraw), float(cOddAway)]))

    df = pd.DataFrame(X.reshape(-1, 32), columns=fields)
    df.to_csv(str(csv_file)[:-4]+'_featured2.csv', index=False)

"""
def setOdd(oddH, oddD, oddA, res, case, lbd=1.1):
    bkpH = 1/oddH
    bkpD = 1/oddD
    bkpA = 1/oddA
    psum = 2 - np.sum(np.array([bkpH, bkpD, bkpA]))
    pH = psum * bkpH
    pD = psum * bkpD
    pA = psum * bkpA
    if res == 'H':
        d = lbd*pH - pH
        pD = (pD - 0.5*d)
        pA = (pA - 0.5*d)
        pH = lbd*pH
    if res == 'D':
        d = lbd*pD - pD
        pH = (pH - 0.5*d)
        pA = (pA - 0.5*d)
        pD = lbd*pD
    if res == 'A':
        d = lbd*pA - pA
        pD = (pD - 0.5*d)
        pH = (pH - 0.5*d)
        pA = lbd*pA
    if case == 'H':
        return pH
    elif case == 'D':
        return pD
    return pA
"""

def dataset(csv_file):
    df = pd.read_csv(csv_file)
    df = df.dropna()
    X = df[df.columns[4:-3]].values.reshape(-1, 25)
    closingOdds = df[['closingOddHome', 'closingOddDraw', 'closingOddAway']].values.reshape(-1, 3)
    def result(x):
        if x['homeGoals'] > x['awayGoals']:
            return 1
        elif x['homeGoals'] == x['awayGoals']:
            return 2
        else:
            return 3 
    y = df.apply(lambda x: result(x), axis=1)
    return X, y, closingOdds

if __name__ == '__main__':
    """
    features('../data/serieA/serieA_1213.csv', 'France')
    features('../data/serieA/serieA_1314.csv', 'France')
    features('../data/serieA/serieA_1415.csv', 'France')
    features('../data/serieA/serieA_1516.csv', 'France')
    features('../data/serieA/serieA_1617.csv', 'France')
    features('../data/serieA/serieA_1718.csv', 'France')

    features('../data/ligue1/ligue1_1213.csv', 'France')
    features('../data/ligue1/ligue1_1314.csv', 'France')
    features('../data/ligue1/ligue1_1415.csv', 'France')
    features('../data/ligue1/ligue1_1516.csv', 'France')
    features('../data/ligue1/ligue1_1617.csv', 'France')
    features('../data/ligue1/ligue1_1718.csv', 'France')

    features('../data/liga/liga_1213.csv', 'France')
    features('../data/liga/liga_1314.csv', 'France')
    features('../data/liga/liga_1415.csv', 'France')
    features('../data/liga/liga_1516.csv', 'France')
    features('../data/liga/liga_1617.csv', 'France')
    features('../data/liga/liga_1718.csv', 'France')

    features('../data/premierleague/premierleague_1213.csv', 'France')
    features('../data/premierleague/premierleague_1314.csv', 'France')
    features('../data/premierleague/premierleague_1415.csv', 'France')
    features('../data/premierleague/premierleague_1516.csv', 'France')
    features('../data/premierleague/premierleague_1617.csv', 'France')
    features('../data/premierleague/premierleague_1718.csv', 'France')

    features('../data/bundesliga/bundesliga_1213.csv', 'France')
    features('../data/bundesliga/bundesliga_1314.csv', 'France')
    features('../data/bundesliga/bundesliga_1415.csv', 'France')
    features('../data/bundesliga/bundesliga_1516.csv', 'France')
    features('../data/bundesliga/bundesliga_1617.csv', 'France')
    features('../data/bundesliga/bundesliga_1718.csv', 'France')
    """
    """
    X_13, y_13, odds13 = dataset('../data/serieA/serieA_1213_featured2.csv')
    X_14, y_14, odds14 = dataset('../data/serieA/serieA_1314_featured2.csv')
    X_15, y_15, odds15 = dataset('../data/serieA/serieA_1415_featured2.csv')
    X_16, y_16, odds16 = dataset('../data/serieA/serieA_1516_featured2.csv')
    X_17, y_17, odds17 = dataset('../data/serieA/serieA_1617_featured2.csv')
    X_18, y_18, odds18 = dataset('../data/serieA/serieA_1718_featured2.csv')
    X_sa = np.concatenate((X_13, X_14, X_15, X_16, X_17, X_18))
    y_sa = np.concatenate((y_13, y_14, y_15, y_16, y_17, y_18))
    odds_sa = np.concatenate((odds13, odds14, odds15, odds16, odds17, odds18))
    np.save('../data/NPY_FILES/V2/X_serieA', X_sa)
    np.save('../data/NPY_FILES/V2/y_serieA', y_sa)
    np.save('../data/NPY_FILES/V2/odds_serieA', odds_sa)

    X_13, y_13, odds13 = dataset('../data/ligue1/ligue1_1213_featured2.csv')
    X_14, y_14, odds14 = dataset('../data/ligue1/ligue1_1314_featured2.csv')
    X_15, y_15, odds15 = dataset('../data/ligue1/ligue1_1415_featured2.csv')
    X_16, y_16, odds16 = dataset('../data/ligue1/ligue1_1516_featured2.csv')
    X_17, y_17, odds17 = dataset('../data/ligue1/ligue1_1617_featured2.csv')
    X_18, y_18, odds18 = dataset('../data/ligue1/ligue1_1718_featured2.csv')
    X_l1 = np.concatenate((X_13, X_14, X_15, X_16, X_17, X_18))
    y_l1 = np.concatenate((y_13, y_14, y_15, y_16, y_17, y_18))
    odds_l1 = np.concatenate((odds13, odds14, odds15, odds16, odds17, odds18))
    np.save('../data/NPY_FILES/V2/X_ligue1', X_l1)
    np.save('../data/NPY_FILES/V2/y_ligue1', y_l1)
    np.save('../data/NPY_FILES/V2/odds_ligue1', odds_l1)

    X_13, y_13, odds13 = dataset('../data/liga/liga_1213_featured2.csv')
    X_14, y_14, odds14 = dataset('../data/liga/liga_1314_featured2.csv')
    X_15, y_15, odds15 = dataset('../data/liga/liga_1415_featured2.csv')
    X_16, y_16, odds16 = dataset('../data/liga/liga_1516_featured2.csv')
    X_17, y_17, odds17 = dataset('../data/liga/liga_1617_featured2.csv')
    X_18, y_18, odds18 = dataset('../data/liga/liga_1718_featured2.csv')
    X_la = np.concatenate((X_13, X_14, X_15, X_16, X_17, X_18))
    y_la = np.concatenate((y_13, y_14, y_15, y_16, y_17, y_18))
    odds_la = np.concatenate((odds13, odds14, odds15, odds16, odds17, odds18))
    np.save('../data/NPY_FILES/V2/X_liga', X_la)
    np.save('../data/NPY_FILES/V2/y_liga', y_la)
    np.save('../data/NPY_FILES/V2/odds_liga', odds_la)

    X_13, y_13, odds13 = dataset('../data/premierleague/premierleague_1213_featured2.csv')
    X_14, y_14, odds14 = dataset('../data/premierleague/premierleague_1314_featured2.csv')
    X_15, y_15, odds15 = dataset('../data/premierleague/premierleague_1415_featured2.csv')
    X_16, y_16, odds16 = dataset('../data/premierleague/premierleague_1516_featured2.csv')
    X_17, y_17, odds17 = dataset('../data/premierleague/premierleague_1617_featured2.csv')
    X_18, y_18, odds18 = dataset('../data/premierleague/premierleague_1718_featured2.csv')
    X_pl = np.concatenate((X_13, X_14, X_15, X_16, X_17, X_18))
    y_pl = np.concatenate((y_13, y_14, y_15, y_16, y_17, y_18))
    odds_pl = np.concatenate((odds13, odds14, odds15, odds16, odds17, odds18))
    np.save('../data/NPY_FILES/V2/X_premierleague', X_pl)
    np.save('../data/NPY_FILES/V2/y_premierleague', y_pl)
    np.save('../data/NPY_FILES/V2/odds_premierleague', odds_pl)

    X_13, y_13, odds13 = dataset('../data/bundesliga/bundesliga_1213_featured2.csv')
    X_14, y_14, odds14 = dataset('../data/bundesliga/bundesliga_1314_featured2.csv')
    X_15, y_15, odds15 = dataset('../data/bundesliga/bundesliga_1415_featured2.csv')
    X_16, y_16, odds16 = dataset('../data/bundesliga/bundesliga_1516_featured2.csv')
    X_17, y_17, odds17 = dataset('../data/bundesliga/bundesliga_1617_featured2.csv')
    X_18, y_18, odds18 = dataset('../data/bundesliga/bundesliga_1718_featured2.csv')
    X_bl = np.concatenate((X_13, X_14, X_15, X_16, X_17, X_18))
    y_bl = np.concatenate((y_13, y_14, y_15, y_16, y_17, y_18))
    odds_bl = np.concatenate((odds13, odds14, odds15, odds16, odds17, odds18))
    np.save('../data/NPY_FILES/V2/X_bundesliga', X_bl)
    np.save('../data/NPY_FILES/V2/y_bundesliga', y_bl)
    np.save('../data/NPY_FILES/V2/odds_bundesliga', odds_bl)

    X = np.concatenate((X_sa, X_l1, X_la, X_pl, X_bl))
    y = np.concatenate((y_sa, y_l1, y_la, y_pl, y_bl))
    odds = np.concatenate((odds_sa, odds_l1, odds_la, odds_pl, odds_bl))
    np.save('../data/NPY_FILES/V2/X', X)
    np.save('../data/NPY_FILES/V2/y', y)
    np.save('../data/NPY_FILES/V2/odds', odds)
    """

    X_13, y_13, odds13 = dataset('../data/serieA/serieA_1213_featured2.csv')
    X_14, y_14, odds14 = dataset('../data/serieA/serieA_1314_featured2.csv')
    X_15, y_15, odds15 = dataset('../data/serieA/serieA_1415_featured2.csv')
    X_16, y_16, odds16 = dataset('../data/serieA/serieA_1516_featured2.csv')
    X_17, y_17, odds17 = dataset('../data/serieA/serieA_1617_featured2.csv')
    X_serieA_18, y_serieA_18, odds_serieA_18 = dataset('../data/serieA/serieA_1718_featured2.csv')
    X_sa = np.concatenate((X_13, X_14, X_15, X_16, X_17))
    y_sa = np.concatenate((y_13, y_14, y_15, y_16, y_17))
    odds_sa = np.concatenate((odds13, odds14, odds15, odds16, odds17))


    X_13, y_13, odds13 = dataset('../data/ligue1/ligue1_1213_featured2.csv')
    X_14, y_14, odds14 = dataset('../data/ligue1/ligue1_1314_featured2.csv')
    X_15, y_15, odds15 = dataset('../data/ligue1/ligue1_1415_featured2.csv')
    X_16, y_16, odds16 = dataset('../data/ligue1/ligue1_1516_featured2.csv')
    X_17, y_17, odds17 = dataset('../data/ligue1/ligue1_1617_featured2.csv')
    X_ligue1_18, y_ligue1_18, odds_ligue1_18 = dataset('../data/ligue1/ligue1_1718_featured2.csv')
    X_l1 = np.concatenate((X_13, X_14, X_15, X_16, X_17))
    y_l1 = np.concatenate((y_13, y_14, y_15, y_16, y_17))
    odds_l1 = np.concatenate((odds13, odds14, odds15, odds16, odds17))


    X_13, y_13, odds13 = dataset('../data/liga/liga_1213_featured2.csv')
    X_14, y_14, odds14 = dataset('../data/liga/liga_1314_featured2.csv')
    X_15, y_15, odds15 = dataset('../data/liga/liga_1415_featured2.csv')
    X_16, y_16, odds16 = dataset('../data/liga/liga_1516_featured2.csv')
    X_17, y_17, odds17 = dataset('../data/liga/liga_1617_featured2.csv')
    X_liga_18, y_liga_18, odds_liga_18 = dataset('../data/liga/liga_1718_featured2.csv')
    X_liga = np.concatenate((X_13, X_14, X_15, X_16, X_17))
    y_liga = np.concatenate((y_13, y_14, y_15, y_16, y_17))
    odds_liga = np.concatenate((odds13, odds14, odds15, odds16, odds17))


    X_13, y_13, odds13 = dataset('../data/premierleague/premierleague_1213_featured2.csv')
    X_14, y_14, odds14 = dataset('../data/premierleague/premierleague_1314_featured2.csv')
    X_15, y_15, odds15 = dataset('../data/premierleague/premierleague_1415_featured2.csv')
    X_16, y_16, odds16 = dataset('../data/premierleague/premierleague_1516_featured2.csv')
    X_17, y_17, odds17 = dataset('../data/premierleague/premierleague_1617_featured2.csv')
    X_pl_18, y_pl_18, odds_pl_18 = dataset('../data/premierleague/premierleague_1718_featured2.csv')
    X_pl = np.concatenate((X_13, X_14, X_15, X_16, X_17))
    y_pl = np.concatenate((y_13, y_14, y_15, y_16, y_17))
    odds_pl = np.concatenate((odds13, odds14, odds15, odds16, odds17))


    X_13, y_13, odds13 = dataset('../data/bundesliga/bundesliga_1213_featured2.csv')
    X_14, y_14, odds14 = dataset('../data/bundesliga/bundesliga_1314_featured2.csv')
    X_15, y_15, odds15 = dataset('../data/bundesliga/bundesliga_1415_featured2.csv')
    X_16, y_16, odds16 = dataset('../data/bundesliga/bundesliga_1516_featured2.csv')
    X_17, y_17, odds17 = dataset('../data/bundesliga/bundesliga_1617_featured2.csv')
    X_bdl_18, y_bdl_18, odds_bdl_18 = dataset('../data/bundesliga/bundesliga_1718_featured2.csv')
    X_bdl = np.concatenate((X_13, X_14, X_15, X_16, X_17))
    y_bdl = np.concatenate((y_13, y_14, y_15, y_16, y_17))
    odds_bdl = np.concatenate((odds13, odds14, odds15, odds16, odds17))

    X_train = np.concatenate((X_sa, X_l1, X_liga, X_pl, X_bdl))
    X_test = np.concatenate((X_serieA_18, X_ligue1_18, X_liga_18, X_pl_18, X_bdl_18))
    y_train = np.concatenate((y_sa, y_l1, y_liga, y_pl, y_bdl))
    y_test = np.concatenate((y_serieA_18, y_ligue1_18, y_liga_18, y_pl_18, y_bdl_18))
    odds_train = np.concatenate((odds_sa, odds_l1, odds_liga, odds_pl, odds_bdl))
    odds_test = np.concatenate((odds_serieA_18, odds_ligue1_18, odds_liga_18, odds_pl_18, odds_bdl_18))
    np.save('../data/NPY_FILES/V2/X_train', X_train)
    np.save('../data/NPY_FILES/V2/X_test', X_test)
    np.save('../data/NPY_FILES/V2/y_train', y_train)
    np.save('../data/NPY_FILES/V2/y_test', y_test)
    np.save('../data/NPY_FILES/V2/odds_train', odds_train)
    np.save('../data/NPY_FILES/V2/odds_test', odds_test)