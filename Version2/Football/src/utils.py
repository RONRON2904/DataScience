import pandas as pd
from collections import OrderedDict

#dframe = pd.read_csv('~/projets/paris_sportifs/Version2/Football/data/ligue1/ligue1_1213.csv')

def home_spoints(dataframe, team_name, season=None, country='Norway'):
    df = dataframe
    if country == 'Norway':
        df = df.loc[df['Season'] == season]
    team = df.loc[(df['HomeTeam'] == team_name)]
    return team['FTHG'].values


def away_spoints(dataframe, team_name, season=None, country='Norway'):
    df = dataframe
    if country == 'Norway':
        df = df.loc[df['Season'] == season]
    team = df.loc[(df['AwayTeam'] == team_name)]
    return team['FTAG'].values

#print(away_spoints(dframe, 'Paris SG', country='France'))

def home_tpoints(dataframe, team_name, season=None, country='Norway'):
    df = dataframe
    if country == 'Norway':
        df = df.loc[df['Season'] == season]
    team = df.loc[(df['HomeTeam'] == team_name)]
    return team['FTAG'].values


def away_tpoints(dataframe, team_name, season=None, country='Norway'):
    df = dataframe
    if country == 'Norway':
        df = df.loc[df['Season'] == season]
    team = df.loc[(df['AwayTeam'] == team_name)]
    return team['FTHG'].values

#print(away_tpoints(dframe, 'Paris SG', country='France'))

def home_shots(dataframe, team_name, season=None, country='Norway', target=False):
    df = dataframe
    if country == 'Norway':
        df = df.loc[df['Season'] == season]
    team = df.loc[df['HomeTeam'] == team_name]
    if target:
        return team['HST'].values
    return team['HS'].values

def home_shots_conceded(dataframe, team_name, season=None, country='Norway', target=False):
    df = dataframe
    if country == 'Norway':
        df = df.loc[df['Season'] == season]
    team = df.loc[df['HomeTeam'] == team_name]
    if target:
        return team['AST'].values        
    return team['AS'].values
    
def away_shots(dataframe, team_name, season=None, country='Norway',  target=False):
    df = dataframe
    if country == 'Norway':
        df = df.loc[df['Season'] == season]
    team = df.loc[df['AwayTeam'] == team_name]
    if target:
        return team['AST'].values
    return team['AS'].values

def away_shots_conceded(dataframe, team_name, season=None, country='Norway', target=False):
    df = dataframe
    if country == 'Norway':
        df = df.loc[df['Season'] == season]
    team = df.loc[df['AwayTeam'] == team_name]
    if target:
        return team['HST'].values
    return team['HS'].values

def home_corners(dataframe, team_name, season=None, country='Norway'):
    df = dataframe
    if country == 'Norway':
        df = df.loc[df['Season'] == season]
    team = df.loc[df['HomeTeam'] == team_name]
    return team['HC'].values

def home_corners_conceded(dataframe, team_name, season=None, country='Norway'):
    df = dataframe
    if country == 'Norway':
        df = df.loc[df['Season'] == season]
    team = df.loc[df['HomeTeam'] == team_name]
    return team['AC'].values

def away_corners(dataframe, team_name, season=None, country='Norway'):
    df = dataframe
    if country == 'Norway':
        df = df.loc[df['Season'] == season]
    team = df.loc[df['AwayTeam'] == team_name]
    return team['AC'].values

def away_corners_conceded(dataframe, team_name, season=None, country='Norway'):
    df = dataframe
    if country == 'Norway':
        df = df.loc[df['Season'] == season]
    team = df.loc[df['AwayTeam'] == team_name]
    return team['HC'].values

def home_fouls(dataframe, team_name, season=None, country='Norway'):
    df = dataframe
    if country == 'Norway':
        df = df.loc[df['Season'] == season]
    team = df.loc[df['HomeTeam'] == team_name]
    return team['HF'].values

def home_fouls_conceded(dataframe, team_name, season=None, country='Norway'):
    df = dataframe
    if country == 'Norway':
        df = df.loc[df['Season'] == season]
    team = df.loc[df['HomeTeam'] == team_name]
    return team['AF'].values

def away_fouls(dataframe, team_name, season=None, country='Norway'):
    df = dataframe
    if country == 'Norway':
        df = df.loc[df['Season'] == season]
    team = df.loc[df['AwayTeam'] == team_name]
    return team['AF'].values

def away_fouls_conceded(dataframe, team_name, season=None, country='Norway'):
    df = dataframe
    if country == 'Norway':
        df = df.loc[df['Season'] == season]
    team = df.loc[df['AwayTeam'] == team_name]
    return team['HF'].values

def season_points(dataframe, season=None, country='Norway'):
    df = dataframe
    if country == 'Norway':
        df = df.loc[df['Season'] == season]
    return df['FTHG'].values, df['FTAG'].values

#print(season_points(dframe, country='France'))

def get_meetings(dataframe, season=None, country='Norway'):
    df = dataframe
    if country == 'Norway':
        df = df.loc[df['Season'] == season]
    return df[['HomeTeam', 'AwayTeam', 'Date']].values

#print(get_meetings(dframe, country='France'))

def get_nb_home_games(dataframe, meeting, season=None, country='Norway'):
    df = dataframe
    if country == 'Norway':
        df = df.loc[df['Season'] == season]
    idx = df.loc[(df['HomeTeam'] == meeting[0]) & (df['AwayTeam'] == meeting[1])].index[0]
    df = df[: idx]
    return df[df['HomeTeam'] == meeting[0]].shape[0]

#print(get_nb_home_games(dframe, ['Paris SG', 'Toulouse'], country='France'))

def get_nb_away_games(dataframe, meeting, season=None, country='Norway'):
    df = dataframe
    if country == 'Norway':
        df = df.loc[df['Season'] == season]
    idx = df.loc[(df['HomeTeam'] == meeting[0]) & (df['AwayTeam'] == meeting[1])].index[0]
    df = df[: idx]
    return df[df['AwayTeam'] == meeting[1]].shape[0]
    
def get_nb_games(dataframe, meeting, season=None, country='Norway'):
    df = dataframe
    if country == 'Norway':
        df = df.loc[df['Season'] == season]
    return df.loc[df['Date'] == meeting[2]].index[0]


#print(get_nb_games(dframe, ['Paris SG', 'Bordeaux', '26/08/12'], country='France'))

def get_form(dataframe, meeting, cat, team_name, season=None, country='Norway'):
    df = dataframe
    if country == 'Norway':
        df = df.loc[df['Season'] == season]
    idx = df.loc[(df['HomeTeam'] == meeting[0]) & (df['AwayTeam'] == meeting[1])].index[0]
    df = df[: idx]
    def outcome(x):
        if cat == 'home' or cat == 'both':
            if x['HomeTeam'] == team_name:
                if x['FTHG'] > x['FTAG']:
                    return 2
                elif x['FTHG'] == x['FTAG']:
                    return 1
                else:
                    return 0
        if cat == 'away' or cat == 'both':
            if x['AwayTeam'] == team_name:
                if x['FTHG'] < x['FTAG']:
                    return 2
                elif x['FTHG'] == x['FTAG']:
                    return 1
                else:
                    return 0
    return df.apply(lambda x: outcome(x), axis=1).dropna().values

#print(get_form(dframe, ['Paris SG', 'Reims'], 'both', 'Paris SG', country='France'))

def get_points(dataframe, type, team_name, meeting, season=None, country='Norway'):
    df = dataframe
    if country == 'Norway':
        df = df.loc[df['Season'] == season]
    idx = df.loc[(df['HomeTeam'] == meeting[0]) & (df['AwayTeam'] == meeting[1])].index[0]
    df = df[: idx]
    def point(x):
        if (x['HomeTeam'] == team_name) and (type != 'away'):
            if x['FTHG'] > x['FTAG']:
                return 3
            elif x['FTHG'] == x['FTAG']:
                return 1
            else:
                return 0
        if (x['AwayTeam'] == team_name) and (type != 'home'):
            if x['FTHG'] < x['FTAG']:
                return 3
            elif x['FTHG'] == x['FTAG']:
                return 1
            else:
                return 0
    return df.apply(lambda x: point(x), axis=1).dropna().values.sum()

#print(get_points(dframe, 'home', 'Paris SG', ['Paris SG', 'Sochaux'], country='France'))

def get_date(dataframe, meeting, season=None, country='Norway'):
    df = dataframe
    if country == 'Norway':
        df = df.loc[df['Season'] == season]
    df = df.loc[(df['HomeTeam'] == meeting[0]) & (df['AwayTeam'] == meeting[1])]
    return df['Date']

def get_rank(dataframe, type, team_name, meeting, season=None, country='Norway'):
    df = dataframe
    if country == 'Norway':
        df = df.loc[df['Season'] == season]
    teams = df.HomeTeam.unique().tolist()
    p_dict = {team: get_points(df, type, team, meeting, season=season, country=country) for team in teams}
    ranks = OrderedDict(sorted(p_dict.items(), key=lambda x: x[1], reverse=True))
    return int(list(ranks.keys()).index(team_name) + 1)

#print(get_rank(dframe, 'home', 'Paris SG', ['Bordeaux', 'Sochaux'], country='France'))

def get_score(dataframe, meeting, season=None, country='Norway'):
    df = dataframe
    if country == 'Norway':
        df = df.loc[df['Season'] == season]
    df = df.loc[(df['HomeTeam'] == meeting[0]) & (df['AwayTeam'] == meeting[1])]
    df = df.iloc[0]
    return df['FTHG'], df['FTAG']

def get_openning_odd(dataframe, meeting, type, season=None, country='Norway'):
    df = dataframe
    if country == 'Norway':
        df = df.loc[df['Season'] == season]
    df = df.loc[(df['HomeTeam'] == meeting[0]) & (df['AwayTeam'] == meeting[1])]
    df = df.iloc[0]
    if type == 'home_win':
        return float(df['PSH'])
    elif(type == 'away_win'):
        return float(df['PSA'])
    elif(type == 'draw'):
        return float(df['PSD'])
    else:
        return 1.

def get_closing_odd(dataframe, meeting, type, season=None, country='Norway'):
    df = dataframe
    if country == 'Norway':
        df = df.loc[df['Season'] == season]
    df = df.loc[(df['HomeTeam'] == meeting[0]) & (df['AwayTeam'] == meeting[1])]
    df = df.iloc[0]
    if type == 'home_win':
        return float(df['PSCH'])
    elif(type == 'away_win'):
        return float(df['PSCA'])
    elif(type == 'draw'):
        return float(df['PSCD'])
    else:
        return 1.