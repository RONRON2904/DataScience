import numpy as np
import utils

class Team:
        def __init__(self, team_name, dataframe, season, country='Norway'):
            self.team_name = team_name
            self.dataframe = dataframe
            self.season = season
            self.country = country
            self.home_spoints = utils.home_spoints(dataframe, team_name, season, country)
            self.home_tpoints = utils.home_tpoints(dataframe, team_name, season, country)
            self.away_spoints = utils.away_spoints(dataframe, team_name, season, country)
            self.away_tpoints =  utils.away_tpoints(dataframe, team_name, season, country)
            self.home_shots = utils.home_shots(dataframe, team_name, season, country)
            self.home_shots_on_target = utils.home_shots(dataframe, team_name, season, country, target=True)
            self.home_shots_conceded = utils.home_shots_conceded(dataframe, team_name, season, country)
            self.home_shots_conceded_on_target = utils.home_shots_conceded(dataframe, team_name, season, country, target=True)
            self.away_shots = utils.away_shots(dataframe, team_name, season, country)
            self.away_shots_on_target = utils.away_shots(dataframe, team_name, season, country, target=True)
            self.away_shots_conceded = utils.away_shots_conceded(dataframe, team_name, season, country)
            self.away_shots_conceded_on_target = utils.away_shots_conceded(dataframe, team_name, season, country, target=True)
            self.home_corners = utils.home_corners(dataframe, team_name, season, country)
            self.away_corners = utils.away_corners(dataframe, team_name, season, country)
            self.home_corners_conceded = utils.home_corners_conceded(dataframe, team_name, season, country)
            self.away_corners_conceded = utils.away_corners_conceded(dataframe, team_name, season, country)


        def get_average_corners(self, cat, first, last):
            if cat == 'home':
                return np.mean(self.home_corners[first: last])
            else:
                return np.mean(self.away_corners[first: last])

        def get_average_corners_conceded(self, cat, first, last):
            if cat == 'home':
                return np.mean(self.home_corners_conceded[first: last])
            else:
                return np.mean(self.away_corners_conceded[first: last])

        def get_average_shots(self, cat, first, last):
            if cat == 'home':
                return np.mean(self.home_shots[first: last])
            else:
                return np.mean(self.away_shots[first: last])

        def get_average_shots_on_target(self, cat, first, last):
            if cat == 'home':
                return np.mean(self.home_shots_on_target[first: last])
            else:
                return np.mean(self.away_shots_on_target[first: last])

        def get_average_shots_conceded(self, cat, first, last):
            if cat == 'home':
                return np.mean(self.home_shots_conceded[first: last])
            else:
                return np.mean(self.away_shots_conceded[first: last])

        def get_average_home_shots_conceded_on_target(self, cat, first, last):
            if cat == 'home':
                return np.mean(self.home_shots_conceded_on_target[first: last])
            else:
                return np.mean(self.away_shots_conceded_on_target[first: last])

        def get_average_accuracy(self, cat, first, last):
            if cat == 'home':
                acc = self.home_shots / self.home_shots_on_target
            else:
                acc = self.away_shots / self.away_shots_on_target
            acc[np.isnan(acc) | (acc == np.inf)] = 0
            return np.mean(acc[first: last])

        def get_average_efficiency(self, cat, first, last):
            if cat == 'home':
                acc = self.home_spoints / self.home_shots
            else:
                acc = self.away_spoints / self.away_shots
            acc[np.isnan(acc) | (acc == np.inf)] = 0
            return np.mean(acc[first: last])

        def get_average_vulnerability(self, cat, first, last):
            if cat == 'home':
                acc = self.home_tpoints / self.home_shots_conceded
            else: 
                acc = self.away_tpoints / self.away_shots_conceded 
            acc[np.isnan(acc) | (acc == np.inf)] = 0
            return np.mean(acc[first: last])

        def get_average_spoints(self, cat, first, last):
            if cat == 'home':
                return np.mean(self.home_spoints[first: last])
            else:
                return np.mean(self.away_spoints[first: last])

        def get_average_tpoints(self, cat, first, last):
            if cat == 'home':
                return np.mean(self.home_tpoints[first: last])
            else:
                return np.mean(self.away_tpoints[first: last])

        def get_outcomes(self, cat, meeting):
            return utils.get_form(self.dataframe, meeting, cat, self.team_name, self.season, self.country)
    
        def get_points(self, cat, meeting):
            return utils.get_points(self.dataframe, cat, self.team_name, meeting, self.season, self.country)
            
        def get_rank(self, cat, meeting):
            return utils.get_rank(self.dataframe, cat, self.team_name, meeting, self.season, self.country)

        def get_nb_games(self, cat, meeting):
            if cat == 'home':
                return utils.get_nb_home_games(self.dataframe, meeting, self.season, self.country)
            else:
                return utils.get_nb_away_games(self.dataframe, meeting, self.season, self.country)
