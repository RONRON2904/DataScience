import numpy as np
import utils

class Championship:
    def __init__(self, dataframe, season, country='Norway'):
        self.dataframe = dataframe
        self.season = season
        self.country = country
        self.home_points, self.away_points = utils.season_points(dataframe, season, country)

    def average_home_allspoints(self, first, last):
        return np.mean(self.home_points[first : last])

    def average_away_allspoints(self, first, last):
        return np.mean(self.away_points[first : last])

    def get_meetings(self):
        return utils.get_meetings(self.dataframe, self.season, self.country)

    def get_score(self, meeting):
        return utils.get_score(self.dataframe, meeting, self.season, self.country)

    def get_home_odd(self, meeting, cat):
        if cat == 'openning':
            return utils.get_openning_odd(self.dataframe, meeting, 'home_win', self.season, self.country)
        else:
            return utils.get_closing_odd(self.dataframe, meeting, 'home_win', self.season, self.country)

    def get_draw_odd(self, meeting, cat):
        if cat == 'openning':
            return utils.get_openning_odd(self.dataframe, meeting, 'draw', self.season, self.country)
        else:
            return utils.get_closing_odd(self.dataframe, meeting, 'draw', self.season, self.country)

    def get_away_odd(self, meeting, cat):
        if cat == 'openning':
            return utils.get_openning_odd(self.dataframe, meeting, 'away_win', self.season, self.country)
        else:
            return utils.get_closing_odd(self.dataframe, meeting, 'away_win', self.season, self.country)            

    def get_nb_games(self, meeting):
        return utils.get_nb_games(self.dataframe, meeting, self.season, self.country)