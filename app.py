import numpy as np
import pandas as pd
from rawdata import file_path, url
# import matplotlib.pyplot as plt
# import seaborn as sn

data = pd.read_csv(file_path, low_memory=False)

# with pd.option_context('display.max_rows', 999, 'display.max_colwidth', 25):
#     print(data.head(4).transpose())

# selected_columns = [
#     'pass_attempt', 'rush_attempt', 'sack', 'qb_hit', 'touchdown', 'play_type'
# ]
# for c in selected_columns:
#     print(data[c].value_counts(normalize=True).to_frame(), '\n')

# run_pass_row_indices = data[data['play_type'].isin(
#     ['run', 'pass', 'sack'])].index

# runs_passes_sacks = data.loc[run_pass_row_indices, :]
# print(runs_passes_sacks['play_type'].value_counts(normalize=True).to_frame())


def pos_win_prob_neg_ep(team_name):
    """
    Returns all plays for a team where win probability
    is increased, but expected points decreased
    """

    total = 0
    for index, row in data.iterrows():
        if row.posteam == team_name and row.epa <= 0 and row.wpa > 0:
            total += 1
        if row.defteam == team_name and row.epa >= 0 and row.wpa < 0:
            total += 1
    return f"{team_name} {total}"


def generate_set_of_row_attribute(attr):
    """Generates a set of all vals for certain row"""

    new_set = set()
    for index, row in data.iterrows():
        new_set.add(row[attr])
    return new_set


def home_and_away_total_air_epa_diff(team_name):
    """
    Total air expected points added for one team
    seperated by home and away
    """

    home_total = 0
    away_total = 0

    for index, row in data.iterrows():
        if row.home_team == team_name and row.game_seconds_remaining == 0:
            home_total += row.total_home_pass_epa
        if row.away_team == team_name and row.game_seconds_remaining == 0:
            away_total += row.total_away_pass_epa

    print(f"{team_name} HOME TOTAL AIR EPA", home_total)
    print(f"{team_name} AWAY TOTAL AIR EPA", away_total)


def find_best_and_worst_higher_number_better(team_vals):
    """
    Takes a dict of all team values for a given statistic
    calculates the mean and standard deviation
    and returns a dict of the outliers
    """

    outliers = {"best": {}, "worst": {}}
    std_dev = np.std(list(team_vals.values()))
    mean = np.mean(list(team_vals.values()))
    for x in team_vals:
        if team_vals[x] > mean + std_dev:
            outliers['best'][x] = team_vals[x]
        if team_vals[x] < mean - std_dev:
            outliers['worst'][x] = team_vals[x]
    print(f"STD DEV: {std_dev}\nMEAN: {mean}")
    return outliers


def find_best_and_worst_lower_number_better(team_vals):
    """
    Takes a dict of all team values for a given statistic
    calculates the mean and standard deviation
    and returns a dict of the outliers
    """

    outliers = {"best": {}, "worst": {}}
    std_dev = np.std(list(team_vals.values()))
    mean = np.mean(list(team_vals.values()))
    for x in team_vals:
        if team_vals[x] > mean + std_dev:
            outliers['worst'][x] = team_vals[x]
        if team_vals[x] < mean - std_dev:
            outliers['best'][x] = team_vals[x]
    print(f"STD DEV: {std_dev}\nMEAN: {mean}")
    return outliers


def find_extreme_best_and_worst_higher_better(team_vals):
    """
    Takes a dict of all team values for a given statistic
    calculates the mean and standard deviation
    and returns a dict of the outliers
    """

    outliers = {"best": {}, "worst": {}}
    std_dev = np.std(list(team_vals.values()))
    mean = np.mean(list(team_vals.values()))
    for x in team_vals:
        if team_vals[x] > mean + 2 * std_dev:
            outliers['best'][x] = team_vals[x]
        if team_vals[x] < mean - 2 * std_dev:
            outliers['worst'][x] = team_vals[x]
    print(f"STD DEV: {std_dev}\nMEAN: {mean}")
    return outliers


def find_extreme_best_and_worst_lower_better(team_vals):
    """
    Takes a dict of all team values for a given statistic
    calculates the mean and standard deviation
    and returns a dict of the outliers
    """

    outliers = {"best": {}, "worst": {}}
    std_dev = np.std(list(team_vals.values()))
    mean = np.mean(list(team_vals.values()))
    for x in team_vals:
        if team_vals[x] > mean + 2 * std_dev:
            outliers['worst'][x] = team_vals[x]
        if team_vals[x] < mean - 2 * std_dev:
            outliers['best'][x] = team_vals[x]
    print(f"STD DEV: {std_dev}\nMEAN: {mean}")
    return outliers


def red_zone_offense_epa():
    """
    Return dict of Expected Points added on offense
    per red zone play
    """

    all_vals = {}
    for x in Team.all():
        total_plays = 0
        total_epa = 0
        for index, row in data.iterrows():
            if row.yardline_100 <= 25 and row.posteam == x and str(
                    row.epa) != 'nan':
                total_epa += row.epa
                total_plays += 1
        all_vals[x] = round(total_epa / total_plays, 3)
    return all_vals


def best_when_trailing_after_half_offense():
    """Return dict of EPA per offensieve play when trailing after half"""

    all_vals = {}
    for x in Team.all():
        total_plays = 0
        total_epa = 0
        for index, row in data.iterrows():
            if str(
                    row.game_half
            ) == 'Half2' and row.posteam == x and row.score_differential < 0 and str(
                    row.epa) != 'nan':
                total_epa += row.epa
                total_plays += 1
        all_vals[x] = round(total_epa / total_plays, 3)
    return all_vals


def best_when_trailing_after_half_margin():
    """Return dict of EPA margin play when trailing after half"""

    all_vals = {}
    for x in Team.all():
        total_plays = 0
        total_epa = 0
        for index, row in data.iterrows():
            if str(
                    row.game_half
            ) == 'Half2' and row.posteam == x and row.score_differential < 0 and str(
                    row.epa) != 'nan':
                total_epa += row.epa
                total_plays += 1
            if str(
                    row.game_half
            ) == 'Half2' and row.defteam == x and row.score_differential > 0 and str(
                    row.epa) != 'nan':
                total_epa -= row.epa
                total_plays += 1
        all_vals[x] = round(total_epa / total_plays, 3)
    return all_vals


def best_when_trailing_in_fourth_margin():
    """Return dict of EPA margin play when trailing in the fourth"""

    all_vals = {}
    for x in Team.all():
        total_plays = 0
        total_epa = 0
        for index, row in data.iterrows():
            if row.qtr == 4 and row.posteam == x and row.score_differential < 0 and str(
                    row.epa) != 'nan':
                total_epa += row.epa
                total_plays += 1
            if row.qtr == 4 and row.defteam == x and row.score_differential > 0 and str(
                    row.epa) != 'nan':
                total_epa -= row.epa
                total_plays += 1
        all_vals[x] = round(total_epa / total_plays, 3)
    return all_vals


def find_yac_epa_per_player():
    full_dict = {}
    final_dict = {}
    for i, row in data.iterrows():
        if row.play_type == 'pass' and str(row.yac_epa) != 'nan':
            name = row.receiver_player_name
            if full_dict.get(name, 0) == 0:
                full_dict[name] = {"yac_epa": 0, "total_plays": 0}
            full_dict[name]["yac_epa"] += row.yac_epa
            full_dict[name]["total_plays"] += 1
    for key in full_dict:
        yac_per_play = full_dict[key]["yac_epa"] / full_dict[key]["total_plays"]
        final_dict[key] = yac_per_play
    return final_dict


def find_percentage_of_passes_defensed():
    """Find percentage of passes deflected or intercepted"""

    final_dict = {}
    for x in Team.all():
        total_passes = 0
        defensed = 0
        for i, row in data.iterrows():
            if row.play_type == 'pass' and row.defteam == x:
                total_passes += 1
                if str(row.pass_defense_1_player_id).lower(
                ) != 'nan' or row.interception == 1:
                    defensed += 1
        final_dict[x] = round((defensed / total_passes) * 100, 2)
    return final_dict


def find_third_and_long_pass_protection():
    """Find pass blocking efficiency on third and long"""

    final_dict = {}
    for x in Team.all():
        total_passes = 0
        bad_ones = 0
        for i, row in data.iterrows():
            if row.play_type == 'pass' and row.posteam == x and row.down == 3 and row.ydstogo >= 8:
                total_passes += 1
                if row.sack == 1 or row.qb_hit == 1:
                    bad_ones += 1
        final_dict[x] = round((bad_ones / total_passes) * 100, 2)
    return final_dict


class Team:
    """Methods for returning data on one team"""

    def __init__(self, team):
        self.team = team

    def __repr__(self):
        return self.team

    @classmethod
    def all(cls):
        teams = [
            'ARI', 'ATL', 'BAL', 'BUF', 'CAR', 'CHI', 'CIN', 'CLE', 'DAL',
            'DEN', 'DET', 'GB', 'HOU', 'IND', 'JAX', 'KC', 'LA', 'LAC', 'MIA',
            'MIN', 'NE', 'NO', 'NYG', 'NYJ', 'OAK', 'PHI', 'PIT', 'SEA', 'SF',
            'TB', 'TEN', 'WAS'
        ]
        return teams

    # Offensive methods

    def epa_per_play_off(self, operation, **kwargs):
        """
        Accepts a required str value operation, either 'mean', 'median', or 'std'.
        Accepts an optional int value for 'down' 1-4 inclusive.
        Accepts an optional int value for 'quarter' 1-5 inclusive.
        Accepts an optional str value for 'play', either 'run', or 'pass'.
        Returns Expected Points Added per play for combination of 'quarter', 'down', and 'play' and calculates based on operation.
        """

        # valid_args = ('play', 'down', 'quarter')
        # plays = ('run', 'pass')
        # dwns = (1, 2, 3, 4)
        # qts = (1, 2, 3, 4, 5)
        # ops = ('mean', 'std', 'median')
        # try:
        #     if operation not in ops:
        #         raise ValueError
        #     for key in kwargs:
        #         if key not in valid_args:
        #             raise ValueError
        #         if 'play' in kwargs and kwargs['play'] not in plays:
        #             raise ValueError
        #         if 'down' in kwargs and kwargs['down'] not in dwns:
        #             raise ValueError
        #         if 'quarter' in kwargs and kwargs['quarter'] not in qts:
        #             raise ValueError
        # except ValueError:
        #     return "Optional argument requirements: 'down' must be 1, 2, 3, or 4. 'operation' must be 'mean', 'median', or 'std'. 'quarter' must be 1, 2, 3, 4, or 5. 'play' must be 'run', or 'pass'."

        query = f"posteam == '{self.team}' & play_type != 'no_play'"

        if "play" in kwargs:
            if kwargs["play"] == "run":
                query += " & rush_attempt == 1"
            if kwargs["play"] == "pass":
                query += " & pass_attempt == 1"
        if "quarter" in kwargs:
            query += f" & qtr == {kwargs['quarter']}"
        if "down" in kwargs:
            query += f" & down == {kwargs['down']}"

        df = data.query(query)['epa']

        if operation == 'mean':
            return round(df.mean(), 3)
        if operation == 'median':
            return round(df.median(), 3)
        if operation == 'std':
            return round(df.std(), 3)

    def positive_play_percentage_offense(self, **kwargs):
        """
        Accepts an optional int value for 'down' 1-4 inclusive.
        Accepts an optional int value for 'quarter' 1-5 inclusive.
        Accepts an optional str value for 'play', either 'run', or 'pass'.
        Returns percentage of plays that increased team's expected points
        """

        # Accept down, quarter, play
        pos_query = f"posteam == '{self.team}' & play_type != 'no_play' & epa > 0"
        all_query = f"posteam == '{self.team}' & play_type != 'no_play'"

        if "play" in kwargs:
            if kwargs["play"] == "run":
                pos_query += " & rush_attempt == 1"
                all_query += " & rush_attempt == 1"
            if kwargs["play"] == "pass":
                pos_query += " & pass_attempt == 1"
                all_query += " & pass_attempt == 1"
        if "quarter" in kwargs:
            pos_query += f" & qtr == {kwargs['quarter']}"
            all_query += f" & qtr == {kwargs['quarter']}"
        if "down" in kwargs:
            pos_query += f" & down == {kwargs['down']}"
            all_query += f" & down == {kwargs['down']}"

        pos_plays = data.query(pos_query)
        all_plays = data.query(all_query)
        percentage = pos_plays.shape[0] / all_plays.shape[0]
        return round(percentage, 3)

    def qb_pressure_offense(self, **kwargs):
        """
        Accepts an optional int value for 'down' 1-4 inclusive.
        Accepts an optional int value for 'quarter' 1-5 inclusive.
        Returns percentage of plays where qb was neither hit nor sacked.
        """

        # Accept down, quarter

        hit_query = f"posteam == '{self.team}' & play_type != 'no_play' & pass_attempt == 1 & qb_hit == 1"
        sack_query = f"posteam == '{self.team}' & play_type != 'no_play' & pass_attempt == 1 & qb_hit == 0 & sack == 1"
        all_pass_query = f"posteam == '{self.team}' & play_type != 'no_play' & pass_attempt == 1"

        if "quarter" in kwargs:
            all_pass_query += f" & qtr == {kwargs['quarter']}"
            sack_query += f" & qtr == {kwargs['quarter']}"
            hit_query += f" & qtr == {kwargs['quarter']}"
        if "down" in kwargs:
            all_pass_query += f" & down == {kwargs['down']}"
            sack_query += f" & down == {kwargs['down']}"
            hit_query += f" & down == {kwargs['down']}"

        passes = data.query(all_pass_query).shape[0]
        hits_and_sacks = data.query(hit_query).shape[0] + data.query(
            sack_query).shape[0]
        percentage = hits_and_sacks / passes
        return round(percentage, 3)

    # Defensive methods

    def epa_per_play_def(self, operation, **kwargs):
        """
        Accepts a required str value operation, either 'mean', 'median', or 'std'.
        Accepts an optional int value for 'down' 1-4 inclusive.
        Accepts an optional int value for 'quarter' 1-5 inclusive.
        Accepts an optional str value for 'play', either 'run', or 'pass'.
        Returns Expected Points Added per play for combination of 'quarter', 'down', and 'play' and calculates based on operation.
        """

        query = f"defteam == '{self.team}' & play_type != 'no_play'"

        if "play" in kwargs:
            if kwargs["play"] == "run":
                query += " & rush_attempt == 1"
            if kwargs["play"] == "pass":
                query += " & pass_attempt == 1"
        if "quarter" in kwargs:
            query += f" & qtr == {kwargs['quarter']}"
        if "down" in kwargs:
            query += f" & down == {kwargs['down']}"

        df = data.query(query)['epa']

        if operation == 'mean':
            return round(df.mean(), 3)
        if operation == 'median':
            return round(df.median(), 3)
        if operation == 'std':
            return round(df.std(), 3)

    def positive_play_percentage_defense(self, **kwargs):
        """
        Accepts an optional int value for 'down' 1-4 inclusive.
        Accepts an optional int value for 'quarter' 1-5 inclusive.
        Accepts an optional str value for 'play', either 'run', or 'pass'.
        Returns percentage of plays that increased team's expected points
        """

        # Accept down, quarter, play
        pos_query = f"defteam == '{self.team}' & play_type != 'no_play' & epa < 0"
        all_query = f"defteam == '{self.team}' & play_type != 'no_play'"

        if "play" in kwargs:
            if kwargs["play"] == "run":
                pos_query += " & rush_attempt == 1"
                all_query += " & rush_attempt == 1"
            if kwargs["play"] == "pass":
                pos_query += " & pass_attempt == 1"
                all_query += " & pass_attempt == 1"
        if "quarter" in kwargs:
            pos_query += f" & qtr == {kwargs['quarter']}"
            all_query += f" & qtr == {kwargs['quarter']}"
        if "down" in kwargs:
            pos_query += f" & down == {kwargs['down']}"
            all_query += f" & down == {kwargs['down']}"

        pos_plays = data.query(pos_query)
        all_plays = data.query(all_query)
        percentage = pos_plays.shape[0] / all_plays.shape[0]
        return round(percentage, 3)

    def qb_pressure_defense(self, **kwargs):
        """
        Accepts an optional int value for 'down' 1-4 inclusive.
        Accepts an optional int value for 'quarter' 1-5 inclusive.
        Returns percentage of plays where qb was either hit or sacked.
        """

        # Accept down, quarter

        hit_query = f"defteam == '{self.team}' & play_type != 'no_play' & pass_attempt == 1 & qb_hit == 1"
        sack_query = f"defteam == '{self.team}' & play_type != 'no_play' & pass_attempt == 1 & qb_hit == 0 & sack == 1"
        all_pass_query = f"defteam == '{self.team}' & play_type != 'no_play' & pass_attempt == 1"

        if "quarter" in kwargs:
            all_pass_query += f" & qtr == {kwargs['quarter']}"
            sack_query += f" & qtr == {kwargs['quarter']}"
            hit_query += f" & qtr == {kwargs['quarter']}"
        if "down" in kwargs:
            all_pass_query += f" & down == {kwargs['down']}"
            sack_query += f" & down == {kwargs['down']}"
            hit_query += f" & down == {kwargs['down']}"

        passes = data.query(all_pass_query).shape[0]
        hits_and_sacks = data.query(hit_query).shape[0] + data.query(
            sack_query).shape[0]
        percentage = hits_and_sacks / passes
        return round(percentage, 3)


class League:
    """Methods for returning league-wide information"""

    @classmethod
    def epa_per_play_off(cls, **kwargs):

        teams = Team.all()
        data = []
        for team in teams:
            x = Team(team)
            data.append((team, x.epa_per_play_off('mean', **kwargs)))
        return data

    @classmethod
    def epa_per_play_def(cls, **kwargs):

        teams = Team.all()
        data = []
        for team in teams:
            x = Team(team)
            data.append((team, x.epa_per_play_def('mean', **kwargs)))
        return data

    @classmethod
    def positive_play_percentage_offense(cls, **kwargs):

        teams = Team.all()
        data = []
        for team in teams:
            x = Team(team)
            data.append((team, x.positive_play_percentage_offense(**kwargs)))
        return data

    @classmethod
    def positive_play_percentage_defense(cls, **kwargs):

        teams = Team.all()
        data = []
        for team in teams:
            x = Team(team)
            data.append((team, x.positive_play_percentage_defense(**kwargs)))
        return data

    @classmethod
    def qb_pressure_offense(cls, **kwargs):

        teams = Team.all()
        data = []
        for team in teams:
            x = Team(team)
            data.append((team, x.qb_pressure_offense(**kwargs)))
        return data

    @classmethod
    def qb_pressure_defense(cls, **kwargs):

        teams = Team.all()
        data = []
        for team in teams:
            x = Team(team)
            data.append((team, x.qb_pressure_defense(**kwargs)))
        return data

    @classmethod
    def outliers(cls, ls):
        """
        Takes a list of tuples, or a list of lists of all team values for a given statistic.
        Sub lists or tuples must take the form ('TEAM', statistic), where 'TEAM' is a string and statistic an int or float.
        Calculates the mean and standard deviation.
        Returns a dict of the outliers
        """

        outliers = {"above": {}, "below": {}}
        nums = []

        for x in ls:
            nums.append(x[1])

        std = np.std(nums)
        mean = np.mean(nums)

        for x in ls:
            if x[1] > mean + std:
                outliers["above"][x[0]] = x[1]
            if x[1] < mean - std:
                outliers["below"][x[0]] = x[1]
        return outliers


teams_list = Team.all()

# my_df = pd.DataFrame(index=teams_list)

# z = []
# for x in teams_list:
#     q = Team(x)
#     res = q.epa_per_play_off('mean', quarter=1, down=1, play='pass')
#     z.append((q.__repr__(), res))
#     # z.append((q.__repr__(), res['median'], res['mean'], res['std_dev']))

# z = sorted(z, key=lambda x: x[1])
# for key in z:
#     print(key)

# teams = Team.all()
# stuff = []
# for team in teams:
#     x = Team(team)
#     diff = x.qb_pressure_offense(quarter=1) - x.qb_pressure_offense()
#     stuff.append((team, round(diff, 3)))

# # stuff = League.epa_per_play_def(play='run', quarter=1)
# stuff = sorted(stuff, key=lambda x: x[1], reverse=True)
# stuff = League.outliers(stuff)
# print(stuff)

ne = Team('NE')
x = ne.qb_pressure_offense()
print(x)

y = League.epa_per_play_def(down=1, play='pass')
y = League.outliers(y)
print(y)