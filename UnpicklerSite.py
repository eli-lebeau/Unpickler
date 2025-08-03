'''
The following module is a webapp meant to evaluate submissions to the MLB Pickle Game
based on their skill and luck factor.
The object of the game is to find the mystery player based on information gained
from the previous guesses.
The module uses expected value calculations to estimate how much information a given 
guess is expected to gain for the player to quantify the quality of a guess and to 
create an optimal solution. The strategy based on expected value always manages to find the 
correct player within seven guesses and can find the correct guess in correct answer in five 
or fewer guesses 98.6% of the time and in four or less guesses 90.2% of the time at the last
time of testing. The player database changes as players switch teams and age, so these
statistics may shift over time.
'''


import pandas as pd
import numpy as np
import streamlit as st

# read in the required files. The playerlist.csv is list of all players in the game
# the columns are 'Name', 'Team', 'Division', 'Bats', 'Throws', 'Born', 'Age', 'Position', 'League', and 'Div'
# The last two columns are not independent in the actual game, and are contained
# together in 'Division'. For example, the 'Division' column may contain a value 'AL East',
# which would then be split into 'AL' in 'League' and 'East' in 'Div'
player_list = pd.read_csv('./playerlist.csv')

# the playerlist_withxp_sorted.csv is the same list but sorted by the expected number
# of remaining players left after guessing that player as the initial guess for
# each player
# this is calculated independently so as to not to slow down the running of the app
updated_player_list_xpsorted = pd.read_csv('./playerlist_withxp_sorted.csv')

# columns of player_list excluding 'Division' 
columns_of_interest = ['Name', 'Team', 'Bats', 'Throws', 'Born', 'Age', 'Position', 'League', 'Div']
# columns that a player either knows the exact correct value or doesn't known anything, as opposed to 'Age', 'League', and 'Div'
binary_columns = ['Name', 'Team', 'Bats', 'Throws', 'Born', 'Position']
# A 0 means that we have not made a guess yet for which the guess's value in that column
# matches that of the answer
# A 1 means that we have made such a guess. Certain columns can also take on the value 2
current_information = {'Name': 0, 'Team': 0, 'Bats': 0, 'Throws': 0, 'Born': 0, 'Age': 0, 'Position': 0,
                     'League': 0, 'Div': 0}
# all possible values for each value in current_information
events_of_interest = {'Name': [0,1], 'Team': [0,1], 'Bats': [0,1], 'Throws': [0,1], 'Born': [0,1], 'Age': [0,1,2], 'Position': [0,1],
                     'League': [0,1,2], 'Div': [0,1,2]}

# guess is a single row from the player_list dataframe that contains all
# information on a player. answer is the same format. 
# returns an updated dictionary new_info based on the information gained from guessing guess
def find_info(guess, current_information, answer):
    new_info = current_information.copy()
    for col in new_info:
        if (guess[col].iloc[0] == answer[col].iloc[0]):
            new_info[col] = 1
        else:
            new_info[col] = 0
    # League and Div may take on the value of 2 if we have made a guess for which one of these columns
    # is a match, but the other is not. A 0 means neither is a match, and a 1 means both are a match
    # in the actual game, one would not know whether League or Div is the match if only one is a match
    if (new_info['League'] != new_info['Div']):
        new_info['League'] = 2
        new_info['Div'] = 2
    # Age may take on the value of 2 if the guess's age does not equal the answer's age, but is within two years
    if (new_info['Age'] == 0):
        if (answer['Age'].iloc[0] - 2 <= guess['Age'].iloc[0] and answer['Age'].iloc[0] + 2 >= guess['Age'].iloc[0]):
            new_info['Age'] = 2
    return new_info


# given the current remaining possible players stored in df and the current known infomation info
# relative to the most recent guess and the values of the most recent guess, filter the remaining
# players to only leave those that align with the current information
def find_remaining_players(df, info, guess):
    remaining_players = df.copy()
    for col in binary_columns:
        remaining_players = binary_event(remaining_players, col, info[col], guess)

    # for age, a 0 excludes not only players with exact same age, but also players with an age
    # within two years of that of the guess
    if (info['Age'] == 0):
        remaining_players = remaining_players[(remaining_players['Age'] - 2 > guess['Age'].iloc[0])
                        | (remaining_players['Age'] + 2 < guess['Age'].iloc[0])]
    elif (info['Age'] == 1):
        remaining_players = remaining_players[remaining_players['Age'] == guess['Age'].iloc[0]]
    # a two means that the answer's age must be within two years of the age of the current guess, but
    # not the same as that of the current guess
    else:
        remaining_players = remaining_players[(remaining_players['Age'] - 2 <= guess['Age'].iloc[0])
                        & (remaining_players['Age'] + 2 >= guess['Age'].iloc[0])
                        & ((remaining_players['Age'] != guess['Age'].iloc[0]))]
    # A zero means that the current gues and answer share neither League or Div
    if (info['League'] == 0):
        remaining_players = remaining_players[(remaining_players['League'] != guess['League'].iloc[0])
                        & (remaining_players['Div'] != guess['Div'].iloc[0])]
    # A one means that the current guess and answer share both
    elif (info['League'] == 1):
        remaining_players = remaining_players[((remaining_players['League'] == guess['League'].iloc[0])
                        & (remaining_players['Div'] == guess['Div'].iloc[0]))]
    # A two means they share one but not the other, but the user does not know which
    # attribute is shared, so an exclusive or is used
    else:
        remaining_players = remaining_players[(remaining_players['League'] == guess['League'].iloc[0])
                        ^ (remaining_players['Div'] == guess['Div'].iloc[0])]
    return remaining_players


# given an initial set of remaining players old_df, the column col of interest, 
# the event of interest, and the current guess find the remaining players
# after filtering for the event under the current information
# Only for use with events that can take on only 0 or 1 as their value
def binary_event(old_df, col, event, guess):
    if(event):
        new_df = old_df[old_df[col] == guess[col].iloc[0]]
    else:
        new_df = old_df[old_df[col] != guess[col].iloc[0]]
    return new_df


# for the given guess, a dataframe for one player with all of their relevant information,
# compute the expected number of remaining players that could possibly be the answer
# after making this guess given the current remaining players in the dataframe current_player_list
# Note: each player in current_player_list is assumed to have equal probability of being the answer
def find_xp(current_player_list, guess):
    # test all possible events except for the Name being correct, because if this is the case, then
    # the number of remaining possible players is 0, so this would not contribute to the expected value
    # calculation anyway
    events_of_interest = {'Name': [0], 'Team': [0,1], 'Bats': [0,1], 'Throws': [0,1], 'Born': [0,1], 'Age': [0,1,2], 'Position': [0,1],
                     'League': [0,1,2], 'Div': [0,1,2]}
    remaining_players = current_player_list.copy()
    initial_remaining = len(remaining_players)
    # initialize the expected number of remaining players to 0
    xp = 0
    
    # Rather than finding the remaining players left after applying all the events of interest for each column, they are applied individually
    # If after any application of any event there are no remaining players, the function moves to the next event of interest for that column
    # without completing the remaining iterations of the nested for loops. The number of remaining players at the end of the last nest for loop
    # divided by the number of initial players is the probability that this set of events will be the result.
    # Applying the events individually and reusing the filtering between different sets of events of interest
    # does not throw off the results because of the chain rule of probability as for events N, Te, Ba, Tr, Bo, A, P, L, D
    # P(N & Te & Ba & Tr & Bo & A & P & L & D) = P(N)*P(Te|N)*P(Ba|NTe)*P(Tr|NTeBa)*P(Bo|NTeBaTr)*P(A|NTeBaTrBo)*P(P|NTeBaTrBo)*P(L|NTeBaTrBoAP)*P(D|NTeBaTrBoAPL)
    for event in events_of_interest['Name']:
        name_df = binary_event(remaining_players, 'Name', event, guess)
        if (name_df.empty):
            continue
        
        for event in events_of_interest['Team']:
            team_df = binary_event(name_df, 'Team', event, guess)
            if (team_df.empty):
                continue
            
            for event in events_of_interest['Bats']:
                bats_df = binary_event(team_df, 'Bats', event, guess)
                if (bats_df.empty):
                    continue
                
                for event in events_of_interest['Throws']:
                    throws_df = binary_event(bats_df, 'Throws', event, guess)
                    if (throws_df.empty):
                        continue
                    
                    for event in events_of_interest['Born']:
                        born_df = binary_event(throws_df, 'Born', event, guess)
                        if (born_df.empty):
                            continue
                        
                        # Age and League/Div require more extensive logic as they
                        # are not binary events. See find_remaining_players for further information
                        for event in events_of_interest['Age']:
                            if (event == 0):
                                age_df = born_df[(born_df['Age'] - 2 > guess['Age'].iloc[0])
                                                | (born_df['Age'] + 2 < guess['Age'].iloc[0])]
                            elif (event == 1):
                                age_df = binary_event(born_df, 'Age', event, guess)
                            else:
                                age_df = born_df[(born_df['Age'] - 2 <= guess['Age'].iloc[0])
                                                & (born_df['Age'] + 2 >= guess['Age'].iloc[0])
                                                & (born_df['Age'] != guess['Age'].iloc[0])]
                            if (age_df.empty):
                                continue
    
                            for event in events_of_interest['Position']:
                                pos_df = binary_event(age_df, 'Position', event, guess)
                                if (pos_df.empty):
                                    continue
    
                                for event in events_of_interest['League']:
                                    if (event == 0):
                                        division_df = pos_df[(pos_df['League'] != guess['League'].iloc[0])
                                                        & (pos_df['Div'] != guess['Div'].iloc[0])]
                                    elif (event == 1):
                                        division_df = binary_event(pos_df, 'Division', event, guess)
                                    else:
                                        division_df = pos_df[(pos_df['League'] == guess['League'].iloc[0])
                                                        ^ (pos_df['Div'] == guess['Div'].iloc[0])]
                                    if (division_df.empty):
                                        continue
                                    # compute the final probability and update the expected value computation
                                    num_remaining = len(division_df)
                                    current_prob = num_remaining/initial_remaining
                                    xp += num_remaining * current_prob
    # Using this method rather than the naive approach of looping over every remaining player as the possible answer and finding the 
    # number of remaining players reduces the runtime of this function from ~1.5 seconds to ~0.06 seconds for the initial list of 
    # over a 1300 players. This initial computation only need be performed every time the initial list is updated, and so it 
    # is not done during the actual execution of the webapp, but each subsequent use of this method after the intial guess
    # will also be faster. The naive approach requires over 1300 iterations in all cases, while this approach requires
    # 288 iterations in the worst case, but often requires fewer due to the continue statements
    return xp

# for every player in the player_list dataframe, comput the expected number of remaining players that could be the answer
# if that player were guessed
# return an updated copy of player_list with the new information
def update_player_list_withxps(player_list):
    xps = np.zeros(len(player_list))
    # a for loop rather than df.apply() is used for ease of debugging should something in the find_xp() fail
    for i in range(len(player_list)):
        guess = player_list.iloc[i:i+1]
        xps[i] = find_xp(player_list, guess)
    updated_player_list = player_list.copy()
    updated_player_list['xp'] = xps
    return updated_player_list


# compute the skill score of a guess as the percentile score for current_guess_xp, a given player's expected number of remaining players (xp)
# should that player be guessed compared to the xp for each player should they be guessed 
def get_percentile_score(xps, current_guess_xp):
    # Due to floating point precision errors in computing the expected number of remaining players
    # a simple == operation should not be used.
    # if current_guess_xp is the minimal xp, even if other guesses have the same xp, award the highest score of 99.9
    # otherwise compute the percentile
    if (abs(xps.min() - current_guess_xp) < 1e-6):
        percentile_rank = 99.9
    else:
        percentile_rank = np.round(np.sum(xps > current_guess_xp) / len(xps) * 100, 1)
    return percentile_rank

# for use in the luck score computation
# for a given guess, find the number of players that would remain after guessing that player for each
# possible remainig answer in remaining_players
def possible_remaining_finder(remaining_players, guess):
    possible_rems = []
    for i in range(len(remaining_players)):
        # update the player list
        current_information = {'Name': 0, 'Team': 0, 'Bats': 0, 'Throws': 0, 'Born': 0, 'Age': 0, 'Position': 0,
                     'League': 0, 'Div': 0}
        prov_answer = remaining_players.iloc[i:i+1]
        current_information = find_info(guess, current_information, prov_answer)
        remaining_guesses = find_remaining_players(remaining_players, current_information, guess)
        # if the only remaining player is the current guess, then there would be no remaining players
        if len(remaining_guesses) == 1:
            if (guess.Name.iloc[0] == remaining_guesses.Name.iloc[0]):
                possible_rems.append(0)
            else:
                possible_rems.append(1)
        else:
            possible_rems.append(len(remaining_guesses))
    return np.array(possible_rems)

# given the an array of the possible number of remaining players possible_num_remaining_players after making a 
# guess and the actual number of remaining players actual_num_remaining, compute the percentile score of
# actual_num_remaining within possible_num_remaining_players
# 50 means neutral luck as the actual number was in the middle of the possible numbers. 
# 99.9 means high luck and 0.0 means bad luck.
# if there is only one possible number of remaining players, return 'None' as there is no
# luck component in the result of the guess
def compute_luck_score(actual_num_remaining, possible_num_remaining_players):
    if len(np.unique(possible_num_remaining_players)) == 1:
        return 'None'
    return np.round(np.sum(possible_num_remaining_players > actual_num_remaining) / len(possible_num_remaining_players) * 100, 1)

# when choosing the optimal guess by xp from the dataframe remaining_guesses
# choose the guess with the lowest xp. If multiple guesses have the minimal xp,
# always choose one that is also possibly the answer if there is such a one
def get_optimal_guess(remaining_guesses, xps, remaining_players):
    optimal_guesses = remaining_guesses.iloc[np.where(xps == xps.min())]
    # find the optimal guesses that are also possibly the answer
    optimal_guess_names = set(optimal_guesses.Name)
    remaining_names = set(remaining_players.Name)
    optimal_possible_guesses = optimal_guess_names.intersection(remaining_names)

    # if there is a guess or guesses that fit these conditions, randomly choose one
    if (len(optimal_possible_guesses)):
        optimal_possible_guesses = list(optimal_possible_guesses)
        return optimal_guesses[optimal_guesses['Name'] == np.random.choice(optimal_possible_guesses)]
    # otherwise, choose the first option among the possible guesses with the lowest xp
    else:
        return optimal_guesses.head(1)

# Perhaps the most important method for webapp interface
# Given the inputted guesses to the webapp and the answe, evaluate the user's guesses and 
# find the optimal guess that the unpickler would guess given the current information
# display all information in tables on the webapp
# returns the average skill and luck scores on all qualifying guesses
def grade_guesses(guesses, answer, player_list = player_list, updated_player_list_xpsorted = updated_player_list_xpsorted):
    # intialize game, no information, all players are possible answers
    current_information = {'Name': 0, 'Team': 0, 'Bats': 0, 'Throws': 0, 'Born': 0, 'Age': 0, 'Position': 0,
                             'League': 0, 'Div': 0}
    remaining_guesses = updated_player_list_xpsorted.copy()
    remaining_players = player_list.copy()

    # track the cumulative skill and luck scores for average computation
    cumulative_skill_score = 0
    cumulative_luck_score = 0
    

    num_guesses = len(guesses)

    # not all guesses receive a numerical luck score, which happens if there is only one possible 
    # remaining number of players after making the guess as there is no luck component
    # in how many players remain after making the guess. To compute the average luck score of guesses
    # with a luck score, it is necessary to subtract the number of guesses without luck scores
    # from the total number of guesses
    num_guesses_with_luck_score = num_guesses

    for player_guess in range(num_guesses):
        current_guess = remaining_guesses[remaining_guesses['Name'] == guesses[player_guess]]
        # find the expected number of remaining players (xp) given the user's guess
        current_guess_xp = current_guess.xp.iloc[0]
        
        xps = np.array(remaining_guesses.xp)
        # round the xps to avoid issues related to the imprecision of the floating point multiplication
        # 6 decimal places are sufficient
        rounded_xps = np.round(xps, 6)

        # find the unpickler's optimal guess
        unpickler_guess = get_optimal_guess(remaining_guesses, xps, remaining_players)
        unpickler_guess_xp =  unpickler_guess.xp.iloc[0]

        # compute the percentile rank for the user's guess and the unpickler's, the unpickler's should always be 99.9
        skill_score = get_percentile_score(rounded_xps, current_guess_xp)
        cumulative_skill_score += skill_score
        unpickler_skill_score = get_percentile_score(rounded_xps, unpickler_guess_xp)

        # find the possible number of remaining players given each possible answer for both guesses
        possible_num_remaining_players = possible_remaining_finder(remaining_players, current_guess)
        possible_num_remaining_players_unpickler = possible_remaining_finder(remaining_players, unpickler_guess)

        # find the updated list of possible answers (remaining_players) given the unpickler's guess
        current_information_unpickler = find_info(unpickler_guess, current_information, answer)
        remaining_players_unpickler = find_remaining_players(remaining_players, current_information_unpickler, unpickler_guess)

        # find the updated list of possible answers (remaining_players) given the unpickler's guess
        # Also update the list of remaining players that could be the answer
        current_information = find_info(current_guess, current_information, answer)
        remaining_players = find_remaining_players(remaining_players, current_information, current_guess)

        # if the only remaining possible answer is the current guess, change the number remaining to 0
        actual_num_remaining = len(remaining_players)
        if (actual_num_remaining == 1):
            if (current_guess.Name.iloc[0] == remaining_players.Name.iloc[0]):
                actual_num_remaining = 0

        actual_num_remaining_unpickler = len(remaining_players_unpickler)
        if (actual_num_remaining_unpickler == 1):
            if (unpickler_guess.Name.iloc[0] == remaining_players_unpickler.Name.iloc[0]):
                actual_num_remaining_unpickler = 0

        # compute the luck scores for the guesses
        luck_score = compute_luck_score(actual_num_remaining, possible_num_remaining_players)

        # if guess does not have a luck score, compute_luck_score() returns the string 'None'
        # if the return value is not a string, it must be a numerical type, and so it is
        # added to the  cumulative_luck_score
        if (type(luck_score) != str):
            cumulative_luck_score += luck_score
        # if there is no numerical luck score, update the count for number of guesses that have
        # a luck score
        else:
            num_guesses_with_luck_score -= 1
        luck_score_unpickler = compute_luck_score(actual_num_remaining_unpickler, possible_num_remaining_players_unpickler)

        # create a dataframe to summarize all metrics for the user's and the unpickler's guesses
        current_guess_ratings = pd.DataFrame(index = ['Your Guess', 'Unpickler Guess Given Current Information'], 
                                             data = {'Name': [guesses[player_guess], unpickler_guess.Name.iloc[0]], 
                                 'Skill Score': [skill_score, unpickler_skill_score],
                                 # compute the rank for the skill scores so that if two players are tied for first, 
                                 # the next player is ranked as third, not second, for example
                                 'Guess skill rank': [np.sum(rounded_xps < current_guess_xp) + 1, np.sum(rounded_xps < unpickler_guess_xp) + 1],
                                 'Luck Score':  [luck_score, luck_score_unpickler],
                                 'Remaining Players': [actual_num_remaining, actual_num_remaining_unpickler]}).T
        # cast columns as string to avoid warning from streamlit since they contain values of different types
        current_guess_ratings['Your Guess'] = current_guess_ratings['Your Guess'].astype(str)
        current_guess_ratings['Unpickler Guess Given Current Information'] = current_guess_ratings['Unpickler Guess Given Current Information'].astype(str)

        st.subheader('Guess ' + str(player_guess + 1) + ': '+ guesses[player_guess])
        st.dataframe(data = current_guess_ratings)
        st.write()


        # display all possible guesses ranked by their xps and and with theur xps
        st.write('Ranked guesses:')

        remaining_guesses_display = remaining_guesses[['Name', 'xp']].copy()
        remaining_guesses_display['xp'] = remaining_guesses_display['xp'].round(6)
        
        # find the percentil score for each possible guess and its skill rank
        remaining_guesses_display['Skill Score'] = remaining_guesses_display.apply(
            lambda row: get_percentile_score(rounded_xps, row['xp']), axis = 1
        )
        remaining_guesses_display['Rank'] = remaining_guesses_display.apply(lambda row: np.sum(rounded_xps < row['xp']) + 1, axis = 1)
        remaining_guesses_display.rename(columns = {'xp':'Expected Possible Players Remaining'}, inplace=True)
        remaining_guesses_display.sort_values('Expected Possible Players Remaining', inplace=True)
        
        # rearrange the columns
        remaining_guesses_display = remaining_guesses_display[['Rank', 'Name', 'Skill Score', 'Expected Possible Players Remaining']]
        st.dataframe(data = remaining_guesses_display, hide_index = True)

        
        # update the remaining possible guesses and calculate the new xps given the remaining players after the user's guess
        remaining_guesses.drop(current_guess.index, inplace=True)
        remaining_guesses.reset_index(inplace=True)
        remaining_guesses.drop('index', axis = 1, inplace=True)    
    
        xps = np.zeros(len(remaining_guesses))
        for i in range(len(remaining_guesses)):
            guess = remaining_guesses.iloc[i:i+1]
            xps[i] = find_xp(remaining_players, guess)
    
        remaining_guesses.drop('xp', axis = 1, inplace = True)
        remaining_guesses['xp'] = xps
    
    # compute and return the average skill and luck scores
    return np.round(cumulative_skill_score/num_guesses,2), np.round(cumulative_luck_score/num_guesses_with_luck_score, 2)

# given the unpickler's guesses (a list of strings) to find the answer, grade the guesses
# see grade_guesses function for more information as these functions are similar
# returns the average skill and luck scores on all qualifying guesses
def grade_unpickler(guesses, answer, player_list = player_list, updated_player_list_xpsorted = updated_player_list_xpsorted):
    current_information = {'Name': 0, 'Team': 0, 'Bats': 0, 'Throws': 0, 'Born': 0, 'Age': 0, 'Position': 0,
                             'League': 0, 'Div': 0}
    remaining_guesses = updated_player_list_xpsorted.copy()
    remaining_players = player_list.copy()
   
    cumulative_skill_score = 0
    cumulative_luck_score = 0
    
    # dictionary that will hold information for each guess
    guess_ratings = {}
    num_guesses = len(guesses)
    num_guesses_with_luck_score = num_guesses
    
    for guess_num in range(num_guesses):
        # for each each guess, find the percentile score, luck socre, guess 
        current_guess = remaining_guesses[remaining_guesses['Name'] == guesses[guess_num]]
    
        current_guess_xp = current_guess.xp.iloc[0]
        
        xps = np.array(remaining_guesses.xp)
        rounded_xps = np.round(xps, 6)

        skill_score = get_percentile_score(rounded_xps, current_guess_xp)
        cumulative_skill_score += skill_score
        possible_num_remaining_players = possible_remaining_finder(remaining_players, current_guess)

        current_information = find_info(current_guess, current_information, answer)
        # update list of remaining players
        remaining_players = find_remaining_players(remaining_players, current_information, current_guess)

        actual_num_remaining = len(remaining_players)
        if (actual_num_remaining == 1):
            if (current_guess.Name.iloc[0] == remaining_players.Name.iloc[0]):
                actual_num_remaining = 0

        luck_score = compute_luck_score(actual_num_remaining, possible_num_remaining_players)
        
        if (type(luck_score) != str):
            cumulative_luck_score += luck_score
        else:
            num_guesses_with_luck_score -= 1

        # add information to dictionary with the guess name as the key
        guess_ratings['Guess '+str(guess_num+1)] = [guesses[guess_num], skill_score, np.sum(rounded_xps < current_guess_xp) + 1, luck_score, actual_num_remaining]


        remaining_guesses.drop(current_guess.index, inplace=True)
        remaining_guesses.reset_index(inplace=True)
        remaining_guesses.drop('index', axis = 1, inplace=True)    

        # update the remaining possible guesses and calculate the new xps given the remaining players after the user's guess
        xps = np.zeros(len(remaining_guesses))
        for i in range(len(remaining_guesses)):
            guess = remaining_guesses.iloc[i:i+1]
            xps[i] = find_xp(remaining_players, guess)

        remaining_guesses.drop('xp', axis = 1, inplace = True)
        remaining_guesses['xp'] = xps

    # create a data frame from the dictionary to display the results of the unpickler's guesses
    all_ratings = pd.DataFrame(index = ['Name', 'Skill Score', 'Guess skill rank', 'Luck Score', 'Remaining Players'], data = guess_ratings)


    # cast columns as string to avoid warning from streamlit since they contain values of different types
    for col in all_ratings.columns:
        all_ratings[col] = all_ratings[col].astype(str)

    # display the results of the unpickler's guesses
    st.dataframe(data = all_ratings)

    # compute and return the average skill and luck scores
    return np.round(cumulative_skill_score/num_guesses, 2), np.round(cumulative_luck_score/num_guesses_with_luck_score, 2)

# find the set of guesses the unpickler uses to find the answer
# These should be our 'optimal solution' using our heuristic that the best
# guess is the guess with the lowest expected number of remaining possible answers (xp)
# after making that guess
def get_guesses(answer, updated_player_list = updated_player_list_xpsorted):
    guesses = []
    # begin with 0 knowledge
    current_information = {'Name': 0, 'Team': 0, 'Bats': 0, 'Throws': 0, 'Born': 0, 'Age': 0, 'Position': 0,
                         'League': 0, 'Div': 0}
    
    # find the guess with the lowest xp as the initial guess
    # all guesses are possible answers at this point, so it is unnecessary to use get_optimal_guess()
    current_guess = updated_player_list.sort_values('xp').head(1)
    guesses.append(current_guess['Name'].iloc[0])

    # update the remaining guess options and the remaining player possible answer list after making the guess
    remaining_guesses = updated_player_list.drop(current_guess.index).reset_index().copy()
    remaining_guesses.drop('index', axis = 1, inplace=True)
    current_information = find_info(current_guess, current_information, answer)
    remaining_players = find_remaining_players(player_list, current_information, current_guess)

    # until there is one remaining player, find the new xps for each player and choose the 
    # optimal guess using get_optimal_guess()
    while len(remaining_players) > 1:
        xps = np.zeros(len(remaining_guesses))
        for i in range(len(remaining_guesses)):
            guess = remaining_guesses.iloc[i:i+1]
            xps[i] = find_xp(remaining_players, guess)
        
        remaining_guesses.drop('xp', axis = 1, inplace = True)
        remaining_guesses['xp'] = xps

        current_guess = get_optimal_guess(remaining_guesses, xps, remaining_players)
        
        guesses.append(current_guess['Name'].iloc[0])

        # update remaining guesses and remaining players
        remaining_guesses.drop(remaining_guesses[remaining_guesses['Name'] == guesses[-1]].index, inplace=True)
        remaining_guesses.reset_index(inplace=True)
        remaining_guesses.drop('index', axis = 1, inplace=True)
        current_information = find_info(current_guess, current_information, answer)
        remaining_players = find_remaining_players(remaining_players, current_information, current_guess)
    
    # if there is one remaining player, if the current_guesses is not the same as
    # the remaining player, the remaining player would be the next guess, but if they
    # are the same, there is no need to add anything to the guesses list
    # checking for this possibility outside of the loop rather than doing it as the previous gueses
    # were done saves a set of computations for finding the xps for each player as this guess
    # would have xp 0 and all others would have xp 1
    if current_guess.Name.iloc[0] != remaining_players.Name.iloc[0]:
        guesses.append(remaining_players['Name'].iloc[0])

    return guesses


# web interface
st.title('Unpickler ⚾')
st.write('Weclome to the Unpickler! The Unpickler will evaluate your guesses' \
'for the MLB Pickle game. Please see the Demo tab for how to use the tool and '
'the Methodology section for how calcuations are done. The Unpickler is not affiliated with '
'MLB Pickle. It is an independent project. All player information used for player attributes (e.g. names, ages, etc.)'
'are sourced from MLB Pickle, but all calculations and figures presented are the product'
'of the Unpickler.')
    


player_names = list(player_list['Name'])


# the website is split into an interactive home where users can input their guesses, 
# a static demo where users can see an example of how the site works
# and a methodology section that explains how each computation is done
home, demo, methodology = st.tabs(["Home", "Demo", "Methodology"])
with home:
    # for formating, four columns are made, though three are used
    col1, col2, col3, col4 = st.columns([1,1,0.1,0.9])

    #selectbox for a single value for user to select the answer from the list of players
    with col1:
        st.write('Please enter the answer to today\'s game.')
        ans = st.selectbox(
            label = 'Please enter the answer to today\'s game.',
            options =  player_names,
            index = None,
            placeholder = 'Answer',
            label_visibility = 'collapsed',
            key = 'answer'
        )

    # find the dataframe of the row associated with the selected player
    answer = player_list[player_list['Name'] == ans]

    with col4:
        # the submit button is placed here so that if pressed, it disables the multiselect box in col2
        # add blank lines to align the button with the other elements on the page
        for i in range(4):
            st.write('')
        done = st.button(label = 'Submit', key = 'done_button')

    with col2:
        # The user enters their guesses into the multiselect box
        # the number of guesses is capped at 9 as in the real game
        # The box is disabled if the submit button is pressed or if no answer has been inputed yet
        st.write('Please input your guesses in the order you made them.')
        guesses = st.multiselect(
            label = 'Please enter your guesses',
            options = player_names,
            placeholder = 'Enter Guess',
            label_visibility = 'collapsed',
            max_selections = 9,
            disabled = (ans == None or done),
        )
    
    # if the submit button is pressed and guesses have been entered, display the results
    # of the grader
    if (done and len(guesses)):
        st.subheader('Results')
        st.write('Skill score is on a scale from 0.0 to 99.9 where 99.9 is the highest skill.')
        st.write('Luck score is also on a scale from 0.0 to 99.9. Not all guesses have a luck score  ' \
        'for some guesses, there is only one possible outcome, so luck does not play a role.')
        
        # displays the results of grading the guesses
        user_average_skill_score, user_average_luck_score = grade_guesses(guesses, answer)

        st.write('Average Skill Score: '+str(user_average_skill_score))
        st.write('Average Luck Score: '+str(user_average_luck_score))

        # finds the unpickler guesses and displays them and their associated grades
        # displays a message for who had fewer guesses, the user or the unpickler
        st.subheader('Did you beat the Unpickler?')
        st.write('Unpickler Guesses:')
        unpickler_guesses = get_guesses(answer)
        unpickler_average_skill_score, unpickler_average_luck_score = grade_unpickler(unpickler_guesses, answer)

        st.write('Unpickler Average Skill Score: '+str(unpickler_average_skill_score))
        st.write('Unpickler Average Luck Score: '+str(unpickler_average_luck_score))

        st.write('')

        num_unpickler_guesses = len(unpickler_guesses)
        num_user_guesses = len(guesses)
        st.write('Total Unpickler Guesses: '+str(num_unpickler_guesses))
        # check if the user found the mystery player in the end
        if (guesses[-1] == ans):
            st.write('Your Total Guesses: '+str(num_user_guesses))
            if (num_unpickler_guesses < num_user_guesses):
                st.write('The Unpickler beat you today')
            elif (num_unpickler_guesses > num_user_guesses):
                st.write('You beat the Unpickler. Great Job!')
            else:
                st.write('You tied the Unpickler. Nice Job!')
        else:
            st.write('The Unpickler beat you today')

        st.write('')

        if (unpickler_average_luck_score < user_average_luck_score):
            st.write('You had better luck than the Unpickler today.')
        elif (unpickler_average_luck_score > user_average_luck_score):
            st.write('The Unpickler had better luck than you today.')
        else:
            st.write('You and the Unpickler had equal luck today.')
        


with demo:
    st.write('The demo displays the results from running the Unpickler on the current answer and set of guesses. You can ' \
    'modify the current answer and set of guesses to familiarize yourself with the different input fields, but to submit' \
    'your entries, please return to the home page.')

    # same setup as the home page, but with the answers to the boxes pre-filled and
    # the submit button clicked
    col1, col2, col3, col4 = st.columns([1,1,0.1,0.9])

    with col1:
        st.write('Please enter the answer to today\'s game.')
        demo_answer = 'Randy Arozarena'
        default_selection_index = player_names.index(demo_answer)
        demo_ans = st.selectbox(
            label = 'Please enter the answer to today\'s game.',
            options =  player_names,
            label_visibility = 'collapsed',
            key = 'demo_answer',
            index = default_selection_index,
            placeholder = 'Answer'
        )
    demo_ans = demo_answer

    demo_answer = player_list[player_list['Name'] == demo_answer]

    with col4:
        # write a blank line to align the submit button with the multiselectbox
        
        for i in range(4):
            st.write('')
        demo_done = st.button(label = 'Submit', key = 'demo_done_button')

    with col2:
        # default answers
        default = ['Kris Bryant', 'Isaac Paredes', 'Osvaldo Bido', 'Eduard Bazardo', 'Randy Arozarena']
        st.write('Please input your guesses in the order you made them.')
        demo_guesses = st.multiselect(
            label = 'Please enter your guesses',
            options = player_names,
            placeholder = 'Enter Guess',
            label_visibility = 'collapsed',
            max_selections = 9,
            disabled = demo_ans == None,
            default = default
        )
        demo_guesses=default

    st.subheader('Results')
    st.write('Skill score is on a scale from 0.0 to 99.9 where 99.9 is the highest skill.')
    st.write('Luck score is also on a scale from 0.0 to 99.9. Not all guesses have a luck score because ' \
    'for some guesses, there is only one possible outcome, so luck does not play a role.')

    # rather than recomputing the demo results each time, the results are stored in csv files
    # and then displayed here with some formatting
    num_user_guesses = len(demo_guesses)
    for player_guess in range(num_user_guesses):
        st.subheader('Guess ' + str(player_guess + 1) + ': '+ demo_guesses[player_guess])
        current_guess_ratings = pd.read_csv('./DemoFiles/demo_guess'+str(player_guess+1)+'_rating.csv')
        current_guess_ratings.rename(columns={'Unnamed: 0':''}, inplace = True)
        st.dataframe(data = current_guess_ratings, hide_index = True)
        st.write()

        st.write('Ranked guesses:')
        remaining_guesses_display = pd.read_csv('./DemoFiles/demo_guess'+str(player_guess+1)+'_remaining_guesses.csv')
        st.dataframe(data = remaining_guesses_display[['Rank', 'Name', 'Skill Score', 'Expected Possible Players Remaining']], hide_index = True)

    demo_average_scores = pd.read_csv('./DemoFiles/demo_averages.csv')

    demo_user_average_luck = demo_average_scores['user_average_luck'].iloc[0]

    st.write('Average Skill Score: '+str(demo_average_scores['user_average_skill'].iloc[0]))
    st.write('Average Luck Score: '+str(demo_user_average_luck))
    
    # displays the results of the unpickler's guesses for the demo answer
    st.subheader('Did you beat the Unpickler?')
    st.write('Unpickler Guesses:')
    demo_ratings = pd.read_csv('./DemoFiles/demo_unpickler_guesses.csv')
    demo_ratings.rename(columns={'Unnamed: 0':''}, inplace = True)
    demo_ratings.set_index('', inplace = True)
    st.dataframe(data = demo_ratings)

    demo_unpickler_average_luck = demo_average_scores['unpickler_average_luck'].iloc[0]
    st.write('Unpickler Average Skill Score: '+str(demo_average_scores['unpickler_average_skill'].iloc[0]))
    st.write('Unpickler Average Luck Score: '+str(demo_unpickler_average_luck))

    st.write('')

    # displays a message for who had fewer guesses, the user or the unpickler
    num_unpickler_guesses = len(demo_ratings.columns)
    st.write('Total Unpickler Guesses: '+str(num_unpickler_guesses))
    
    st.write('Your Total Guesses: '+str(num_user_guesses))
    if (num_unpickler_guesses < num_user_guesses):
        st.write('The Unpickler beat you today')
    elif (num_unpickler_guesses > num_user_guesses):
        st.write('You beat the Unpickler. Great Job!')
    else:
        st.write('You tied the Unpickler. Nice Job!')

    st.write('')

    if (demo_unpickler_average_luck < demo_user_average_luck):
            st.write('You had better luck than the Unpickler today.')
    elif (demo_unpickler_average_luck > demo_user_average_luck):
        st.write('The Unpickler had better luck than you today.')
    else:
        st.write('You and the Unpickler had equal luck today.')

with methodology:
    st.subheader('Methodology:')
    st.write('Optimal Guess: To compute the optimal guess, we calculate the expected number of remaining possible players that the answer could be' \
    'given our current guess and the information we already have. Expected value takes the weighted average number of remaining players after each guess given' \
    'each remaining possible final answer weighted by the probability that possible answer is the actual final answer. Expected value is not' \
    'a guarantee that a certain number of players will remain after a certain guess, but it remains a good strategy as it prioritizes the guesses'
    'that have the highest probability of revealing the most information in this case. When testing this strategy' \
    'for all possible answers, this strategy never failed to find the answer in more than seven guesses. Furthermore, this strategy found the' \
    'correct answer in five or fewer guesses 98.96% of the time and in four or fewer guesses 92.94% of the time.')
    st.write('')
    st.write('Skill Score: The skill score is the percentile score for your guess with respect to having the lowest expected remaining players' \
    'after guessing that guess. A score of 99.9 indicates that your guess was as good or better than 99.9% of possible guesses, or that it was the best guess' \
    'while any lower score indicates that your guess was strictly better than that percentage of possible guesses.')
    st.write('')
    st.write('Guess Rank: The rank of your guess with respect to having the lowest expected number of remaining guesses after making that guess')
    st.write('')
    st.write('Luck Score: The luck score is the percentile score of the actual number of remaining possible answers after making your guess with respect to ' \
    'all possible numnbers of remaining possible answers given your guess and the possible answers. A luck score of 99.9 means that your guess resulted' \
    'in the lowest number of remaining possible answers given all possible remaining number of players, good luck, while a luck score of 0.0 represents' \
    'that your guess resulted in the highest number of remaining possible players, bad luck. Some guesses do not have luck scores if there is' \
    'only one possible number of remaining possible answers, meaning that there is no luck component in the result of the guess.')