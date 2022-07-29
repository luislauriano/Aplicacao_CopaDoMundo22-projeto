import streamlit as st
import pickle as pkl
import pandas as pd
import numpy as np
from scipy.stats import poisson
import itertools
from matplotlib import pyplot as plt
import matplotlib.image as mpimg

model=pkl.load(open("poisson_model.pkl","rb"))

def predict_grupo(time1, time2):

    def get_proba_match(foot_model, team1, team2, max_goals=10):
        
        t1_goals_avg = foot_model.predict(pd.DataFrame(data={'team': team1, 'opponent': team2}, index=[1])).values[0]
        t2_goals_avg = foot_model.predict(pd.DataFrame(data={'team': team2, 'opponent': team1}, index=[1])).values[0]
        
      
        team_pred = [[poisson.pmf(i, team_avg) for i in range(0, max_goals+1)] for team_avg in [t1_goals_avg, t2_goals_avg]]
        
        #todos as probabilidades de resultados possiveis
        match = np.outer(np.array(team_pred[0]), np.array(team_pred[1]))
        
        
        t1_wins = np.sum(np.tril(match, -1))
        draw = np.sum(np.diag(match))
        t2_wins = np.sum(np.triu(match, 1))
        result_proba = [t1_wins, draw, t2_wins]
        
        
        result_proba =  np.array(result_proba)/ np.array(result_proba).sum(axis=0,keepdims=1)
        team_pred[0] = np.array(team_pred[0])/np.array(team_pred[0]).sum(axis=0,keepdims=1)
        team_pred[1] = np.array(team_pred[1])/np.array(team_pred[1]).sum(axis=0,keepdims=1)
        return result_proba, [np.array(team_pred[0]), np.array(team_pred[1])], match
    
    def get_match_result(foot_model, team1, team2, elimination=False, max_draw=50, max_goals=10):
       
        proba,score_proba, match = get_proba_match(foot_model, team1, team2, max_goals)
        
       
        results = pd.Series([np.random.choice([team1, 'draw', team2], p=proba) for i in range(0,max_draw)]).value_counts()
        result = results.index[0] if not elimination or (elimination and results.index[0] != 'draw') else results.index[1]
        
     
        if (result != 'draw'): 
            i_win, i_loose = (0,1) if result == team1 else (1,0)
            score_proba[i_win] = score_proba[i_win][1:]/score_proba[i_win][1:].sum(axis=0,keepdims=1)
            winner_score = pd.Series([np.random.choice(range(1, max_goals+1), p=score_proba[i_win]) for i in range(0,max_draw)]).value_counts().index[0]
            
            score_proba[i_loose] = score_proba[i_loose][:winner_score]/score_proba[i_loose][:winner_score].sum(axis=0,keepdims=1)
            looser_score = pd.Series([np.random.choice(range(0, winner_score), p=score_proba[i_loose]) for i in range(0,max_draw)]).value_counts().index[0]
            
            
            score = [winner_score, looser_score]
            
     
        else:
            score = np.repeat(pd.Series([np.random.choice(range(0, max_goals+1), p=score_proba[0]) for i in range(0,max_draw)]).value_counts().index[0],2)
        looser = team2 if result == team1 else team1 if result != 'draw' else 'draw'
        
        
        t1_wins = round(np.sum(np.tril(match, -1)),2)
        draw = round(np.sum(np.diag(match)),2)
        t2_wins = round(np.sum(np.triu(match, 1)),2)
            
        porcent = (f'Chances de vitória {team1}: {t1_wins}%, chances de empate: {draw}%, chances de vitória {team2}: {t2_wins}%')
            
        
        return result, looser, score, porcent
        

        

    prediction= get_match_result(model, time1, time2, max_goals=10)
    return prediction




def predict_eliminatoria(time1_eliminatoria, time2_eliminatoria):

    def get_proba_match(foot_model, team1, team2, max_goals=10):
      
        t1_goals_avg = foot_model.predict(pd.DataFrame(data={'team': team1, 'opponent': team2}, index=[1])).values[0]
        t2_goals_avg = foot_model.predict(pd.DataFrame(data={'team': team2, 'opponent': team1}, index=[1])).values[0]
        
        
        team_pred = [[poisson.pmf(i, team_avg) for i in range(0, max_goals+1)] for team_avg in [t1_goals_avg, t2_goals_avg]]
        
        #todos as probabilidades de resultados possiveis
        match = np.outer(np.array(team_pred[0]), np.array(team_pred[1]))
        
      
        t1_wins = np.sum(np.tril(match, -1))
        draw = np.sum(np.diag(match))
        t2_wins = np.sum(np.triu(match, 1))
        result_proba = [t1_wins, draw, t2_wins]
        
      
        result_proba =  np.array(result_proba)/ np.array(result_proba).sum(axis=0,keepdims=1)
        team_pred[0] = np.array(team_pred[0])/np.array(team_pred[0]).sum(axis=0,keepdims=1)
        team_pred[1] = np.array(team_pred[1])/np.array(team_pred[1]).sum(axis=0,keepdims=1)
        return result_proba, [np.array(team_pred[0]), np.array(team_pred[1])], match
    
    
    def get_match_result(foot_model, team1, team2, elimination=False, max_draw=50, max_goals=10):

        proba,score_proba, match = get_proba_match(foot_model, team1, team2, max_goals)
        
       
        results = pd.Series([np.random.choice([team1, 'draw', team2], p=proba) for i in range(0,max_draw)]).value_counts()
        result = results.index[0] if not elimination or (elimination and results.index[0] != 'draw') else results.index[1]
        
        
        if (result != 'draw'): 
            i_win, i_loose = (0,1) if result == team1 else (1,0)
            score_proba[i_win] = score_proba[i_win][1:]/score_proba[i_win][1:].sum(axis=0,keepdims=1)
            winner_score = pd.Series([np.random.choice(range(1, max_goals+1), p=score_proba[i_win]) for i in range(0,max_draw)]).value_counts().index[0]
            
            score_proba[i_loose] = score_proba[i_loose][:winner_score]/score_proba[i_loose][:winner_score].sum(axis=0,keepdims=1)
            looser_score = pd.Series([np.random.choice(range(0, winner_score), p=score_proba[i_loose]) for i in range(0,max_draw)]).value_counts().index[0]
            
            
            score = [winner_score, looser_score]
            
        
        
        else:
            score = np.repeat(pd.Series([np.random.choice(range(0, max_goals+1), p=score_proba[0]) for i in range(0,max_draw)]).value_counts().index[0],2)
        looser = team2 if result == team1 else team1 if result != 'draw' else 'draw'
        
        
        t1_wins = round(np.sum(np.tril(match, -1)),2)
        draw = round(np.sum(np.diag(match)),2)
        t2_wins = round(np.sum(np.triu(match, 1)),2)
            
        porcent = (f'Chances de vitória {team1}: {t1_wins}%, chances de empate: {draw}%, chances de vitória {team2}: {t2_wins}%')
            
        
        return result, looser, score, porcent
    

    

    prediction= get_match_result(model, time1_eliminatoria, time2_eliminatoria, max_goals=10, elimination=True)
    return prediction


#Funções para simular os jogos

groupA = ['Qatar', 'Senegal', 'Netherlands', 'Ecuador']
groupB = ['England', 'Iran','United States', 'Wales']
groupC = ['Argentina', 'Saudi Arabia', 'Mexico', 'Poland']
groupD = ['France', 'Australia', 'Denmark', 'Tunisia']
groupE = ['Spain', 'Costa Rica','Germany', 'Japan']
groupF = ['Belgium', 'Canada', 'Morocco', 'Croatia']
groupG = ['Brazil', 'Serbia', 'Switzerland', 'Cameroon']
groupH = ['Portugal', 'Ghana','Uruguay', 'Korea Republic']
groups = [groupA, groupB, groupC, groupD, groupE, groupF, groupG, groupH]

def get_proba_match(foot_model, team1, team2, max_goals=10):
        
        t1_goals_avg = foot_model.predict(pd.DataFrame(data={'team': team1, 'opponent': team2}, index=[1])).values[0]
        t2_goals_avg = foot_model.predict(pd.DataFrame(data={'team': team2, 'opponent': team1}, index=[1])).values[0]
        
      
        team_pred = [[poisson.pmf(i, team_avg) for i in range(0, max_goals+1)] for team_avg in [t1_goals_avg, t2_goals_avg]]
        
        #todos as probabilidades de resultados possiveis
        match = np.outer(np.array(team_pred[0]), np.array(team_pred[1]))
        
        
        t1_wins = np.sum(np.tril(match, -1))
        draw = np.sum(np.diag(match))
        t2_wins = np.sum(np.triu(match, 1))
        result_proba = [t1_wins, draw, t2_wins]
        
        
        result_proba =  np.array(result_proba)/ np.array(result_proba).sum(axis=0,keepdims=1)
        team_pred[0] = np.array(team_pred[0])/np.array(team_pred[0]).sum(axis=0,keepdims=1)
        team_pred[1] = np.array(team_pred[1])/np.array(team_pred[1]).sum(axis=0,keepdims=1)
        return result_proba, [np.array(team_pred[0]), np.array(team_pred[1])], match

def get_match_result(foot_model, team1, team2, elimination=False, max_draw=50, max_goals=10):
       
        proba,score_proba, match = get_proba_match(foot_model, team1, team2, max_goals)
        
       
        results = pd.Series([np.random.choice([team1, 'draw', team2], p=proba) for i in range(0,max_draw)]).value_counts()
        result = results.index[0] if not elimination or (elimination and results.index[0] != 'draw') else results.index[1]
        
     
        if (result != 'draw'): 
            i_win, i_loose = (0,1) if result == team1 else (1,0)
            score_proba[i_win] = score_proba[i_win][1:]/score_proba[i_win][1:].sum(axis=0,keepdims=1)
            winner_score = pd.Series([np.random.choice(range(1, max_goals+1), p=score_proba[i_win]) for i in range(0,max_draw)]).value_counts().index[0]
            
            score_proba[i_loose] = score_proba[i_loose][:winner_score]/score_proba[i_loose][:winner_score].sum(axis=0,keepdims=1)
            looser_score = pd.Series([np.random.choice(range(0, winner_score), p=score_proba[i_loose]) for i in range(0,max_draw)]).value_counts().index[0]
            
            
            score = [winner_score, looser_score]
            
     
        else:
            score = np.repeat(pd.Series([np.random.choice(range(0, max_goals+1), p=score_proba[0]) for i in range(0,max_draw)]).value_counts().index[0],2)
        looser = team2 if result == team1 else team1 if result != 'draw' else 'draw'
        
        
        t1_wins = round(np.sum(np.tril(match, -1)),2)
        draw = round(np.sum(np.diag(match)),2)
        t2_wins = round(np.sum(np.triu(match, 1)),2)
            
        porcent = (f'Chances de vitória {team1}: {t1_wins}%, chances de empate: {draw}%, chances de vitória {team2}: {t2_wins}%')
            
        
        return result, looser, score, porcent
        
#Jogos da fase de grupos

def get_group_result(foot_model, group):
    ranking = pd.DataFrame({'points':[0,0,0,0], 'diff':[0,0,0,0], 'goals':[0,0,0,0]}, index=group)
    for team1, team2 in itertools.combinations(group, 2):
        result, looser, score, porcent = get_match_result(foot_model, team1, team2)
        if result == 'draw':
            ranking.loc[[team1, team2], 'points'] += 1
            ranking.loc[[team1, team2], 'goals'] += score[0]
        else:
            ranking.loc[result, 'points'] += 3
            ranking.loc[result, 'goals'] += score[0]
            ranking.loc[looser, 'goals'] += score[1]
            ranking.loc[result, 'diff'] += score[0]-score[1]
            ranking.loc[looser, 'diff'] -= score[0]-score[1]
            
    return ranking.sort_values(by=['points','diff','goals'], ascending=False)

groups_ranking = []
for group in groups:
    groups_ranking.append(get_group_result(model, group))


#Jogos da fase mata-mata

def get_final_result(foot_model, groups_result):
    round_of_16 = []
    quarter_finals = []
    semi_finals = []
    
    #Simulando rodada 16
    for i in range(0, 8, 2):
        round_of_16.append(get_match_result(foot_model, groups_result[i].index[0], groups_result[i+1].index[1], elimination=True))
        round_of_16.append(get_match_result(foot_model, groups_result[i].index[1], groups_result[i+1].index[0], elimination=True))
    
    #Simulando quartas de final
    quarter_finals.append(get_match_result(foot_model, round_of_16[0][0], round_of_16[2][0], elimination=True))
    quarter_finals.append(get_match_result(foot_model, round_of_16[1][0], round_of_16[3][0], elimination=True))
    quarter_finals.append(get_match_result(foot_model, round_of_16[4][0], round_of_16[6][0], elimination=True))
    quarter_finals.append(get_match_result(foot_model, round_of_16[5][0], round_of_16[7][0], elimination=True))
    
    #Simulando semi-final
    semi_finals.append(get_match_result(foot_model, quarter_finals[0][0], quarter_finals[2][0], elimination=True))
    semi_finals.append(get_match_result(foot_model, quarter_finals[1][0], quarter_finals[3][0], elimination=True))
    
    #Simulando disputa de terceiro lugar
    little_final = get_match_result(foot_model, semi_finals[0][1], semi_finals[1][1], elimination=True)
    
    #Simulando final
    final = get_match_result(foot_model, semi_finals[0][0], semi_finals[1][0], elimination=True)
    
    return round_of_16, quarter_finals, semi_finals, little_final, final

round_of_16, quarter_finals, semi_finals, little_final, final = get_final_result(model, groups_ranking)

fig = plt.figure(figsize = (40,45))
img = mpimg.imread('tabela.png')
plt.imshow(img)
plt.axis('off')

def text_match(x, y, match, final=False):
    col_win, col_loose = ('green', 'red') if (not final) else ('gold', 'silver')
    plt.text(x, y, match[0], fontsize=23, color=col_win, weight='bold')
    plt.text(x+120, y+1, match[2][0], fontsize=38, color=col_win, weight='bold')
    plt.text(x, y+50, match[1], fontsize=23, color=col_loose, weight='bold')
    plt.text(x+120, y+51, match[2][1], fontsize=38, color=col_loose, weight='bold')


round_of_16_xy = [(40,110),(898,110),(40,280),(898,280),(40,430),(898,430),(40,600),(898,600)]
quarter_finals_xy = [(212,198),(726,198),(212,518),(726,518)]
semi_finals_xy = [(378,365),(560,365)]
x_little_final, y_little_final = 560, 576
x_final, y_final = 469, 157


for (x, y), match in zip(round_of_16_xy, round_of_16):
     text_match(x, y, match)
for (x, y), match in zip(quarter_finals_xy, quarter_finals):
        text_match(x, y, match)
for (x, y), match in zip(semi_finals_xy, semi_finals):
        text_match(x, y, match)
    
text_match(x_little_final, y_little_final, little_final)
text_match(x_final, y_final, final, final=True)

   