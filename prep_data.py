import streamlit as st
import pickle as pkl
import pandas as pd
import numpy as np
from scipy.stats import poisson


#Load the saved model
model=pkl.load(open("poisson_model.pkl","rb"))

def predict_grupo(time1, time2):

    def get_proba_match(foot_model, team1, team2, max_goals=10):
        # Get the average goal for each team
        t1_goals_avg = foot_model.predict(pd.DataFrame(data={'team': team1, 'opponent': team2}, index=[1])).values[0]
        t2_goals_avg = foot_model.predict(pd.DataFrame(data={'team': team2, 'opponent': team1}, index=[1])).values[0]
        
        # Get probability of all possible score for each team
        team_pred = [[poisson.pmf(i, team_avg) for i in range(0, max_goals+1)] for team_avg in [t1_goals_avg, t2_goals_avg]]
        
        #todos as probabilidades de resultados possiveis
        match = np.outer(np.array(team_pred[0]), np.array(team_pred[1]))
        
        # Get the proba for each possible outcome
        t1_wins = np.sum(np.tril(match, -1))
        draw = np.sum(np.diag(match))
        t2_wins = np.sum(np.triu(match, 1))
        result_proba = [t1_wins, draw, t2_wins]
        
        # Adjust the proba to sum to one
        result_proba =  np.array(result_proba)/ np.array(result_proba).sum(axis=0,keepdims=1)
        team_pred[0] = np.array(team_pred[0])/np.array(team_pred[0]).sum(axis=0,keepdims=1)
        team_pred[1] = np.array(team_pred[1])/np.array(team_pred[1]).sum(axis=0,keepdims=1)
        return result_proba, [np.array(team_pred[0]), np.array(team_pred[1])], match
    
    def get_match_result(foot_model, team1, team2, elimination=False, max_draw=50, max_goals=10):
        # Get the proba
        proba,score_proba, match = get_proba_match(foot_model, team1, team2, max_goals)
        
        # Get the result, if it's an elimination game we have to be sure the result is not draw
        results = pd.Series([np.random.choice([team1, 'draw', team2], p=proba) for i in range(0,max_draw)]).value_counts()
        result = results.index[0] if not elimination or (elimination and results.index[0] != 'draw') else results.index[1]
        
        # If the result is not a draw game then we calculate the score of the winner from 1 to the max_goals 
        # and the score of the looser from 0 to the score of the winner
        if (result != 'draw'): 
            i_win, i_loose = (0,1) if result == team1 else (1,0)
            score_proba[i_win] = score_proba[i_win][1:]/score_proba[i_win][1:].sum(axis=0,keepdims=1)
            winner_score = pd.Series([np.random.choice(range(1, max_goals+1), p=score_proba[i_win]) for i in range(0,max_draw)]).value_counts().index[0]
            
            score_proba[i_loose] = score_proba[i_loose][:winner_score]/score_proba[i_loose][:winner_score].sum(axis=0,keepdims=1)
            looser_score = pd.Series([np.random.choice(range(0, winner_score), p=score_proba[i_loose]) for i in range(0,max_draw)]).value_counts().index[0]
            
            
            score = [winner_score, looser_score]
            
        
        # If it's a draw then we calculate a score and repeat it twice
        else:
            score = np.repeat(pd.Series([np.random.choice(range(0, max_goals+1), p=score_proba[0]) for i in range(0,max_draw)]).value_counts().index[0],2)
        looser = team2 if result == team1 else team1 if result != 'draw' else 'draw'
        
        
        t1_wins = round(np.sum(np.tril(match, -1)),2)
        draw = round(np.sum(np.diag(match)),2)
        t2_wins = round(np.sum(np.triu(match, 1)),2)
            
        porcent = (f'Chances de vitória {team1}: {t1_wins}%, chances de empate: {draw}%, chances de vitória {team2}: {t2_wins}%')
            
        
        return result, looser, score, porcent
        

        

    prediction= get_match_result(model, time1, time2, max_goals=10)
    #print(prediction)
    return prediction




def predict_eliminatoria(time1_eliminatoria, time2_eliminatoria):

    def get_proba_match(foot_model, team1, team2, max_goals=10):
        # Get the average goal for each team
        t1_goals_avg = foot_model.predict(pd.DataFrame(data={'team': team1, 'opponent': team2}, index=[1])).values[0]
        t2_goals_avg = foot_model.predict(pd.DataFrame(data={'team': team2, 'opponent': team1}, index=[1])).values[0]
        
        # Get probability of all possible score for each team
        team_pred = [[poisson.pmf(i, team_avg) for i in range(0, max_goals+1)] for team_avg in [t1_goals_avg, t2_goals_avg]]
        
        #todos as probabilidades de resultados possiveis
        match = np.outer(np.array(team_pred[0]), np.array(team_pred[1]))
        
        # Get the proba for each possible outcome
        t1_wins = np.sum(np.tril(match, -1))
        draw = np.sum(np.diag(match))
        t2_wins = np.sum(np.triu(match, 1))
        result_proba = [t1_wins, draw, t2_wins]
        
        # Adjust the proba to sum to one
        result_proba =  np.array(result_proba)/ np.array(result_proba).sum(axis=0,keepdims=1)
        team_pred[0] = np.array(team_pred[0])/np.array(team_pred[0]).sum(axis=0,keepdims=1)
        team_pred[1] = np.array(team_pred[1])/np.array(team_pred[1]).sum(axis=0,keepdims=1)
        return result_proba, [np.array(team_pred[0]), np.array(team_pred[1])], match
    
    
    def get_match_result(foot_model, team1, team2, elimination=False, max_draw=50, max_goals=10):
        # Get the proba
        proba,score_proba, match = get_proba_match(foot_model, team1, team2, max_goals)
        
        # Get the result, if it's an elimination game we have to be sure the result is not draw
        results = pd.Series([np.random.choice([team1, 'draw', team2], p=proba) for i in range(0,max_draw)]).value_counts()
        result = results.index[0] if not elimination or (elimination and results.index[0] != 'draw') else results.index[1]
        
        # If the result is not a draw game then we calculate the score of the winner from 1 to the max_goals 
        # and the score of the looser from 0 to the score of the winner
        if (result != 'draw'): 
            i_win, i_loose = (0,1) if result == team1 else (1,0)
            score_proba[i_win] = score_proba[i_win][1:]/score_proba[i_win][1:].sum(axis=0,keepdims=1)
            winner_score = pd.Series([np.random.choice(range(1, max_goals+1), p=score_proba[i_win]) for i in range(0,max_draw)]).value_counts().index[0]
            
            score_proba[i_loose] = score_proba[i_loose][:winner_score]/score_proba[i_loose][:winner_score].sum(axis=0,keepdims=1)
            looser_score = pd.Series([np.random.choice(range(0, winner_score), p=score_proba[i_loose]) for i in range(0,max_draw)]).value_counts().index[0]
            
            
            score = [winner_score, looser_score]
            
        
        # If it's a draw then we calculate a score and repeat it twice
        else:
            score = np.repeat(pd.Series([np.random.choice(range(0, max_goals+1), p=score_proba[0]) for i in range(0,max_draw)]).value_counts().index[0],2)
        looser = team2 if result == team1 else team1 if result != 'draw' else 'draw'
        
        
        t1_wins = round(np.sum(np.tril(match, -1)),2)
        draw = round(np.sum(np.diag(match)),2)
        t2_wins = round(np.sum(np.triu(match, 1)),2)
            
        porcent = (f'Chances de vitória {team1}: {t1_wins}%, chances de empate: {draw}%, chances de vitória {team2}: {t2_wins}%')
            
        
        return result, looser, score, porcent
    

    

    prediction= get_match_result(model, time1_eliminatoria, time2_eliminatoria, max_goals=10, elimination=True)
    #print(prediction)
    return prediction

