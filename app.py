import streamlit as st
import pickle as pkl
import streamlit.components.v1 as components 
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
        
        # Do the product of the 2 vectors to get the matrix of the match
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
        return result_proba, [np.array(team_pred[0]), np.array(team_pred[1])]
    
    def get_match_result(foot_model, team1, team2, elimination=False, max_draw=50, max_goals=10):
        # Get the proba
        proba, score_proba = get_proba_match(foot_model, team1, team2, max_goals)
        
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
        return result, looser, score
    

    

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
        
        # Do the product of the 2 vectors to get the matrix of the match
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
        return result_proba, [np.array(team_pred[0]), np.array(team_pred[1])]
    
    def get_match_result(foot_model, team1, team2, elimination=False, max_draw=50, max_goals=10):
        # Get the proba
        proba, score_proba = get_proba_match(foot_model, team1, team2, max_goals)
        
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
        return result, looser, score
    

    

    prediction= get_match_result(model, time1_eliminatoria, time2_eliminatoria, max_goals=10, elimination=True)
    #print(prediction)
    return prediction


def main():
    st.title("Aplicação de Machine Learning para prever possíveis resultados de jogos da copa 2022")
     
    html_temp2 = """
		<div style="background-color:royalblue;padding:-120px;border-radius:-120px ; height:200px">
		<h2 style="color:white;text-align:center;">Copa do mundo 2022 </h2>
        <h1 style="color:white;text-align:center;">Escolha dois times e veja o possível resultado para um jogo da copa do mundo de 2022</h1>
		</div>
		"""
    # a simple html code for heading which is in blue color and we can even use "st.write()" also ut for back ground color i used this HTML ..... 
    #  to render this we use ...
    components.html(html_temp2)
    # components.html() will render the render the 

    #components.html("""
                #<img src="http://sportinsider.com.br/wp-content/uploads/2022/05/copa-do-mundo-do-catar-2022-1.jpg" width="1500px" height="1050px">
                
                #""")


    # this is to insert the image the in the wed app simple <imag/> tag in HTML
    
    #now lets get the test input from the user by wed app 
    # for this we can use "st.text_input()" which allows use to get the input from the user 
    
  
    st.subheader('Prever partidas da fase de grupo')

    grupo_time1 = st.selectbox('Time 1 (Fase de grupo)', ['Escolha um time', 'Qatar', 'Senegal', 'Netherlands', 'Ecuador', 'England', 'Iran',
       'USA', 'Wales', 'Argentina', 'Saudi Arabia', 'Mexico', 'Poland',
       'France', 'Australia', 'Denmark', 'Tunisia', 'Spain', 'Costa Rica',
       'Germany', 'Japan', 'Belgium', 'Canada', 'Morocco', 'Croatia',
       'Brazil', 'Serbia', 'Switzerland', 'Cameroon', 'Portugal', 'Ghana',
       'Uruguay', 'Korea Republic'])
    
    grupo_time2 = st.selectbox('Time 2 (Fase de grupo)', ['Escolha um time', 'Qatar', 'Senegal', 'Netherlands', 'Ecuador', 'England', 'Iran',
       'USA', 'Wales', 'Argentina', 'Saudi Arabia', 'Mexico', 'Poland',
       'France', 'Australia', 'Denmark', 'Tunisia', 'Spain', 'Costa Rica',
       'Germany', 'Japan', 'Belgium', 'Canada', 'Morocco', 'Croatia',
       'Brazil', 'Serbia', 'Switzerland', 'Cameroon', 'Portugal', 'Ghana',
       'Uruguay', 'Korea Republic'])

    result_grupo =""


    # done we got all the user inputs to predict and we need a button like a predict button we do that by "st.button()"
    # after hitting the button the prediction process will go on and then we print the success message by "st.success()"

    if st.button("Prever resultado de partida da fase de grupos"):
        if ((grupo_time1 or grupo_time2) != 'Escolha um time'):
            result_grupo = predict_eliminatoria(grupo_time1, grupo_time2)
            st.success('Resultado da partida: {}'.format(result_grupo))
         
       
        else:
            st.warning('Você precisa escolher um time')
        





    #previsão para partidas de eliminatórias 

    
    st.subheader('Prever partidas da fase eliminatória')

    eliminatoria_time1 = st.selectbox('Time 1 (Fase eliminatória)', ['Escolha um time','Qatar', 'Senegal', 'Netherlands', 'Ecuador', 'England', 'Iran',
       'USA', 'Wales', 'Argentina', 'Saudi Arabia', 'Mexico', 'Poland',
       'France', 'Australia', 'Denmark', 'Tunisia', 'Spain', 'Costa Rica',
       'Germany', 'Japan', 'Belgium', 'Canada', 'Morocco', 'Croatia',
       'Brazil', 'Serbia', 'Switzerland', 'Cameroon', 'Portugal', 'Ghana',
       'Uruguay', 'Korea Republic'], 0)
    
    eliminatoria_time2= st.selectbox('Time 2 (Fase eliminatória)', ['Escolha um time', 'Qatar', 'Senegal', 'Netherlands', 'Ecuador', 'England', 'Iran',
       'USA', 'Wales', 'Argentina', 'Saudi Arabia', 'Mexico', 'Poland',
       'France', 'Australia', 'Denmark', 'Tunisia', 'Spain', 'Costa Rica',
       'Germany', 'Japan', 'Belgium', 'Canada', 'Morocco', 'Croatia',
       'Brazil', 'Serbia', 'Switzerland', 'Cameroon', 'Portugal', 'Ghana',
       'Uruguay', 'Korea Republic'],0)

    result_eliminatoria=""


    # done we got all the user inputs to predict and we need a button like a predict button we do that by "st.button()"
    # after hitting the button the prediction process will go on and then we print the success message by "st.success()"

    if st.button("Prever resultado de partida eliminatória"):
        if ((eliminatoria_time1 or eliminatoria_time2) != 'Escolha um time'):
            result_eliminatoria = predict_eliminatoria(eliminatoria_time1, eliminatoria_time2)
            st.success('Resultado da partida: {}'.format(result_eliminatoria))
       
            
        else:
            st.warning('Você precisa escolher um time')
            
    # one more button saying About ...
    if st.button("About"):
        st.text("Lets LEarn")
        st.text("Built with Streamlit")
    
    st.sidebar.info("Essa aplicação de machine learning está sendo alimentada por um modelo que é capaz de prever um possível resultado para jogos da copa do mundo")
    st.sidebar.info("O modelo foi construído a partir de dados da FIFA e resultados de copas anteriores")
    st.sidebar.info("O resultado para uma partida pode mudar dependendo se for uma partida da fase de grupos ou fase eliminatória, por isso existem duas opções")

    
if __name__=='__main__':
    main()




