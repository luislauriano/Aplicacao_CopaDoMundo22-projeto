import streamlit as st

import streamlit.components.v1 as components 

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import prep_data as prep



def main():
    st.title("Aplicação de Machine Learning para prever possíveis resultados de jogos da copa 2022")
     
    html_temp2 = """
		<div style="background-color:white;padding:-120px;border-radius:-120px ; height:200px">
		<h2 style="color:#FF0031;text-align:center;">Copa do mundo 2022 </h2>
        <h1 style="color:#FF0031;text-align:center;">Escolha dois times e veja o possível resultado para um jogo da copa do mundo de 2022</h1>
		</div>
		"""
    
    components.html(html_temp2)
    
    
  
    st.subheader('Prever partidas da fase de grupo')

    grupo_time1 = st.selectbox('Time 1 (Fase de grupo)', ['Escolha um time', 'Qatar', 'Senegal', 'Netherlands', 'Ecuador', 'England', 'Iran',
       'United States', 'Wales', 'Argentina', 'Saudi Arabia', 'Mexico', 'Poland',
       'France', 'Australia', 'Denmark', 'Tunisia', 'Spain', 'Costa Rica',
       'Germany', 'Japan', 'Belgium', 'Canada', 'Morocco', 'Croatia',
       'Brazil', 'Serbia', 'Switzerland', 'Cameroon', 'Portugal', 'Ghana',
       'Uruguay', 'Korea Republic'])
    
    grupo_time2 = st.selectbox('Time 2 (Fase de grupo)', ['Escolha um time', 'Qatar', 'Senegal', 'Netherlands', 'Ecuador', 'England', 'Iran',
       'United States', 'Wales', 'Argentina', 'Saudi Arabia', 'Mexico', 'Poland',
       'France', 'Australia', 'Denmark', 'Tunisia', 'Spain', 'Costa Rica',
       'Germany', 'Japan', 'Belgium', 'Canada', 'Morocco', 'Croatia',
       'Brazil', 'Serbia', 'Switzerland', 'Cameroon', 'Portugal', 'Ghana',
       'Uruguay', 'Korea Republic'])

    result_grupo =""


   

    if st.button("Prever resultado de partida da fase de grupos"):
        if ((grupo_time1 or grupo_time2) != 'Escolha um time'):
            result_grupo = (prep.predict_grupo(grupo_time1, grupo_time2)[0:3])
            chances_vitor_empate_derrota = str(prep.predict_grupo(grupo_time1, grupo_time2)[3:])
            st.success('Possível resultado da partida: {}'.format(result_grupo))
            st.success('Chances de vitória, empate e derrota: {}'.format(chances_vitor_empate_derrota))
         
       
        else:
            st.warning('Você precisa escolher um time')
        



    #previsão para partidas de eliminatórias 
    st.subheader('Prever partidas da fase eliminatória')

    eliminatoria_time1 = st.selectbox('Time 1 (Fase eliminatória)', ['Escolha um time','Qatar', 'Senegal', 'Netherlands', 'Ecuador', 'England', 'Iran',
       'United States', 'Wales', 'Argentina', 'Saudi Arabia', 'Mexico', 'Poland',
       'France', 'Australia', 'Denmark', 'Tunisia', 'Spain', 'Costa Rica',
       'Germany', 'Japan', 'Belgium', 'Canada', 'Morocco', 'Croatia',
       'Brazil', 'Serbia', 'Switzerland', 'Cameroon', 'Portugal', 'Ghana',
       'Uruguay', 'South Korea'], 0)
    
    eliminatoria_time2= st.selectbox('Time 2 (Fase eliminatória)', ['Escolha um time', 'Qatar', 'Senegal', 'Netherlands', 'Ecuador', 'England', 'Iran',
       'United States', 'Wales', 'Argentina', 'Saudi Arabia', 'Mexico', 'Poland',
       'France', 'Australia', 'Denmark', 'Tunisia', 'Spain', 'Costa Rica',
       'Germany', 'Japan', 'Belgium', 'Canada', 'Morocco', 'Croatia',
       'Brazil', 'Serbia', 'Switzerland', 'Cameroon', 'Portugal', 'Ghana',
       'Uruguay', 'South Korea'],0)

    result_eliminatoria=""



    if st.button("Prever resultado de partida eliminatória"):
        if ((eliminatoria_time1 or eliminatoria_time2) != 'Escolha um time'):
            result_eliminatoria = str(prep.predict_eliminatoria(eliminatoria_time1, eliminatoria_time2)[0:3])
            chances_vitoria_empate_derrota = str(prep.predict_eliminatoria(eliminatoria_time1, eliminatoria_time2)[3:])
            st.success('Possível resultado da partida: {}'.format(result_eliminatoria))
            st.success('Chances de vitória, empate e derrota: {}'.format(chances_vitoria_empate_derrota))
       
            
        else:
            st.warning('Você precisa escolher um time')


    #Simulação dos jogos fase de grupos
    st.subheader('Simulação da fase de grupos e fase mata-mata')
    st.write('Ao selecionar um simulador você irá obter uma simulação de como possivelmente irá terminar a fase de grupos e a fase mata-mata com base nos resultados possíveis que o modelo preveu.')
    st.write('O simulador da fase mata-mata acaba demorando um pouco para retornar o resultado por conta que a imagem leva um tempo para ser carregada. A imagem pode ser ampliada clicando no ícone de expansão ao lado')
    col1, col2, col3 = st.columns(3)
    if col2.button('Simular resultado da fase de grupos'):
       for group_rank in prep.groups_ranking:
           st.write(group_rank)
    
    #Simulação dos jogos mata-mata
 

   


    
    

    col4, col5, col6 = st.columns(3)
    if col5.button('Simular resultado da fase mata-mata'):
        prep.imagem_resultado()
         
       
    
    st.sidebar.info('Autor: Luis Vinicius (el tito) - @l.uisvinicius')
    st.sidebar.info("Essa aplicação de machine learning foi construída a partir de um modelo linear generalizado com base na regressão de Poisson, o objetivo do modelo é prever um possível resultado para jogos da copa do mundo, utilizando como target a quantidade de gols em uma partida")
    st.sidebar.info("O modelo foi construído a partir de dados do ranking de seleções e partidas internacionais da FIFA")
    st.sidebar.info("O resultado da partida é feito a partir da escolha do resultado mais provável, diante da probabilidade de cada resultado possível (vitória, empate e derrota) que foi calculado em uma função com base em todas as probabilidades de resultados possíveis. Por esse motivo, o resultado da partida pode acabar sendo diferente se testado outra vez. Para ficar mais claro, podemos imaginar a cena do doutor estranho no filme guerra infinita onde ele encontra um único resultado positivo para eles vencerem a guerra diante de todos os resultados possiveis finais que a guerra contra thanos poderia ter.")
    st.sidebar.info("O resultado de gols de uma partida pode acabar se repetindo em muitos casos o placar de 1x0, talvez por o modelo não ter uma precisão para quantificar tão bem saldos maiores, mas o interessante é entender e levar em consideração quem o modelo está prevendo como possível vencedor da partida")
    st.sidebar.info("Para uma partida eliminatória o resultado de um jogo não pode ser empate, então existe uma opção apenas para ser calculado o possível resultado de partidas eliminatórias")
    
    
if __name__=='__main__':
    main()




