import streamlit as st

import streamlit.components.v1 as components 


import prep_data as prep



def main():
    st.title("Aplicação de Machine Learning para prever possíveis resultados de jogos da copa 2022")
     
    html_temp2 = """
		<div style="background-color:#FF0031;padding:-120px;border-radius:-120px ; height:200px">
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


    # done we got all the user inputs to predict and we need a button like a predict button we do that by "st.button()"
    # after hitting the button the prediction process will go on and then we print the success message by "st.success()"

    if st.button("Prever resultado de partida da fase de grupos"):
        if ((grupo_time1 or grupo_time2) != 'Escolha um time'):
            result_grupo = prep.predict_grupo(grupo_time1, grupo_time2)
            st.success('Possível esultado da partida: {}'.format(result_grupo))
         
       
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


    # done we got all the user inputs to predict and we need a button like a predict button we do that by "st.button()"
    # after hitting the button the prediction process will go on and then we print the success message by "st.success()"

    if st.button("Prever resultado de partida eliminatória"):
        if ((eliminatoria_time1 or eliminatoria_time2) != 'Escolha um time'):
            result_eliminatoria = str(prep.predict_eliminatoria(eliminatoria_time1, eliminatoria_time2))
            st.success('Possível resultado da partida: {}'.format(result_eliminatoria))
       
            
        else:
            st.warning('Você precisa escolher um time')
            
    # one more button saying About ...
    if st.button("About"):
        st.text("Lets LEarn")
        st.text("Built with Streamlit")
    
    st.sidebar.info("Essa aplicação de machine learning está sendo alimentada por um modelo que é capaz de prever um possível resultado para jogos da copa do mundo")
    st.sidebar.info("O modelo foi construído a partir de dados da FIFA e resultados de copas anteriores")
    st.sidebar.info("Os resultados de uma mesma simulação podem acabar sendo diferentes, tendo em vista que o resultado é um cálculo de probabilidade ")
    st.sidebar.info("O resultado para uma partida pode mudar dependendo se for uma partida da fase de grupos ou fase eliminatória, por isso existem duas opções")

    
if __name__=='__main__':
    main()




