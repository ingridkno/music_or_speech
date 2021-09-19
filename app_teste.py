import streamlit as st
import pandas as pd
#import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import itertools

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Use the full page instead of a narrow central column
st.set_page_config(layout="wide")

def rescale(max_new, min_new, max_old, min_old, value):
    new_value = (((max_new - min_new)*(value-max_old))/(max_old-min_old))+max_new
    return new_value


def ml_model(x, y, SEED, scaler_on, stratify_test):
    np.random.seed(SEED)

    if stratify_test:
        # st.text('STRATIFY')
        treino_x, teste_x, treino_y, teste_y = train_test_split(x, y,
                                                                test_size=0.25,
                                                                stratify=y)
    else:
        treino_x, teste_x, treino_y, teste_y = train_test_split(x, y,
                                                                test_size=0.25)
    #
    # st.text(treino_x.shape)
    # st.text(teste_x.shape)
    #
    # st.text("\nTreinaremos com %d elementos e testaremos com %d elementos"
    #         % (len(treino_x), len(teste_x)))

    x_train_size = len(treino_x)
    x_test_size = len(teste_x)

    if scaler_on:
        # st.text('SCALER')
        scaler = StandardScaler()
        scaler.fit(treino_x)
        treino_x = scaler.transform(treino_x)
        teste_x = scaler.transform(teste_x)


    else:
        pass

    model = SVC()
    model.fit(treino_x, treino_y)
    previsoes = model.predict(teste_x)

    acuracia = accuracy_score(teste_y, previsoes) * 100
    # st.text("A acurácia foi %.2f%%" % (acuracia))

    previsoes_de_base = np.ones(len(teste_y))
    acuracia_baseline = accuracy_score(teste_y, previsoes_de_base) * 100
    #st.text("A acurácia do algoritmo baseline foi %.2f%%" % (acuracia_baseline))

    return acuracia, acuracia_baseline, x_train_size, x_test_size, teste_x, teste_y, model

def countour_plot(teste_x, var_1, var_2, model, scaler=True):
    if scaler:
        #st.text(scaler_on)
        data_x = teste_x[:, 0]
        data_y = teste_x[:, 1]
        x_min = data_x.min()
        # st.text(x_min)
        x_max = data_x.max()
        # st.text(x_max)
        y_min = data_y.min()
        # st.text(y_min)
        y_max = data_y.max()
        # st.text(y_max)
    else:
        
        data_x = teste_x[var_1]
        data_y = teste_x[var_2]
        x_min = teste_x[var_1].min()
        x_max = teste_x[var_1].max()

        y_min = teste_x[var_2].min()
        y_max = teste_x[var_2].max()

    pixels = 100

    eixo_x = np.arange(x_min, x_max, (x_max - x_min) / pixels)
    eixo_y = np.arange(y_min, y_max, (y_max - y_min) / pixels)

    xx, yy = np.meshgrid(eixo_x, eixo_y)
    pontos = np.c_[xx.ravel(), yy.ravel()]
    # pontos

    Z = model.predict(pontos)
    Z = Z.reshape(xx.shape)
    # Z

    # st.text(str(x_min)+' '+ str(x_max)+ ' '+ str(y_min)+ ' ' +str(y_max))
    return (xx, yy, Z, data_x, data_y)

df = pd.read_csv('musan_features_mod.csv')

# df[df.columns[2:]].to_csv('musan_features_mod.csv')

# st.dataframe(df.head())
# st.write(list(df.columns)[8:])

features=list(df.columns)[7:-1]

# st.write(features)

st.sidebar.write('Features')
feature_1 = st.sidebar.selectbox('1st feature',features, 0)
feature_2 = st.sidebar.selectbox('2nd feature',features, 1)

st.sidebar.write('Parameters')
stratify_test = st.sidebar.checkbox('Stratify',value=True)
scaler_on = st.sidebar.checkbox('StandardScaler',value=True)
SEED = st.sidebar.slider('SEED number', 0, 100, 20)


fig = px.scatter(data_frame=df, x=feature_1, y=feature_2, color="class", width=450, height=400, title="Feature 1 versus Feature 2")

df_=df[[feature_1, feature_2, 'music']].copy().dropna()
x = df_[[feature_1, feature_2]]
y = df_["music"]

df_counts=pd.DataFrame(df['class'].value_counts())
df_counts['class %'] = 100*df_counts['class']/(df_counts['class'].sum()) 
#st.dataframe(df_counts)

fig_2 = px.bar(df_counts[['class']], orientation='h', title="Audio files for each class", width=300, height=400)
fig_2.update_layout(showlegend=False)#, xaxis_showticklabels=False)

# acuracia, x_train_size, x_test_size, teste_x = ml_model(x, y, SEED, scaler_on, stratify_test)

#plt.scatter(df[feature_1], df[feature_2], c = df['music'], s=1)
video_file = open('teste_video.mp4', 'rb')
video_bytes = video_file.read()






left_column, middle_column = st.columns((2,1))

with left_column:
    st.title('Music :musical_note: or Speech :speech_balloon:?')
with middle_column:
    st.video(video_bytes)
st.subheader('\nTwo dimensional analysis in machine learning model applied to audio')
st.write('\nIn this app, we will be able to run through the steps to build a simple machine learning model applied to audio.') 


st.subheader('CHALLENGE')
st.write('\nImagine you work from a famous and solid radio station. There are old programs that are being digitized from cassettes and your first mission is to organize the audio files. In this organization, you want to classify what is music and what is broadcast. You have already a classified dataset and you want to use the help of artificial intelligence to optimize your work.') 
st.write('What would be your first step?')



st.subheader('1- Analysing dataset')

my_expander = st.expander(label='Data exploring and features')
with my_expander:
    'At the left, there are lists in order to choose two features. This features are responsible for building the model and '
    #clicked = st.button('Click me!')

    left_column, right_column = st.columns((3,5))

    with left_column:
        st.plotly_chart(fig_2)
        
    with right_column:        
        st.plotly_chart(fig)
        
        
my_expander = st.expander(label='Parameters')
with my_expander:
    'At the left, there are lists in order to choose two features. This features are responsible for building the model and '
    #clicked = st.button('Click me!')

    left_column, right_column, _ = st.columns(3)

    with left_column:
        st.subheader('Stratify')
        if stratify_test:
            st.write(":heavy_check_mark:")
            st.write('When it is on, test data is split in a stratified fashion. That means that train and test data will have approximately same percentage to each class.') 
        else:
            st.write(":o:")
            st.write('When it is off, quantity of each class in test data will be chosen randomly.')
    
    with right_column:
        st.subheader('Scaler')
        st.write('Standardization of a dataset is a common requirement for many machine learning estimators: they might behave badly if the individual features do not more or less look like standard normally distributed data (e.g. Gaussian with 0 mean and unit variance).')
        if scaler_on:
            st.write(":heavy_check_mark:")
            st.write('When it is on, standard scaler will normalize the features, so that each class will have μ = 0 and σ = 1') 
        else:
            st.write(":o:")
            st.write('When it is off, features stay with the same values.')

    
    with _:
        st.subheader('Seed')
        
       


my_expander = st.expander(label='Running model')
with my_expander:
    'The algorithm that was chosen to run this data was SVC'

    if st.button('Run Model'):
        accuracy, acuracia_baseline, x_train_size, x_test_size, teste_x, teste_y, model = ml_model(x, y, SEED, scaler_on, stratify_test)
        #st.text(scaler_on)
        size = x_test_size + x_train_size
        xx, yy, Z, data_x, data_y =countour_plot(teste_x, feature_1, feature_2, model, scaler=scaler_on)
        
        st.subheader("Baseline model Accuracy")
        st.text(str(round(acuracia_baseline,2)) + ' %')
        
        left_column, middle1_column, middle2_column, middle3_column, right_column = st.columns(5)
           
        with left_column:
            st.subheader("Accuracy")
            st.text(str(round(accuracy,2)) + ' %')

        with middle1_column:
            st.subheader("Train size")
            st.text(str(x_train_size) + ' ('+str(round(100*x_train_size/size,1)) + ' %)')

        with middle2_column:
            st.subheader("Test size")
            st.text(str(x_test_size) + ' ('+str(round(100*x_test_size/size,1)) + ' %)')
            
        with middle3_column:
            st.subheader("Scaler")
            st.text(scaler_on)
        
        with right_column:
            st.subheader("Stratify")
            st.text(stratify_test)

        # fig, ax = plt.subplots()

        # ax.contourf(xx, yy, Z, cmap='Blues')
        # # plt.contourf(xx, yy, Z, alpha = 0.3)
        # ax.scatter(data_x, data_y, c = teste_y, s=10)

        # st.pyplot(fig)

        
        
        import plotly.graph_objects as go
        
        fig = go.Figure()

        fig.add_trace(
            go.Contour(
                z=Z,
                colorscale='magenta',
                
                
                # ncontours=30,
                showscale=False
            )
        )
        colors={0:'red', 1:'blue'}
        classe={0:'Speech', 1:'Music'}
        
        df=pd.DataFrame()
        df['x']=(rescale(100, 0, data_x.max(), data_x.min(), data_x))
        df['y']=(rescale(100, 0, data_y.max(), data_y.min(), data_y))
        df['class']=teste_y

        fig.add_trace(
            go.Scatter(
                x=df['x'].tolist(),#data_x,
                y=df['y'].tolist(),#data_y,
                mode="markers",
                marker=dict(color=teste_y.map(colors).tolist()),
                text=teste_y.map(classe).tolist()
                #color='blue'
                # line=dict(
                #     color="black"
                )
            )


        st.plotly_chart(fig)
    

my_expander = st.expander(label='Optimization')
with my_expander:
    if st.button('Optimize Model'):
        optimizing_df = pd.DataFrame(columns=['Feature_1', 'Feature_2','Accuracy'])

        combinacoes = list(itertools.combinations(features, 2))
        
        counter=0
        for combinacao in combinacoes:
            #st.text(combinacao[0]+' '+combinacao[1])
            
            df_=df[[combinacao[0], combinacao[1], 'music']].copy().dropna()
            x = df_[[combinacao[0], combinacao[1]]]
            y = df_["music"]
            
            accuracy, acuracia_baseline, x_train_size, x_test_size, teste_x, teste_y, model = ml_model(x, y, SEED, scaler_on, stratify_test)
            optimizing_df.loc[counter, :]=[combinacao[0], combinacao[1], accuracy]            
            counter+=1
        
        st.dataframe(optimizing_df.sort_values(by='Accuracy', ascending=False))
    

st.write('The audio dataset used was MUSAN and can be found at this [link](https://dblp.uni-trier.de/rec/journals/corr/SnyderCP15.htmlS).')
#st.write("check out this [link](https://share.streamlit.io/mesmith027/streamlit_webapps/main/MC_pi/streamlit_app.py).")

st.write('\nAlso the pre-processed features table and code that lead to it are bellow:')
