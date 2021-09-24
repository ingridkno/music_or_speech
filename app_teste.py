import streamlit as st
import pandas as pd
#import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import itertools
import graphviz as graphviz

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

st.sidebar.subheader('Features')
feature_1 = st.sidebar.selectbox('1st feature',features, 3)
feature_2 = st.sidebar.selectbox('2nd feature',features, 7)

#st.sidebar.markdown("---")

st.sidebar.subheader('Parameters')
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
    st.write('*by Ingrid Knochenhauer de Souza Mendes*')
with middle_column:
    st.video(video_bytes)
    
st.subheader('\nA simple machine learning model applied to audio')
st.write('\nIn this app, we will be able to run through the steps to build a simple machine learning model applied to audio.') 


st.subheader('CHALLENGE')
st.write('\nImagine you work for a famous and solid radio station. There are old programs that are being digitized from cassettes and your first mission is to organize the audio files. In this organization, you want to classify what is **music** and what is **speech** or broadcast. You have already a classified dataset and you want to use the help of **artificial intelligence** to optimize your work.')
st.write('**What would be your first step?**')

st.write('In a machine learning model that predicts data from an existent dataset, the process are the following steps:')

# Create a graphlib graph object
graph = graphviz.Digraph(graph_attr={'rankdir':'LR'})

graph.attr('node', shape='box')

graph.edge('Data Input', 'Develop Model')
graph.edge('Develop Model', 'Train Model')
graph.edge('Train Model', 'Test and Analysis')

st.graphviz_chart(graph)

st.write("**Let's detail more about them!**")


st.subheader('Data Input')

my_expander = st.expander(label='What will be the input in your model?')
with my_expander:
    'Features are individual independent variables that act like a input in your system and are responsible for the prevision result. They are individual measurable properties or characteristics of a phenomenon.'
    st.write('In this first project, the focus will not be in choosing the features or in the feature extraction itself. As it written, features that are used in speech-music classification were already extracted. The chosen ones, in order to exemplify this machine learning model, were:')
    
    stats_list=[]
    feat_list=[]
    dict_stats = {'mean':'Mean', 'std': 'Standard Deviation', 'count': 'Count', 'max': 'Maximum value'}
    dict_feats = {'centroid': 'Centroid', 'flatness': 'Flatness', 'rms': 'RMS value', 'zcr': 'Zero Crossing Rate (zcr)', 'fund_freq':'Fundamental Frequency'}
        
    
    for feature in features:
        stats = (feature.split('_')[-1])
        feat = ("_".join(feature.split('_')[:-1]))
        
        stats_list.append(stats)
        feat_list.append(feat)

    left_column, right_column = st.columns(2)
    
    
    with left_column:
        st.subheader('Acoustics features :musical_note: :speech_balloon:')
        for name in list(set(feat_list)):
            if (name in feature_1) or (name in feature_2):
                st.write('* **'+dict_feats[name]+'**')
            else:
                st.write('* '+dict_feats[name])  
        
        
    with right_column:              
        st.subheader('Statistical measures along time :chart_with_upwards_trend:')
        for name in list(set(stats_list)):
            if (name in feature_1) or (name in feature_2):
                st.write('* **'+dict_stats[name]+'**')
            else:
                st.write('* '+dict_stats[name])
        # st.write('* Mean')
        # st.write('* Standard Deviation (std)')
        # st.write('* Maximum value (max)')
        # st.write('* Count')
    st.write(' ')
    st.write(':point_left: Along these steps, you can choose two of these features in the **sidebar** and see how your machine learning model would classify music from speech.')
    st.write(' ')
    st.write('[Librosa] (https://librosa.org/doc/main/feature.html) was the library used to extract the acoustics features and the dataset was [**MUSAN**](https://www.openslr.org/17/). You can check the [table] (https://github.com/ingridkno/music_or_speech/blob/main/musan_features_mod.csv) with the features that was generated from this [code] (https://github.com/ingridkno/music_or_speech/blob/main/audio_features_musan.ipynb). ')
    

my_expander = st.expander(label='How do the features behave? ')
with my_expander:
    'Before you develop the model, it is important to explore your dataset and see how the features behave. Probably, you will come back to this step over and over again because it is here, generally, you will develop insights to improve your model later.'
    #clicked = st.button('Click me!')
    st.write('In the graphs below, you can see the **stratified** dataset for the two classes and also a **2D analysis** for the two chosen features in the sidebar.')
    left_column, right_column = st.columns((3,5))

    with left_column:
        st.plotly_chart(fig_2)
        st.write('In this bar graph, data composition shows that the MUSAN **dataset is not balanced**. There are more audio files labeled as music than as speech.')
        
    with right_column: 
        st.plotly_chart(fig)
        st.write('In this scatter plot graph, the features chosen in the sidebar are shown against each other. Depending on them, **data can be more or less clusterized** to speech-music classification.')
        
st.subheader('Develop Model')

my_expander = st.expander(label='Parameters')
with my_expander:
    'There are many algorithms that can be used behind the scenes of a machine learning model. For this project, the algorithm chosen was [**SVC**] (https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) - Support Vector Classification and some important parameters for its running can be changed in the sidebar and influence our predictions.'
    st.write('Check out the explanation for the parameters turning them on and off and choosing the seed number.')
    #clicked = st.button('Click me!')

    left_column, right_column, _ = st.columns(3)

    with left_column:
        st.subheader('Stratify')
        st.write('This parameter is about split in the training and test data.')
        if stratify_test:
            st.write(":heavy_check_mark:")
            st.write('When it is on, test data is split in a stratified fashion. That means that train and test data will have approximately same percentage to each class.') 
        else:
            st.write(":o:")
            st.write('When it is off, quantity of each class in test data will be chosen randomly.')
    
    with right_column:
        st.subheader('Standard Scaler')
        st.write('Standardization of a dataset is a common requirement for many machine learning estimators: they might behave badly if the individual features do not more or less look like standard normally distributed data.')
        if scaler_on:
            st.write(":heavy_check_mark:")
            st.write('When it is on, standard scaler will normalize the features, so that each class will have mean(μ) = 0 and standard deviation(σ) = 1')
        else:
            st.write(":o:")
            st.write('When it is off, features stay with the same values. If there is a vast difference in the range, the machine learning model makes the underlying assumption that higher ranging numbers have more influence in the prediction result.')

    
    with _:
        st.subheader('Seed')
        st.write("If you use the same seed you will get exactly the same pattern of numbers. This means that whether you're making a train test split or fitting this machine learning model, setting a seed will be giving you the same set of results time and again.")
        st.write('**SEED NUMBER**: :seedling:'+str(SEED)+' :seedling:')

        
       

st.subheader('Train Model')
my_expander = st.expander(label='Running model')
with my_expander:
    st.write('Having chosen the features and parameters, it is time to train and test model. However, before that, it is important to stablish a measurement and a baseline model to compare the results.')
    st.subheader('Measurement')
    st.write("**Accuracy** is the measurement chosen to evaluate the models. That will state how many correct predictions compared to total number of predictions.") 
    st.subheader('Baseline model')
    st.write('A baseline is the result of a very basic model. In this project, the baseline will predict our test results as the class with more samples, music. So, our initial goal model is to achieve a better score than the baseline.')
             
             
    if st.button('Run Model'):
        accuracy, acuracia_baseline, x_train_size, x_test_size, teste_x, teste_y, model = ml_model(x, y, SEED, scaler_on, stratify_test)
        #st.text(scaler_on)
        size = x_test_size + x_train_size
        xx, yy, Z, data_x, data_y =countour_plot(teste_x, feature_1, feature_2, model, scaler=scaler_on)
        
        st.subheader("Baseline Model Accuracy")
        st.write(':pushpin: '+ str(round(acuracia_baseline,1)) + ' %')
        
        left_column, middle1_column, middle2_column, middle3_column, right_column = st.columns(5)
           
        with left_column:
            st.subheader("Accuracy")
            st.write(str(round(accuracy,1)) + ' %')

        with middle1_column:
            st.subheader("Train size")
            st.write(str(x_train_size) + ' samples')
            st.write('('+str(round(100*x_train_size/size,1)) + ' %)')

        with middle2_column:
            st.subheader("Test size")
            st.write(str(x_test_size) + ' samples')
            st.write('('+str(round(100*x_test_size/size,1)) + ' %)')
            
        with middle3_column:
            st.subheader("Scaler")
            if scaler_on:
                st.write(":heavy_check_mark: "+str(scaler_on))
            else:
                st.write(":o: "+str(scaler_on))
                
        with right_column:
            st.subheader("Stratify")
            if stratify_test:
                st.write(":heavy_check_mark: "+str(stratify_test))
            else:
                st.write(":o: "+str(stratify_test))
            
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

        st.subheader(':crystal_ball: Prediction Curve')
        st.plotly_chart(fig)
    
st.subheader('Test and Analysis')
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
        
        st.dataframe(optimizing_df.sort_values(by='Accuracy', ascending=False).reset_index())

my_expander = st.expander(label='Congratulations!')
with my_expander:
    'You have been through the first steps of a machine learning model.'
    st.write('**What is next?!?**')
    st.write('Follow me for more upcoming tutorials!')
    st.balloons()
    
