import streamlit as st
import pandas as pd
import inference

st.title("Generate")

concepts = [l.strip() for l in open("resources/concepts.txt").readlines()]
models = {"Base":"kevincstowe/concept2seq", "SRL":"kevincstowe/concept2seq-srl", "CEFR":"kevincstowe/concept2seq-cefr"}

examples = pd.DataFrame([("girl sit red clothes","a little girl in a red shirt and purple tights sits surrounded by clothes and holding a book."),
            ("sadie door mom unlocked return","sadie returned home when her mom had unlocked the door."),
            ("music feature several famous","several million people gathered in rio de janeiro for a massive celebration featuring fireworks and music on brazil’s famous copacabana beach."),
            ("mary get dirty clothes","mary gets upset when tom leaves his dirty clothes on the floor."),
            ("spanish performer musician music area group outdoor entertain","a group of musicians and performers entertain with spanish music in an outdoor area."),
            ("top push ready clothes","an older gentleman standing in front of a washer and dryer with his hand getting ready to push a button and a black basket of clothes sitting on top of the dryer next to him."),
            ("tent go pink clothes","a man in a gray shirt, a young boy in a blue shirt and a young girl in pink clothes all sitting down in front of tents with a fire going."),
            ("baby clothe wash throw clean clothes","frank cleaned the baby up and threw the clothes in the wash."),
            ("man cross dress clothing","man dressed in orange clothing with face covered seemingly balancing on a cane being held be a similarly dressed man sitting crossed legged on the ground at a shopping mall."),
            ("street separate yellow participate","bikers participate in a two way street ride, with lanes separated by yellow cones."),
            ("mars carry gravitational atmosphere","solar winds carry the thin, weak atmosphere away because mars has a weak gravitational and magnetic field."),
            ("ball lays brown clothes","a young boy with light brown hair, dressed in play clothes runs earnestly as a blue, white, and red small ball lays on the grass behind him."),
            ("cloth sow old clothing","in a old urban house, a old middle eastern man sows together a striped purple and light blue cloth on a wooden table."),
            ("mary weight try promise lose","tom promised mary he would try to lose some weight."),
            ("origin color communal clothes","people of indian origin wash their brightly colored clothes in a communal place."),
            ("shirt cover messy clothes","stacks of shirts and clothes cover a bed in a messy room.")], columns=["Concepts", "Target Sentence"])

concept = st.text_input("Write what concepts you'd like to see in the output:", value="girl sit red clothes", max_chars=64)
#st.write("Some examples")
#st.dataframe(examples)
model_name = st.selectbox("Model", ["Base", "SRL", "CEFR"])
model_path = models[model_name]

if model_name == "SRL":
    st.write("The SRL model can take input concepts along with SRL labels. Join the SRL label to the concept with a hyphen, ie. 'girl-ARG0', 'shirt-ARG1', 'clothes-ARG2'")
elif model_name == "CEFR":
    st.write("The CEFR can take a CEFR level (A1-C2) as the last word in the input, ie. 'girl sit red clothes B2'") 


col1, col2, col3 = st.columns(3)
iterations = col1.number_input("# sentences generated", min_value=1, max_value=100, value=5)
min_length = col2.number_input("Minimum # words", value=2)
max_length = col3.number_input("Maximum # words", value=40)

button_clicked = st.button("Run generation!")

bart = inference.load_model(model_path)

if button_clicked:
    st.write("Outputs")
    for o in inference.generate(bart, concept, min_length=min_length, max_length=max_length, iterations=iterations):
        st.write(o)
