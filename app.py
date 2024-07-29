from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq

from dash import Dash, html, dcc, callback, Output, Input, State
import dash_ag_grid as dag
import pandas as pd
import os
from assets.api_keys import groq_key
import re

# GROQ_API_KEY = groq_key

df = pd.read_csv('data/GEDEvent_v24_1.csv')
df_5_rows = df.head()
csv_string = df_5_rows.to_string(index=False)

#chose the model
model = ChatGroq(
    api_key=groq_key,
    model="llama3.1-70b-versatile"
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You're a data visulaization expert and use your favourite graphin library Plotly only.  Suppose, that "
            "the data is provided as a GEDEvent_v24_1.csv file and consists of a list of violent events throught the world.  "
            "Here are the first 5 rows of the data set: {data} "
            "Follow the user's instructions when creating the graph."
        ),
        MessagesPlaceholder(variable_name="messages")
    ]
)
chain = prompt | model

def get_fig_from_code(code):
    local_variables = {}
    exec(code, {}, local_variables)
    return local_variables['fig']

app = Dash()
server = app.server
app.layout = html.Div([
    html.H1("Plotly AI for Creating Graphs"),
    dag.AgGrid(
        rowData=df.to_dict("records"),
        columnDefs = [{"field": i} for i in df.columns],
        defaultColDef={"filter": True, "sortable": True, "floatingFilter": True}
    ),
    dcc.Textarea(id= 'user-request', style={'width': '50%', 'height':50, 'margin-top': 20}),
    html.Br(),
    html.Button('Submit', id='my-button'),
    dcc.Loading(
        [
            html.Div(id='my-figure', children=''),
            dcc.Markdown(id='content', children='')
        ],
        type='cube'
    )
])

@callback(
    Output('my-figure', 'children'),
    Output('content', 'children'),
    Input('my-button', 'n_clicks'),
    State('user-request', 'value'),
    prevent_initial_call = True
)
def create_graph(_, user_input):
    response = chain.invoke(
        {
            "messages": [HumanMessage(content=user_input)],
            "data": csv_string
        },
    )
    result_output =  response.content
    print(result_output)
    
    # check if the answer includes code. This regular expression will match code blocks
    # with or without the language specifier (like 'python')
    code_block_match = re.search(r'```(?:[Pp]ython)?(.*?)```', result_output, re.DOTALL)
    print(code_block_match)
    if code_block_match:
          code_block =  code_block_match.group(1).strip()
          cleaned_code = re.sub(r'(?m)*\s=fig\.show\(\)\s=$', '', code_block)
          fig =get_fig_from_code(cleaned_code)
          return dcc.Graph(figure=fig), result_output
    else:
        return "", result_output

if __name__ == '__main__':
    app.run(debug=True, port=3050)
