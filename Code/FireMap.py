# -*- coding: utf-8 -*-

# Imports 
import numpy as np
import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html

df = pd.read_csv('/Users/kelson/Documents/DataKind/Red_Cross_Fire/Data/Transformed/NFIRS_2009_2016_Combined_Census_Tract.csv',
                 encoding = 'latin1',
                 low_memory =0)
df = df.head()    

app = dash.Dash()
scl = [ [0,"rgb(5, 10, 172)"],[0.35,"rgb(40, 60, 190)"],[0.5,"rgb(70, 100, 245)"],\
    [0.6,"rgb(90, 120, 245)"],[0.7,"rgb(106, 137, 247)"],[1,"rgb(220, 220, 220)"] ]

data = [ dict(
        type = 'scattergeo',
        locationmode = 'USA-states',
        lon = df['X'],
        lat = df['Y'],
       # text = df['ID'],
        mode = 'markers',
        marker = dict(
            size = 8,
            opacity = 0.8,
            reversescale = True,
            autocolorscale = False,
            symbol = 'square',
            line = dict(
                width=1,
                color='rgba(102, 102, 102)'
            ),
            colorscale = scl,
            cmin = 0,
           # color = df['cnt'],
           # cmax = df['cnt'].max(),
           # colorbar=dict(
           #     title="Incoming flightsFebruary 2011"
           #)
           color = 1 
        ))]

layout = dict(
        title = 'Most trafficked US airports<br>(Hover for airport names)',
        colorbar = True,
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showland = True,
            landcolor = "rgb(250, 250, 250)",
            subunitcolor = "rgb(217, 217, 217)",
            countrycolor = "rgb(217, 217, 217)",
            countrywidth = 0.5,
            subunitwidth = 0.5
        ),
    )

fig = dict( data=data, layout=layout )    

app.layout  = html.Div([
    dcc.Graph(id='graph', figure=fig)
])


if __name__ == '__main__':
    app.run_server(debug = True)
