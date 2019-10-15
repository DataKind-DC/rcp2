# -*- coding: utf-8 -*-

# Imports 
import numpy as np
import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html




app = dash.Dash()

app.layout = html.Div(children= [ 
                html.H1 ('DataKind FireMap'),
                dcc.Graph(id ='example',
                    figure = {
                      'data': [ 
                    {'x':[1,2,3,4] , 'y':[2,2,4,2], 'type':'line' ,'name':'boats'},
                    {'x':[1,2,3] , 'y':[4,2,1], 'type':'bar' ,  'name':'cars'}
                                  ],
                      'layout' : {
                              'title':'Basic title'
                              
                              
                              }
                          
                          })
                   ])

if __name__ == '__main__':
    app.run_server(debug = True)
