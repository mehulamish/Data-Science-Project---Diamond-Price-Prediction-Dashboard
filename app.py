import dash
from dash import dcc, html, Input, Output
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# Load the data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Convert 'cut', 'color', and 'clarity' variables to categorical
for df in [train, test]:
    df['cut'] = pd.Categorical(df['cut'])
    df['color'] = pd.Categorical(df['color'])
    df['clarity'] = pd.Categorical(df['clarity'])

# Feature engineering
def engineer_features(df):
    df['volume'] = df['x'] * df['y'] * df['z']
    df['density'] = df['carat'] / (df['volume'] + 0.000001)
    df['depth_per_volume'] = df['depth'] / (df['volume'] + 0.000001)
    df['depth_per_density'] = df['depth'] / (df['density'] + 0.000001)
    df['depth_per_table'] = df['depth'] / (df['table'] + 0.000001)
    return df

train = engineer_features(train)
test = engineer_features(test)

# Split the data
X = train.drop('price', axis=1)
y = train['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Train the model
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.08, max_depth=5, random_state=123)
model.fit(X_train, y_train)

# Make predictions
preds = model.predict(X_test)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test, preds))

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout
app.layout = html.Div([
    html.H1("Gemstone Price Dataset Dashboard"),
    
    html.Div([
        html.Div([
            html.H4("Choose a dataset:"),
            dcc.Dropdown(
                id='dataset',
                options=[{'label': i, 'value': i} for i in ['Train', 'Test']],
                value='Train'
            ),
            html.Br(),
            html.Hr(),
            html.H4("Choose a plot type:"),
            dcc.Dropdown(
                id='plot_type',
                options=[{'label': i, 'value': i} for i in [
                    "Cut Value Counts (Bar)", "Color Value Counts (Bar)", "Clarity Value Counts (Bar)",
                    "Cut vs Price (Boxplot)", "Color vs Price (Boxplot)", "Clarity vs Price (Boxplot)"
                ]],
                value="Cut Value Counts (Bar)"
            ),
            html.Br(),
            html.Hr(),
            html.H5("Root Mean Squared Error"),
            html.Div(id='rmse'),
            html.H5("Missing Values"),
            html.Div(id='missing_values'),
            html.H5("Value Counts"),
            html.Div(id='value_counts'),
            html.Br(),
            html.Hr(),
            html.H4("Choose corr column"),
            dcc.Dropdown(
                id='corr_column',
                options=[{'label': i, 'value': i} for i in [
                    "x", "y", "z", "color", "clarity", "carat", "cut", "depth", 
                    "depth_per_volume", "depth_per_density", "depth_per_table"
                ]],
                value="x"
            ),
        ], style={'width': '25%', 'float': 'left'}),

        html.Div([
            dcc.Graph(id='main_plot'),
            html.Br(),
            html.Hr(),
            dcc.Graph(id='corrplot'),
            html.Br(),
            html.Hr(),
            dcc.Graph(id='comparisonPlot'),
            html.Br(),
            html.Hr(),
            dcc.Graph(id='priceCompPlot'),
        ], style={'width': '75%', 'float': 'right'})
    ])
])

# Callback for updating plots
@app.callback(
    [Output('main_plot', 'figure'),
     Output('corrplot', 'figure'),
     Output('comparisonPlot', 'figure'),
     Output('priceCompPlot', 'figure'),
     Output('rmse', 'children'),
     Output('missing_values', 'children'),
     Output('value_counts', 'children')],
    [Input('dataset', 'value'),
     Input('plot_type', 'value'),
     Input('corr_column', 'value')]
)
def update_plots(dataset, plot_type, corr_column):
    df = train if dataset == 'Train' else test

    # Main plot
    if plot_type.endswith("(Bar)"):
        var = plot_type.split()[0].lower()
        fig = px.bar(df[var].value_counts().reset_index(), x='index', y=var)
    else:
        var = plot_type.split()[0].lower()
        fig = px.box(df, x=var, y='price', color=var)

    # Correlation plot
    corr = df.select_dtypes(include=[np.number]).corr()
    corr_fig = px.imshow(corr, color_continuous_scale='RdBu_r', zmin=-1, zmax=1)

    # Comparison plot
    comp_fig = px.scatter(df, x=corr_column, y='price')

    # Price comparison plot
    price_comp_fig = px.scatter(x=y_test, y=preds)
    price_comp_fig.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], 
                                        y=[y_test.min(), y_test.max()], 
                                        mode='lines', line=dict(color='red')))

    # RMSE
    rmse_text = f"Root Mean Squared Error: {rmse:.2f}"

    # Missing values
    missing_vals = df.isnull().sum().to_string()

    # Value counts
    value_counts = df['cut'].value_counts().to_string()

    return fig, corr_fig, comp_fig, price_comp_fig, rmse_text, missing_vals, value_counts

if __name__ == '__main__':
    app.run_server(debug=True)
