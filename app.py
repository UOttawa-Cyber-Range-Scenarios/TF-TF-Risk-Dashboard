import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import random
from car_data import car_data_dict
from datetime import datetime, timedelta
from itertools import groupby

# Set page config for wide layout
st.set_page_config(page_title="Autonomous Vehicle Dashboard", layout="wide")

#px.set_mapbox_access_token(open(".mapbox_token").read())

@st.cache_resource()
def get_admin_controls():
    return {
        "HQ": "Online",
        "Toronto": "Online",
        "Vancouver": "Online"
    }

# Function to get admin inputs
@st.cache_resource()
def get_admin_inputs():
    return {
        'risk':'Low',
        'risk-value':0,
        'date_start': datetime.now() - timedelta(days = 7),
        'date_end': datetime.now(),
        'module':1
    }

@st.cache_data()
def generate_data(date_range):
    #np.random.seed(42)

    date_ranges=[]
    dfs = []
    for dates in range(len(date_range)):
        date_ranges = pd.date_range(start=date_range[dates][0], end=date_range[dates][1], freq='h')
        n = len(date_ranges)
    
        data = {
            'time': date_ranges,
            'server_load': np.random.uniform(10, 60, n),
            'uptime': np.random.uniform(95, 100, n),
            'latency': np.random.uniform(10, 50, n),
            'trip_count': np.random.randint(1, 10, n),
            'trip_duration': np.random.uniform(5, 60, n),
            'activity': np.random.uniform(0, 20, n),
            'emissions': np.random.uniform(50, 150, n),
            'fuel_consumption': np.random.uniform(10, 50, n),
            'energy_usage': np.random.uniform(20, 80, n)
        }

        df = pd.DataFrame(data)

        if dates == 1:
            # Introduce a few anomalies for medium risk level
            anomalies = np.random.choice([True, False], size=n, p=[0.1, 0.9])
            df.loc[anomalies, 'server_load'] = np.random.uniform(70, 100, anomalies.sum())
            df.loc[anomalies, 'activity'] = np.random.uniform(70, 100, anomalies.sum())
            df.loc[anomalies, 'latency'] = np.random.uniform(50, 80, anomalies.sum())
        
        elif dates == 4:
            # Introduce a few anomalies for medium risk level
            anomalies = np.random.choice([True, False], size=n, p=[0.8, 0.2])
            df.loc[anomalies, 'server_load'] = np.random.uniform(80, 100, anomalies.sum())
            df.loc[anomalies, 'activity'] = np.random.uniform(80, 100, anomalies.sum())
            df.loc[anomalies, 'latency'] = np.random.uniform(70, 90, anomalies.sum())
            df.loc[anomalies, 'uptime'] = np.random.uniform(20, 40, anomalies.sum())
            # Corrupt or missing data
            missing_indices = np.random.choice(df.index, size=int(n*0.2), replace=False)
            for col in ['emissions', 'fuel_consumption', 'energy_usage', 'activity', 'server_load']:
                df.loc[missing_indices, col] = np.nan
        
        elif dates == 5:
            # Introduce a few anomalies for medium risk level
            anomalies = np.random.choice([True, False], size=n, p=[0.8, 0.2])
            df.loc[anomalies, 'server_load'] = np.random.uniform(80, 100, anomalies.sum())
            df.loc[anomalies, 'activity'] = np.random.uniform(80, 100, anomalies.sum())
            df.loc[anomalies, 'latency'] = np.random.uniform(70, 90, anomalies.sum())
            df.loc[anomalies, 'uptime'] = np.random.uniform(20, 40, anomalies.sum())
            # Corrupt or missing data
            missing_indices = np.random.choice(df.index, size=int(n), replace=False)
            for col in ['emissions', 'fuel_consumption', 'energy_usage', 'activity', 'server_load','uptime','latency','trip_count','trip_duration']:
                df.loc[missing_indices, col] = np.nan

        dfs.append(df)


    return pd.concat(dfs, ignore_index=True)

# Function to get online cars based on risk level
def get_online_cars(car_data, risk_level):
    if risk_level == 'Low':
        online_percentage = random.uniform(0.75, 0.99)
    elif risk_level == 'Medium':
        online_percentage = random.uniform(0.45, 0.75)
    elif risk_level == 'High':
        online_percentage = random.uniform(0.1, 0.45)
    else:
        online_percentage = 0.0
    
    num_online_cars = int(len(car_data) * online_percentage)
    online_cars = random.sample(car_data, num_online_cars)
    return online_cars

# Function to display additional vehicle statistics
def display_vehicle_statistics(online_cars):
    st.subheader("Vehicle Statistics")
    total_vehicles = 100
    vehicles_online = online_cars
    average_miles_driven = random.randint(2900, 3100)
    total_data_transferred = random.randint(480, 520)  # in GB

    col1, col2 = st.columns(2)

    return {
        'online': vehicles_online,
        'avg_miles': average_miles_driven,
        'data':total_data_transferred
    }

def create_new_map(online_cars):
    # Define the city names
    cities = {
        "Toronto",
        "Vancouver",
        "Montreal",
        "Calgary",
        "Ottawa",
        "Edmonton",
        "Quebec City",
        "Winnipeg"
    }

    # Create a dataframe for the car data
    car_data = []
    for car in online_cars:
        car_data.append([car["latitude"], car["longitude"], car["id"], car["car_model"], car["ai_model"], car["ai_version"], car["miles_driven"], car["city"], car['system_upgrade']])

    car_df = pd.DataFrame(car_data, columns=['lat', 'lon', 'id', 'car_model', 'ai_model', 'ai_version', 'miles_driven', 'city','system_upgrade'])

    # Plotting cars
    fig = px.scatter_mapbox(car_df, lat="lat", lon="lon", hover_data={"car_model": True, "ai_model": True, "ai_version": True, "miles_driven": True, "city": True, 'system_upgrade':True},
                            color='city', zoom=3, height=500, mapbox_style='carto-darkmatter',size='miles_driven')

    fig.update_layout(mapbox_style="carto-darkmatter",showlegend=False)
    #fig.update_traces(cluster=dict(enabled=True))

    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

    return fig


# Function to create a map of buildings
def create_service_map(num_affected_points):
    # Headquarters and servers data
    headquarters = {
        "HQ": [45.350143, -76.072208],
        "Toronto": [43.829752, -79.331226],
        "Vancouver": [49.277201, -123.061013]
    }
    servers = {
        "HQ": [[45.523748, -73.684116]],
        "Toronto": [[47.369253, -82.124214]],
        "Vancouver": [[51.027184, -114.247661], [53.435266, -113.694941]]
    }

    hq_data = []
    server_data = []
    line_data = []

    for key, loc in headquarters.items():
        status = num_affected_points[key]
        color = 'blue' if status == 'Online' else 'lightgray' if status == 'Isolated' else 'red'
        size = 15 if status == 'Critical' else 10
        hq_data.append([loc[0], loc[1], key, status, color, size])

        for server in servers[key]:
            server_data.append([server[0], server[1], key, status, 'lightgreen' if status == 'Online' else 'lightgray' if status == 'Isolated' else 'red'])
            if status == 'Online':
                line_data.append([loc[0], loc[1], server[0], server[1], 'lightgreen'])
            elif status == 'Critical':
                line_data.append([loc[0], loc[1], server[0], server[1], 'red'])

    hq_df = pd.DataFrame(hq_data, columns=['lat', 'lon', 'name', 'status', 'color', 'size'])
    server_df = pd.DataFrame(server_data, columns=['lat', 'lon', 'name', 'status', 'color'])
    line_df = pd.DataFrame(line_data, columns=['lat1', 'lon1', 'lat2', 'lon2', 'color'])

    # Plotting headquarters and servers
    fig = px.scatter_mapbox(hq_df, lat="lat", lon="lon", color="color", size="size", text="name",
                            color_discrete_map={"blue": "blue", "lightgray": "lightgray", "red": "red"},
                            size_max=15, zoom=3, height=500)
    fig.add_trace(px.scatter_mapbox(server_df, lat="lat", lon="lon", color="color",
                                    color_discrete_map={"lightgreen": "lightgreen", "lightgray": "lightgray", "red": "red"}).data[0])

    # Adding lines
    for _, row in line_df.iterrows():
        fig.add_trace(go.Scattermapbox(
            lat=[row['lat1'], row['lat2']],
            lon=[row['lon1'], row['lon2']],
            mode='lines',
            line=dict(color=row['color']),
            showlegend=False
        ))

    fig.update_layout(mapbox_style="carto-darkmatter",showlegend=False)
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

    return fig

def create_gradient_colors(num_colors):
    # Gradient from blue to green to yellow to red
    colors = []
    for i in range(num_colors):
        if i < num_colors // 3:
            # Transition from blue to green
            ratio = i / (num_colors // 3)
            r = int(0 + ratio * (0 - 0))
            g = int(0 + ratio * (255 - 0))
            b = int(255 * (1 - ratio))
        elif i < 2 * (num_colors // 3):
            # Transition from green to yellow
            ratio = (i - num_colors // 3) / (num_colors // 3)
            r = int(0 + ratio * (255 - 0))
            g = 255
            b = 0
        else:
            # Transition from yellow to red
            ratio = (i - 2 * (num_colors // 3)) / (num_colors // 3)
            r = 255
            g = int(255 * (1 - ratio))
            b = 0
        colors.append(f'rgba({r}, {g}, {b}, 1)')
    return colors
# Function to create a risk donut chart
def display_risk_chart(risk):
    # Define risk label
    if risk <= 33:
        risk_label = "Low Risk"
    elif risk <= 67:
        risk_label = "Medium Risk"
    else:
        risk_label = "High Risk"

    # Number of segments in the gradient
    num_segments = 100
    colors = ['#00FF00' for i in range(num_segments)]#create_gradient_colors(num_segments)
    
    values = [1] * num_segments
    colors = colors[:risk] + ['#000000'] * (num_segments - risk)

    fig=go.Figure()

    fig.add_trace(go.Pie(
        values=[0,2],
        marker=dict(line=dict(width=8, color='#e0e0e0')),
        hole=0.6,
        sort=False,
        direction='clockwise',
        textinfo='none',
        showlegend=False,
        hoverinfo='none',
        domain={'x': [0, 1], 'y': [0, 1]},
        opacity=1
    ))

    fig.add_trace(
        go.Pie(
        values=values,
        marker=dict(colors=colors, line=dict(width=2, color=colors)),
        hole=0.6,
        sort=False,
        direction='clockwise',
        textinfo='none',
        showlegend=False,
        hoverinfo='none',
    ))



    # Define annotations
    f_color = '#E0E0E0' if risk == 0 else colors[risk-1]
    annotations = [
        dict(
            text=f"{risk}%",
            x=0.5,
            y=0.5,
            font_size=60,
            showarrow=False,
            font=dict(
                color=f_color
            ),
        ),

    ]

    
    fig.update_layout(
        margin=dict(t=10, b=10, l=10, r=10),
        font=dict(size=34),
        annotations=annotations,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=718,
        title=dict(font=dict(color=f_color, size=60), text=risk_label,y=0,automargin=True),
        title_x=0.5,
        title_xanchor='center'
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Function to create a network latency chart
def create_latency_chart(df):
    '''data = {
        'time': pd.date_range(start='1/1/2024', periods=100, freq='H'),
        'latency': np.random.uniform(10, 50, 100)
    }
    df = pd.DataFrame(data)'''
    fig = px.line(df, x='time', y='latency', title='Network Latency',color_discrete_sequence=['#00ff00'])
    return fig

# Function to create a connection status chart
def create_connection_status_chart(car_data):
    '''data = {
        'car_id': [f'Car {i}' for i in range(1, 101)],
        'status': np.random.choice(['online', 'offline'], 100)
    }
    df = pd.DataFrame(data)'''
    citywise_connections = {
        "Toronto":0,
        "Vancouver":0,
        "Montreal":0,
        "Calgary":0,
        "Ottawa":0,
        "Edmonton":0,
        "Quebec City":0,
        "Winnipeg":0
    }

    for car in car_data:
        citywise_connections[car['city']] += 1
    
    df = pd.DataFrame(list(citywise_connections.items()), columns=['Cities', 'Connections'])
    
    fig = px.bar(df, x='Cities', y='Connections', title='Connection Status',color_discrete_sequence=['#00ff00'])
    return fig

# Function to create a data transfer rates chart
def create_data_transfer_chart(df):
    fig = px.bar(df, x='id', y='ai_version', title='Data Transfer Rates',color_discrete_sequence=['#00ff00'])
    return fig

# Function to create a server uptime chart
def create_server_uptime_chart(df):
    fig = px.line(df, x='time', y=['uptime','server_load'], title='Server Uptime')#, color_discrete_sequence=['#00ff00'])
    return fig

# Function to create a server load chart
def create_server_load_chart(df):
    '''data = {
        'time': pd.date_range(start='1/1/2024', periods=100, freq='H'),
        'cpu_load': np.random.uniform(10, 90, 100),
        'memory_load': np.random.uniform(10, 90, 100),
        'disk_usage': np.random.uniform(10, 90, 100)
    }
    df = pd.DataFrame(data)'''
    fig = px.line(df, x='time', y='server_load' , title='Server Load',color_discrete_sequence=['#00ff00']) #y=['cpu_load', 'memory_load', 'disk_usage']
    return fig

# Function to create a VLAN activity chart
def create_vlan_activity_chart(df):
    '''data = {
        'time': pd.date_range(start='1/1/2024', periods=100, freq='H'),
        'activity': np.random.uniform(0, 100, 100)
    }
    df = pd.DataFrame(data)'''
    fig = px.line(df, x='time', y='activity', title='VLAN Activity', color_discrete_sequence=['#00ff00'])
    return fig

# Function to create an incident report chart
def create_incident_report_chart(risk):
    data = {
        'incident_type': ['connectivity', 'malfunction', 'cybersecurity']
    }
    if risk == 'Low':
        data['count'] = [random.randint(1, 2) for _ in range(2)]+[0]
    elif risk == 'Medium':
        data['count'] = [random.randint(3, 7) for _ in range(2)]+[random.randint(1, 5)]
    elif risk == 'High':
        data['count'] = [random.randint(8, 15) for _ in range(2)]+[random.randint(6, 12)]

    df = pd.DataFrame(data)
    fig = px.bar(df, x='incident_type', y='count', title='Incident Types',color_discrete_sequence=['#00ff00'])
    fig.update_yaxes(range = [0,20])
    return fig

# Function to create a usage metrics chart
def create_usage_metrics_chart(df):
    '''data = {
        'time': pd.date_range(start='1/1/2024', periods=100, freq='H'),
        'trip_count': np.random.randint(1, 10, 100),
        'trip_duration': np.random.uniform(5, 60, 100)
    }
    df = pd.DataFrame(data)'''
    fig = px.line(df, x='time', y=['trip_count', 'trip_duration'], title='Usage Metrics',color_discrete_sequence=['#00ff00'])
    return fig

# Function to create a heatmap of vehicle locations
def create_vehicle_heatmap(df):
    fig = px.density_mapbox(df, lat='latitude', lon='longitude', z='miles_driven', radius=10, center=dict(lat=45.4215, lon=-75.6972), zoom=0,
                            mapbox_style="stamen-terrain")
    return fig

# Function to create a route analysis chart
def create_route_analysis_chart():
    data = {
        'route': [f'Route {i}' for i in range(1, 11)],
        'usage': np.random.randint(1, 100, 10)
    }
    df = pd.DataFrame(data)
    fig = px.bar(df, x='route', y='usage', title='Route Analysis')
    return fig

# Function to create a maintenance and updates chart
def create_maintenance_chart(df):
    # Convert the list of dictionaries to a pandas DataFrame if it's not already a DataFrame
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    # Count the occurrences of each system_upgrade status
    upgrade_counts = df['system_upgrade'].value_counts().reset_index()
    upgrade_counts.columns = ['System Upgrade', 'Count']

    # Create the bar chart using Plotly Express
    fig = px.bar(upgrade_counts, x='System Upgrade', y='Count', title='Maintenance and Updates',color_discrete_sequence=['#00ff00'])
    return fig

# Function to create an environmental impact chart
def create_environmental_impact_chart(df):
    '''data = {
        'time': pd.date_range(start='1/1/2024', periods=100, freq='D'),
        'emissions': np.random.uniform(50, 150, 100),
        'fuel_consumption': np.random.uniform(10, 50, 100),
        'energy_usage': np.random.uniform(20, 80, 100)
    }
    df = pd.DataFrame(data)'''
    fig = px.line(df, x='time', y=['emissions', 'fuel_consumption', 'energy_usage'], title='Environmental Impact')
    return fig

# Main function to render the dashboard
def main():
    # Sidebar

    admin_controls = get_admin_controls()
    admin_inputs = get_admin_inputs()
    min_risk=0
    max_risk=100
    login = st.sidebar.checkbox("Login as Admin")

    if login:
        username = st.sidebar.text_input("Username", key="username")
        password = st.sidebar.text_input("Password", type="password", key="password")
        if username == "admin" and password == "password":
            st.sidebar.success("Login Successful")
            admin_inputs['module'] = st.sidebar.selectbox("Select Module: ",[1,2,3,4,5,6,7], key='module')
            if admin_inputs['module'] in [1,2,6,7]:
                admin_inputs['risk'] = 'Low'
            elif admin_inputs['module'] in [3,4]:
                admin_inputs['risk'] = 'Medium'
            elif admin_inputs['module'] in [5]:
                admin_inputs['risk'] = 'High'

            if admin_inputs['module'] in [1,2,3,4,7]:
                for bldg in admin_controls.keys():
                    admin_controls[bldg] = 'Online'
            elif admin_inputs['module'] in [5]:
                admin_controls['Toronto'] = 'Offline'
                admin_controls['Vancouver'] = 'Offline'
                admin_controls['HQ'] = 'Online'
            elif admin_inputs['module'] in [6]:
                admin_controls['Toronto'] = 'Isolated'
                admin_controls['Vancouver'] = 'Isolated'
                admin_controls['HQ'] = 'Isolated'

            if admin_inputs['risk'] == 'Low':
                max_risk=10
                min_risk=2
            elif admin_inputs['risk'] == 'Medium':
                max_risk=65
                min_risk=40
            elif admin_inputs['risk'] == 'High':
                max_risk=100
                min_risk=71

            admin_inputs['risk-value'] = random.randint(min_risk,max_risk)

            

        else:
            st.sidebar.error("Invalid Credentials")
            return
    else:
        st.sidebar.warning("Login Required")


    # Main content
    st.title("Autonomous Vehicle Dashboard")
    st.markdown("""
        <style>
        /* Set font color for all text elements */
        body, .stText, .stMarkdown, .stMarkdown p, .stButton button, .stSidebar .sidebar-content {
            color: #00FF00;  /* Replace with your desired color */
        }

        /* Set border color for all input fields, buttons, and containers */
        .stTextInput input, .stTextArea textarea, .stCheckbox input, .stRadio input, .stSelectbox select, .stButton button, .stFileUploader input {
            border-color: #00FF00;  /* Replace with your desired color */
        }

        /* Set background color for input fields, buttons, and containers */
        .stTextInput input, .stTextArea textarea, .stCheckbox input, .stRadio input, .stSelectbox select, .stButton button, .stFileUploader input {
            background-color: #000000;  /* Replace with your desired color */
        }

        /* Set border and background color for containers */
        /* Set border and background color for containers */
        .stApp, .stAppViewContainer, .stSidebar {
            border-color: #00FF00;  /* Replace with your desired color */
            background-color: #000000;  /* Replace with your desired color */
            border-width: 2px;  /* Adjust border width as needed */
            border-style: solid;  /* Set border style */
        }
        div[data-testid="stVerticalBlockBorderWrapper"]:has(
        >div>div>div[data-testid="element-container"] 
        .red-frame
        ) {
        outline: 2px solid #00FF00;
        border-radius: 15px; 
        }

        /* Set font color and background color for headers */
        h1, h2, h3, h4, h5, h6, .stHeader, .stHeader h1, .stHeader h2, .stHeader h3, .stHeader h4, .stHeader h5, .stHeader h6, div[data-testid="stMetric"] {
            color: #00FF00;  /* Replace with your desired color */
            background-color: #000000;  /* Replace with your desired color */
        }

        /* Set border color for charts */
        .plotly-graph-div .main-svg, .plotly-graph-div .main-svg path, .plotly-graph-div .main-svg line {
            stroke: #00FF00;  /* Replace with your desired color */
        }

        /* Set font color for chart labels and titles */
        .plotly-graph-div .main-svg text {
            fill: #00FF00;  /* Replace with your desired color */
        }
        </style>

        """, unsafe_allow_html=True)

    # Load data
    car_data = get_online_cars(car_data_dict,admin_inputs['risk'])
    risk_level = admin_inputs['risk']
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.markdown('<span class="red-frame"/>', unsafe_allow_html=True)
            st.plotly_chart(create_new_map(car_data))
    with col2:
        with st.container(border=True):
            st.markdown('<span class="red-frame"/>', unsafe_allow_html=True)
            st.plotly_chart(create_service_map(admin_controls))

    col2_1,col2_2 = st.columns([1,2])

    with col2_1:
        with st.container(border=30, height=850):
            st.markdown('<span class="red-frame"/>', unsafe_allow_html=True)
            st.subheader("Overall Risk Value")
            display_risk_chart(admin_inputs['risk-value'])
    with col2_2:
        with st.container(border=3, height=850):
            st.markdown('<span class="red-frame"/>', unsafe_allow_html=True)
            vehicle_statistics = display_vehicle_statistics(len(car_data))
            with st.container(border=3):
                st.markdown('<span class="red-frame"/>', unsafe_allow_html=True)
                col_m1,col_m2 = st.columns(2)
                with col_m1:
                    st.metric("Total Vehicles", 100)
                    st.metric("Vehicles Online", vehicle_statistics['online'])
                with col_m2:
                    st.metric("Average Miles Driven", vehicle_statistics['avg_miles'])
                    st.metric("Total Data Transferred (GB)", vehicle_statistics['data'])

            col1_1,col1_2 = st.columns(2)
            
            
            fig = go.Figure(go.Indicator(
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    value = len(car_data),
                    mode = "gauge+number+delta",
                    title = {'text': "No. of cars Online"},
                    delta = {'reference': 30},
                    gauge = {'axis': {'range': [0, 100]},
                             'bar': {'color': "#00ff00"},
                        'steps' : [
                            {'range': [0, 45], 'color': 'red'},
                            {'range': [45, 75], 'color': 'yellow'},
                            {'range': [75, 100], 'color': 'green'}],
                        'threshold' : {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': 20}
                }))
                #fig.show()

            with col1_1:
                with st.container(border=3):
                    st.markdown('<span class="red-frame"/>', unsafe_allow_html=True)
                    st.plotly_chart(fig,use_container_width=True)
            with col1_2:
                with st.container(border=3):
                    st.markdown('<span class="red-frame"/>', unsafe_allow_html=True)
                    st.plotly_chart(create_connection_status_chart(car_data))

      

    time_data = generate_data([(datetime(2024,8,26),datetime(2024,9,2)),
                               (datetime(2024,9,2),datetime(2024,9,3)),
                               (datetime(2024,9,3),datetime(2024,9,4)),
                               (datetime(2024,9,4),datetime(2024,9,5)),
                               (datetime(2024,9,5),datetime(2024,9,6)),
                               (datetime(2024,9,6),datetime(2024,9,7)),
                               (datetime(2024,9,7),datetime(2024,9,8))])

    col3_1,col3_2 = st.columns(2)

    with col3_1:
        with st.container(border=3):
            st.markdown('<span class="red-frame"/>', unsafe_allow_html=True)
            st.plotly_chart(create_latency_chart(time_data[(time_data['time'] >= datetime(2024,9,1)+timedelta(admin_inputs['module']-4)) & (time_data['time'] <= datetime(2024,9,1)+timedelta(admin_inputs['module']))]))
    with col3_2:
        with st.container(border=3):
            st.markdown('<span class="red-frame"/>', unsafe_allow_html=True)
            st.plotly_chart(create_incident_report_chart(admin_inputs['risk']))

    col4_1,col4_2,col4_3 = st.columns(3)

    with col4_1:
        with st.container(border=3):
            st.markdown('<span class="red-frame"/>', unsafe_allow_html=True)
            st.plotly_chart(create_server_uptime_chart(time_data[(time_data['time'] >= datetime(2024,9,1)+timedelta(admin_inputs['module']-4)) & (time_data['time'] <= datetime(2024,9,1)+timedelta(admin_inputs['module']))]))
    with col4_2:
        with st.container(border=3):
            st.markdown('<span class="red-frame"/>', unsafe_allow_html=True)
            st.plotly_chart(create_vlan_activity_chart(time_data[(time_data['time'] >= datetime(2024,9,1)+timedelta(admin_inputs['module']-4)) & (time_data['time'] <= datetime(2024,9,1)+timedelta(admin_inputs['module']))]))
    with col4_3:
        with st.container(border=3):
            st.markdown('<span class="red-frame"/>', unsafe_allow_html=True)
            
            st.plotly_chart(create_maintenance_chart(car_data))


    #st.plotly_chart(create_latency_chart(time_data))

    


    st.plotly_chart(create_usage_metrics_chart(time_data[(time_data['time'] >= datetime(2024,9,1)+timedelta(admin_inputs['module']-4)) & (time_data['time'] <= datetime(2024,9,1)+timedelta(admin_inputs['module']))]))




    st.plotly_chart(create_environmental_impact_chart(time_data[(time_data['time'] >= datetime(2024,9,1)+timedelta(admin_inputs['module']-4)) & (time_data['time'] <= datetime(2024,9,1)+timedelta(admin_inputs['module']))]))

if __name__ == "__main__":
    main()
