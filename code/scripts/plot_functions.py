import numpy as np
import plotly.graph_objects as go

def get_trayectory_trace(x, y, time_secs, color_ts, colorbar_title, name,
                         colorscale, opacity=0.3):
    trace = go.Scatter(x=x, y=y, mode="markers", name=name,
                       marker={"color": color_ts,
                               "opacity": opacity,
                               # "colorscale": colorscale,
                               "colorbar": {"title": colorbar_title}},
                       customdata=np.stack((time_secs, color_ts)).T,
                       hovertemplate="<b>x:</b>%{x:.3f}<br><b>y</b>:%{y:.3f}<br><b>time</b>:%{customdata[0]:.3f} sec<br><b>"+colorbar_title.lower()+"</b>:%{customdata[1]:.3f}",
                       showlegend=False,
                       )
    return trace

