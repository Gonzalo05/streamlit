import os
import streamlit.components.v1 as components

# Path to the "build" folder created by Parcel
_component_dir = os.path.join(os.path.dirname(__file__), "frontend", "build")

# Declare the component, referencing that build folder
my_map_component = components.declare_component(
    "my_map_component",  # The internal name
    path=_component_dir
)

# (Optional) A small helper function so we can call it more nicely from Python
def my_map(data=None, key=None):
    """
    Renders a custom map with markers. 'data' is a list of dicts:
      [{"name": "Marker A", "lat": 51.5, "lon": -0.09}, ...]
    Returns the last click info as a dict: {"clickedName": ..., "lat": ..., "lon": ...}
    """
    return my_map_component(data=data, key=key)
