//This is a react module that will be used as our custom streamlit map component. here we will define the funciotnaly of the map. 
// This includes loading the map and displaying the marker; filtering markers when differnt brands are selected, and highlgight markers when clicked and send their information back go the streamlit python file
import React, { useEffect, useRef } from "react";
import { Streamlit, withStreamlitConnection } from "streamlit-component-lib";
import L from "leaflet";
import "leaflet/dist/leaflet.css";

//Leaflet default icon fix - Chatgpt gnerated:
import iconUrl from "leaflet/d ist/images/marker-icon.png";
import iconRetinaUrl from "leaflet/dist/images/marker-icon-2x.png";
import shadowUrl from "leaflet/dist/images/marker-shadow.png";

delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl,
  iconUrl,
  shadowUrl,
});

// Hwre we import your custom images
import teslaIconUrl from "../images/tesla.png";
import allegoIconUrl from "../images/allego.png";
import aralIconUrl from "../images/aral.webp";
import enbwIconUrl from "../images/EnBW.png";
import eOnIconUrl from "../images/eOn.png";
import ionityIconUrl from "../images/IONITY.png";

//define the icons
const brandIconsNormal = {
  Tesla: L.icon({
    iconUrl: teslaIconUrl,
    iconSize: [40, 40],
    iconAnchor: [20, 40], 
    shadowUrl,
  }),
  Allego: L.icon({
    iconUrl: allegoIconUrl,
    iconSize: [40, 40],
    iconAnchor: [20, 40],
    shadowUrl,
  }),
  Aral: L.icon({
    iconUrl: aralIconUrl,
    iconSize: [40, 40],
    iconAnchor: [20, 40],
    shadowUrl,
  }),
  EnBW: L.icon({
    iconUrl: enbwIconUrl,
    iconSize: [40, 40],
    iconAnchor: [20, 40],
    shadowUrl,
  }),
  eOn: L.icon({
    iconUrl: eOnIconUrl,
    iconSize: [40, 40],
    iconAnchor: [20, 40],
    shadowUrl,
  }),
  IONITY: L.icon({
    iconUrl: ionityIconUrl,
    iconSize: [40, 40],
    iconAnchor: [20, 40],
    shadowUrl,
  }),
  default: L.icon({
    iconUrl: iconUrl,
    iconRetinaUrl,
    shadowUrl,
    iconSize: [25, 41],
    iconAnchor: [12, 41],
  }),
};

// Defining large icons
const brandIconsLarge = {
  Tesla: L.icon({
    iconUrl: teslaIconUrl,
    iconSize: [60, 60],
    iconAnchor: [30, 60],
    shadowUrl,
  }),
  Allego: L.icon({
    iconUrl: allegoIconUrl,
    iconSize: [60, 60],
    iconAnchor: [30, 60],
    shadowUrl,
  }),
  Aral: L.icon({
    iconUrl: aralIconUrl,
    iconSize: [60, 60],
    iconAnchor: [30, 60],
    shadowUrl,
  }),
  EnBW: L.icon({
    iconUrl: enbwIconUrl,
    iconSize: [60, 60],
    iconAnchor: [30, 60],
    shadowUrl,
  }),
  eOn: L.icon({
    iconUrl: eOnIconUrl,
    iconSize: [60, 60],
    iconAnchor: [30, 60],
    shadowUrl,
  }),
  IONITY: L.icon({
    iconUrl: ionityIconUrl,
    iconSize: [60, 60],
    iconAnchor: [30, 60],
    shadowUrl,
  }),
  default: L.icon({
    iconUrl: iconUrl,
    iconRetinaUrl,
    shadowUrl,
    iconSize: [35, 56],
    iconAnchor: [17, 56],
  }),
};

// the components funcionality:
function MyMapComponent(props) {
  const mapRef = useRef(null);

  // We'll store references to the previously selected marker and brand
  const lastSelectedMarkerRef = useRef(null);
  const lastSelectedBrandRef = useRef(null);

  // The marker data from Python, e.g. [{name, lat, lon, brand}, ...]
  const markers = props.args.data || [];
  // Funcionality for the map loading and clicking effects - Chatgpt generated:
  useEffect(() => {
    // Create the map only once
    if (!mapRef.current) {
      mapRef.current = L.map("myLeafletMap", {
        center: [markers[0].cx, markers[0].cy], 
        zoom: 14, 
        zoomControl: false,
      });
      L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
        attribution: "© OpenStreetMap contributors",
      }).addTo(mapRef.current);
    }

    // Remove old markers on re-render
    mapRef.current.eachLayer((layer) => {
      if (layer instanceof L.Marker || layer instanceof L.CircleMarker) {
        mapRef.current.removeLayer(layer);
      }
    });

    // Create a Leaflet marker for each station
    markers.forEach((m) => {
      const icon = brandIconsNormal[m.brand] || brandIconsNormal.default;
      const marker = L.marker([m.lat, m.lon], { icon }).addTo(mapRef.current);

      // Optional tooltip
      marker.bindTooltip(m.name);

      // Funcinality when marker is clicked
      marker.on("click", () => {
        // Revert the previously selected marker to normal size
        if (lastSelectedMarkerRef.current) {
          const prevBrand = lastSelectedBrandRef.current || "default";
          lastSelectedMarkerRef.current.setIcon(
            brandIconsNormal[prevBrand] || brandIconsNormal.default
          );
        }

        // Enlarge this marker
        const brand = m.brand || "default";
        marker.setIcon(brandIconsLarge[brand] || brandIconsLarge.default);

        // Center the map on this marker
        mapRef.current.setView([m.lat, m.lon], mapRef.current.getZoom(), {
          animate: true,
        });

        // Update references
        lastSelectedMarkerRef.current = marker;
        lastSelectedBrandRef.current = brand;

        // Send data back to Python
        Streamlit.setComponentValue(m);
      });
    });

    // Let Streamlit recalc iframe size if needed
    Streamlit.setFrameHeight();
  }, [markers]);

  return (
    <div
      id="myLeafletMap"
      style={{
        width: "100vw",
        height: "100vh",
        position: "fixed",
        top: "0",
        left: "0",
      }}
    />
  );
}

// Wrap the component with "withStreamlitConnection" so we can receive props from Python - Chatgpt generated:
export default withStreamlitConnection(MyMapComponent);
