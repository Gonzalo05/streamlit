import React, { useEffect, useRef } from "react";
import { Streamlit, withStreamlitConnection } from "streamlit-component-lib";
import L from "leaflet";
import "leaflet/dist/leaflet.css";

// --- Leaflet default icon fix (optional) ---
import iconUrl from "leaflet/dist/images/marker-icon.png";
import iconRetinaUrl from "leaflet/dist/images/marker-icon-2x.png";
import shadowUrl from "leaflet/dist/images/marker-shadow.png";

delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl,
  iconUrl,
  shadowUrl,
});

// --- Import your custom images ---
import teslaIconUrl from "../images/tesla.png";
import allegoIconUrl from "../images/allego.png";
import aralIconUrl from "../images/aral.webp";
import enbwIconUrl from "../images/EnBW.png";
import eOnIconUrl from "../images/eOn.png";
import ionityIconUrl from "../images/IONITY.png";

// 1) Normal-size icons (e.g., 40×40 squares)
const brandIconsNormal = {
  Tesla: L.icon({
    iconUrl: teslaIconUrl,
    iconSize: [40, 40],
    iconAnchor: [20, 40], // anchor at bottom center
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

// 2) Larger icons for when a marker is clicked (e.g., 60×60 squares)
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

function MyMapComponent(props) {
  const mapRef = useRef(null);

  // We'll store references to the previously selected marker and brand
  const lastSelectedMarkerRef = useRef(null);
  const lastSelectedBrandRef = useRef(null);

  // The marker data from Python, e.g. [{name, lat, lon, brand}, ...]
  const markers = props.args.data || [];

  useEffect(() => {
    // Create the map only once
    if (!mapRef.current) {
      mapRef.current =  L.map("myLeafletMap", {
        center: [48.1371, 11.5754], // your default center
        zoom: 14,                   // your default zoom
        zoomControl: false          // disable the zoom control
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

      // On marker click
      marker.on("click", () => {
        // 1) Revert the previously selected marker to normal size
        if (lastSelectedMarkerRef.current) {
          const prevBrand = lastSelectedBrandRef.current || "default";
          lastSelectedMarkerRef.current.setIcon(
            brandIconsNormal[prevBrand] || brandIconsNormal.default
          );
        }

        // 2) Enlarge this marker
        const brand = m.brand || "default";
        marker.setIcon(brandIconsLarge[brand] || brandIconsLarge.default);

        // 3) Center the map on this marker
        mapRef.current.setView([m.lat, m.lon], mapRef.current.getZoom(), {
          animate: true,
        });

        // 4) Update references
        lastSelectedMarkerRef.current = marker;
        lastSelectedBrandRef.current = brand;

        // 5) Send data back to Python
        Streamlit.setComponentValue(m);
      });
    });

    // Let Streamlit recalc iframe size if needed
    Streamlit.setFrameHeight();
  }, [markers]);

  return (
    <div
      id="myLeafletMap"
      style={{ width: "100vw", height: "100vh", position: "fixed", top: "0", left:"0" }}
    />
  );
}

// Wrap the component with "withStreamlitConnection" so we can receive props from Python
export default withStreamlitConnection(MyMapComponent);
