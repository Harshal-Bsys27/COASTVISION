export const STORAGE_API_URL_KEY = "coastvision_api_url";
export const STORAGE_SESSION_TOKEN_KEY = "coastvision_session_token";
export const STORAGE_LIFEGUARD_KEY = "coastvision_lifeguard";
export const STORAGE_STREAM_QUALITY_KEY = "coastvision_stream_quality";

export const POLL_HEALTH_MS = 8000;
export const POLL_ZONES_MS = 3000;
export const POLL_DETECTIONS_MS = 1000;
export const POLL_ALERTS_MS = 4000;
export const POLL_TIMELINE_MS = 8000;

export const DEFAULT_ALERT_LIMIT = 120;

export const SEVERITY_COLORS = {
  drowning: "#ef4444",
  high: "#ef4444",
  medium: "#f59e0b",
  low: "#22c55e",
  default: "#64748b",
};

export const ZONE_COLORS = [
  "#2dd4bf",
  "#38bdf8",
  "#a78bfa",
  "#fb923c",
  "#f472b6",
  "#4ade80",
];

export const ANALYTICS_TABS = [
  { id: "overview", label: "Overview" },
  { id: "person_count", label: "Person Count" },
  { id: "crowd", label: "Crowd" },
  { id: "response", label: "Response" },
];

export const LOG_FILTER_TABS = [
  { id: "all", label: "All" },
  { id: "alerts", label: "Alerts" },
  { id: "crowd", label: "Crowd" },
  { id: "responses", label: "Responses" },
];

export const LIFEGUARD_ZONE_IDS = ["1", "2", "3", "4"];

