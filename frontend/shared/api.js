export function normalizeBaseUrl(url) {
  if (!url || typeof url !== "string") return "";
  return url.trim().replace(/\/+$/, "");
}

export function isValidApiUrl(url) {
  const normalized = normalizeBaseUrl(url);
  return /^https?:\/\/.+/i.test(normalized);
}

async function request(base, path, options = {}) {
  const normalized = normalizeBaseUrl(base);
  if (!normalized) {
    throw new Error("Server URL is not configured");
  }

  const headers = { ...(options.headers || {}) };
  if (options.body && !headers["Content-Type"]) {
    headers["Content-Type"] = "application/json";
  }

  const response = await fetch(`${normalized}${path}`, {
    cache: "no-store",
    ...options,
    headers,
  });

  const text = await response.text();
  let data = null;
  if (text) {
    try {
      data = JSON.parse(text);
    } catch {
      data = text;
    }
  }

  if (!response.ok) {
    const message =
      (data && typeof data === "object" && (data.error || data.message)) ||
      `Request failed (${response.status})`;
    throw new Error(message);
  }

  return data;
}

export function createApi(baseUrl) {
  const get = (path) => request(baseUrl, path);
  const post = (path, body) =>
    request(baseUrl, path, {
      method: "POST",
      body: body ? JSON.stringify(body) : undefined,
    });

  return {
    health: () => get("/api/health"),
    zones: () => get("/api/zones"),
    reloadZones: () => post("/api/zones/reload"),
    alerts: (limit = 120, zone) => {
      const zoneQuery = zone ? `&zone=${encodeURIComponent(zone)}` : "";
      return get(`/api/alerts?limit=${limit}${zoneQuery}`);
    },
    analysis: (zone) => {
      const zoneQuery = zone ? `?zone=${encodeURIComponent(zone)}` : "";
      return get(`/api/analysis${zoneQuery}`);
    },
    detections: (zid) => get(`/api/zones/${zid}/detections`),
    timeline: (zid) => get(`/api/zones/${zid}/timeline`),
    crowdStatus: () => get("/api/analytics/crowd-status"),
    crowdAlerts: (limit = 50) => get(`/api/analytics/crowd-alerts?limit=${limit}`),
    responseTimes: (limit = 50) => get(`/api/analytics/response-times?limit=${limit}`),
    listLifeguards: () => get("/api/lifeguards"),
    lifeguardRegister: (name, phone) => post("/api/lifeguards/register", { name, phone }),
    assignLifeguardZones: (lgId, zones) => post(`/api/lifeguards/${lgId}/assign`, { zones }),
    lifeguardLogin: (phone) => post("/api/lifeguards/login", { phone }),
    lifeguardMe: () => get("/api/lifeguards/me"),
    lifeguardLogout: () => post("/api/lifeguards/logout"),
    hlsUrl: (zid) => `${normalizeBaseUrl(baseUrl)}/api/zones/${zid}/hls/stream.m3u8`,
    frameUrl: (zid, width) => {
      const base = normalizeBaseUrl(baseUrl);
      const w = width ? `&w=${width}` : "";
      return `${base}/api/zones/${zid}/frame.jpg?t=${Date.now()}${w}`;
    },
    mjpegUrl: (zid) => `${normalizeBaseUrl(baseUrl)}/api/zones/${zid}/stream.mjpg`,
  };
}
