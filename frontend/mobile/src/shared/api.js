export function normalizeBaseUrl(url) {
  if (!url || typeof url !== "string") return "";
  // Trim leading/trailing whitespace
  let s = url.trim();
  // If scheme is present (http:// or https://), remove any accidental spaces
  // between the scheme and the host (e.g. "http:// 10.0.0.1:8000")
  const m = s.match(/^([a-z]+:\/\/)(.*)$/i);
  if (m) {
    const scheme = m[1];
    // Remove all whitespace characters from the rest and strip trailing slashes
    const rest = m[2].replace(/\s+/g, "").replace(/\/+$/, "");
    return `${scheme}${rest}`;
  }
  // No explicit scheme: collapse internal whitespace and strip trailing slashes
  return s.replace(/\s+/g, "").replace(/\/+$/, "");
}

export function isValidApiUrl(url) {
  const normalized = normalizeBaseUrl(url);
  return /^https?:\/\/.+/i.test(normalized);
}

export class ApiRequestError extends Error {
  constructor(message, details = {}) {
    super(message);
    this.name = "ApiRequestError";
    this.url = details.url;
    this.method = details.method || "GET";
    this.status = details.status;
    this.responseBody = details.responseBody;
    this.requestBody = details.requestBody;
    if (details.cause) {
      this.cause = details.cause;
    }
  }
}

async function request(base, path, options = {}, getToken) {
  const normalized = normalizeBaseUrl(base);
  if (!normalized) {
    throw new ApiRequestError("Server URL is not configured", { url: path, method: options.method || "GET" });
  }

  const method = options.method || "GET";
  const url = `${normalized}${path}`;
  const headers = { ...(options.headers || {}) };
  const token = getToken?.();
  if (token) {
    headers.Authorization = `Bearer ${token}`;
  }
  if (options.body && !headers["Content-Type"]) {
    headers["Content-Type"] = "application/json";
  }

  let response;
  let text = "";

  try {
    response = await fetch(url, {
      cache: "no-store",
      ...options,
      headers,
    });
    text = await response.text();
  } catch (cause) {
    throw new ApiRequestError(cause?.message || "Network request failed", {
      url,
      method,
      requestBody: options.body,
      cause,
    });
  }

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
    throw new ApiRequestError(message, {
      url,
      method,
      status: response.status,
      responseBody: data ?? text,
      requestBody: options.body,
    });
  }

  return data;
}

export function createApi(baseUrl, getToken) {
  const req = (path, options) => request(baseUrl, path, options, getToken);
  const get = (path) => req(path);
  const post = (path, body) =>
    req(path, {
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
    lifeguardLogin: (phone) => post("/api/lifeguards/login", { phone }),
    lifeguardMe: () => get("/api/lifeguards/me"),
    lifeguardLogout: () => post("/api/lifeguards/logout"),
    lifeguardAlerts: (lgId, limit = 50) => get(`/api/lifeguards/${lgId}/alerts?limit=${limit}`),
    lifeguardRespond: (lgId, alertId, zone, status = "acknowledged") =>
      post(`/api/lifeguards/${lgId}/respond`, { alert_id: alertId, zone, status }),
    lifeguardHeartbeat: (lgId) => post(`/api/lifeguards/${lgId}/heartbeat`),
    hlsUrl: (zid) => `${normalizeBaseUrl(baseUrl)}/api/zones/${zid}/hls/stream.m3u8`,
    frameUrl: (zid, width) => {
      const base = normalizeBaseUrl(baseUrl);
      const query = [];
      if (width) query.push(`w=${width}`);
      return `${base}/api/zones/${zid}/frame.jpg${query.length ? `?${query.join("&")}` : ""}`;
    },
    mjpegUrl: (zid) => `${normalizeBaseUrl(baseUrl)}/api/zones/${zid}/stream.mjpg`,
  };
}

