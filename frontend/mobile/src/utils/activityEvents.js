import { isDrowningAlert } from "./format";

export const EVENT_CATEGORIES = {
  CROWD_INCREASE: "crowd_increase",
  HIGH_CROWD_ALERT: "high_crowd_alert",
  CROWD_DECREASE: "crowd_decrease",
  SAFE_RESTORED: "safe_restored",
  LIFEGUARD_RESPONSE: "lifeguard_response",
  DROWNING_ALERT: "drowning_alert",
  RAW_DETECTION: "raw_detection",
};

const GROUP_WINDOW_MS = 2 * 60 * 1000;

function parseTimestamp(value) {
  if (!value) return 0;
  const ms = typeof value === "number" && value < 1e12 ? value * 1000 : value;
  const date = new Date(ms);
  return Number.isNaN(date.getTime()) ? 0 : date.getTime();
}

function makeEvent({
  id,
  category,
  title,
  zone,
  timestamp,
  severity = "default",
  subtitle = "",
  icon = "information",
  source = "system",
  respondable = false,
  alertId = null,
}) {
  return {
    id,
    category,
    title,
    zone: zone != null ? String(zone) : "—",
    timestamp: timestamp || Date.now(),
    severity,
    subtitle,
    icon,
    source,
    respondable,
    alertId,
  };
}

export function crowdAlertsToEvents(crowdAlerts = []) {
  return crowdAlerts.map((alert, index) => {
    const severity = String(alert.severity || "medium").toLowerCase();
    const isHigh = severity === "high";
    return makeEvent({
      id: `crowd-${alert.zone}-${alert.timestamp || index}`,
      category: isHigh ? EVENT_CATEGORIES.HIGH_CROWD_ALERT : EVENT_CATEGORIES.CROWD_INCREASE,
      title: isHigh ? "High Crowd Alert" : "Crowd Threshold Exceeded",
      zone: alert.zone,
      timestamp: parseTimestamp(alert.timestamp),
      severity: isHigh ? "high" : severity === "medium" ? "medium" : "low",
      subtitle: `${alert.person_count ?? "?"} / ${alert.threshold ?? "?"} people`,
      icon: isHigh ? "alert-circle" : "account-group",
      source: "crowd",
      respondable: isHigh,
    });
  });
}

export function responsesToEvents(responses = []) {
  return responses.map((row, index) =>
    makeEvent({
      id: `response-${row.zone}-${row.timestamp || index}`,
      category: EVENT_CATEGORIES.LIFEGUARD_RESPONSE,
      title: "Lifeguard Response",
      zone: row.zone,
      timestamp: parseTimestamp(row.timestamp || row.responded_at),
      severity: "success",
      subtitle: row.lifeguard_name
        ? `${row.lifeguard_name} · ${row.response_time_seconds ?? "?"}s`
        : `${row.response_time_seconds ?? "?"}s response time`,
      icon: "shield-check",
      source: "response",
    })
  );
}

export function drowningAlertsToEvents(alerts = []) {
  return alerts
    .filter(isDrowningAlert)
    .map((alert, index) =>
      makeEvent({
        id: alert.event_id || `drown-${alert.zone}-${alert.ts_utc || index}`,
        category: EVENT_CATEGORIES.DROWNING_ALERT,
        title: "Drowning Risk Detected",
        zone: alert.zone,
        timestamp: parseTimestamp(alert.ts_utc || alert.timestamp),
        severity: "high",
        subtitle: alert.label || "Emergency alert",
        icon: "lifebuoy",
        source: "alert",
        respondable: true,
        alertId: alert.event_id || null,
      })
    );
}

export function crowdStatusChangesToEvents(previousZones = {}, currentZones = []) {
  const events = [];

  for (const zone of currentZones) {
    const zoneId = String(zone.zone || zone.id);
    const prev = previousZones[zoneId];
    if (!prev) continue;

    const prevCount = Number(prev.person_count ?? prev.count ?? 0);
    const nextCount = Number(zone.person_count ?? zone.count ?? 0);
    const prevExceeded = Boolean(prev.exceeded || prev.status === "crowded");
    const nextExceeded = Boolean(zone.exceeded || zone.status === "crowded");

    if (prevExceeded && !nextExceeded) {
      events.push(
        makeEvent({
          id: `safe-${zoneId}-${Date.now()}`,
          category: EVENT_CATEGORIES.SAFE_RESTORED,
          title: "Safe Status Restored",
          zone: zoneId,
          timestamp: Date.now(),
          severity: "success",
          subtitle: `${nextCount} people · below threshold`,
          icon: "check-circle",
          source: "crowd",
        })
      );
    } else if (prevCount > 0 && nextCount < prevCount && (prevCount - nextCount) / prevCount >= 0.25) {
      events.push(
        makeEvent({
          id: `decrease-${zoneId}-${Date.now()}`,
          category: EVENT_CATEGORIES.CROWD_DECREASE,
          title: "Crowd Decrease",
          zone: zoneId,
          timestamp: Date.now(),
          severity: "low",
          subtitle: `${prevCount} → ${nextCount} people`,
          icon: "trending-down",
          source: "crowd",
        })
      );
    }
  }

  return events;
}

export function groupRawDetections(alerts = []) {
  const buckets = new Map();

  for (const alert of alerts) {
    if (isDrowningAlert(alert)) continue;

    const label = String(alert.label || alert.class || "detection").toLowerCase();
    const zone = alert.zone ?? "—";
    const ts = parseTimestamp(alert.ts_utc || alert.timestamp);
    const bucketKey = `${zone}-${label}-${Math.floor(ts / GROUP_WINDOW_MS)}`;

    if (!buckets.has(bucketKey)) {
      buckets.set(bucketKey, {
        zone,
        label,
        count: 0,
        firstTs: ts,
        lastTs: ts,
        maxConf: 0,
      });
    }

    const bucket = buckets.get(bucketKey);
    bucket.count += 1;
    bucket.firstTs = Math.min(bucket.firstTs, ts);
    bucket.lastTs = Math.max(bucket.lastTs, ts);
    bucket.maxConf = Math.max(bucket.maxConf, Number(alert.conf) || 0);
  }

  return Array.from(buckets.values())
    .sort((a, b) => b.lastTs - a.lastTs)
    .map((bucket, index) =>
      makeEvent({
        id: `raw-${bucket.zone}-${bucket.label}-${bucket.lastTs}-${index}`,
        category: EVENT_CATEGORIES.RAW_DETECTION,
        title: `${bucket.label} detections`,
        zone: bucket.zone,
        timestamp: bucket.lastTs,
        severity: "default",
        subtitle: `${bucket.count} grouped · peak ${Math.round(bucket.maxConf * 100)}% confidence`,
        icon: "robot",
        source: "developer",
      })
    );
}

export function mergeActivityEvents(...lists) {
  const seen = new Set();
  const merged = [];

  for (const list of lists) {
    for (const event of list) {
      if (!event?.id || seen.has(event.id)) continue;
      seen.add(event.id);
      merged.push(event);
    }
  }

  return merged.sort((a, b) => b.timestamp - a.timestamp);
}

export function filterActivityEvents(events, filterId) {
  if (filterId === "all") {
    return events.filter((event) => event.category !== EVENT_CATEGORIES.RAW_DETECTION);
  }

  if (filterId === "alerts") {
    return events.filter((event) =>
      [EVENT_CATEGORIES.DROWNING_ALERT, EVENT_CATEGORIES.HIGH_CROWD_ALERT].includes(event.category)
    );
  }

  if (filterId === "crowd") {
    return events.filter((event) =>
      [
        EVENT_CATEGORIES.CROWD_INCREASE,
        EVENT_CATEGORIES.HIGH_CROWD_ALERT,
        EVENT_CATEGORIES.CROWD_DECREASE,
        EVENT_CATEGORIES.SAFE_RESTORED,
      ].includes(event.category)
    );
  }

  if (filterId === "responses") {
    return events.filter((event) => event.category === EVENT_CATEGORIES.LIFEGUARD_RESPONSE);
  }

  return events;
}

export function getEventSeverityColor(event, colors) {
  switch (event.severity) {
    case "high":
      return colors.danger;
    case "medium":
      return colors.warning;
    case "success":
    case "low":
      return colors.success;
    default:
      return colors.textMuted;
  }
}
