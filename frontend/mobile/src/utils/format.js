export function formatTimestamp(value) {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return date.toLocaleString();
}

export function formatConfidence(conf) {
  const num = Number(conf);
  if (Number.isNaN(num)) return "—";
  return `${Math.round(num * 100)}%`;
}

export function isDrowningAlert(alert) {
  const label = String(alert?.label || alert?.class || "").toLowerCase();
  return label.includes("drown");
}

export function getSeverityColor(label, colors) {
  const normalized = String(label || "").toLowerCase();
  if (normalized.includes("drown")) return colors.danger;
  if (normalized.includes("crowd")) return colors.warning;
  return colors.textMuted;
}
