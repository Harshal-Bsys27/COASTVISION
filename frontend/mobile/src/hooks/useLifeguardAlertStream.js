import { useEffect, useRef } from "react";
import { normalizeBaseUrl } from "../shared/api";
import { logInfo } from "../utils/logger";

const RECONNECT_MS = 5000;

function parseSseChunk(buffer, onEvent) {
  const parts = buffer.split("\n\n");
  const remainder = parts.pop() || "";
  for (const part of parts) {
    const dataLine = part
      .split("\n")
      .find((line) => line.startsWith("data:"));
    if (!dataLine) continue;
    try {
      const payload = JSON.parse(dataLine.slice(5).trim());
      onEvent(payload);
    } catch {
      // ignore malformed chunks
    }
  }
  return remainder;
}

export function useLifeguardAlertStream(baseUrl, lifeguardId, sessionToken, onAlert, enabled = true) {
  const onAlertRef = useRef(onAlert);
  onAlertRef.current = onAlert;

  useEffect(() => {
    if (!enabled || !baseUrl || !lifeguardId || !sessionToken) return undefined;

    let active = true;
    let xhr = null;
    let reconnectTimer = null;
    let buffer = "";
    let lastIndex = 0;

    const connect = () => {
      if (!active) return;
      const url = `${normalizeBaseUrl(baseUrl)}/api/lifeguards/${lifeguardId}/stream`;
      xhr = new XMLHttpRequest();
      xhr.open("GET", url);
      xhr.setRequestHeader("Authorization", `Bearer ${sessionToken}`);
      xhr.setRequestHeader("Accept", "text/event-stream");
      lastIndex = 0;
      buffer = "";

      xhr.onprogress = () => {
        const chunk = xhr.responseText.substring(lastIndex);
        lastIndex = xhr.responseText.length;
        buffer += chunk;
        buffer = parseSseChunk(buffer, (payload) => {
          if (payload?.type === "alert" && payload.alert) {
            logInfo("SSE alert received", { zone: payload.alert?.zone });
            onAlertRef.current?.(payload.alert);
          }
        });
      };

      xhr.onerror = () => scheduleReconnect();
      xhr.onloadend = () => {
        if (active && xhr.status !== 0) scheduleReconnect();
      };

      xhr.send();
    };

    const scheduleReconnect = () => {
      if (!active || reconnectTimer) return;
      reconnectTimer = setTimeout(() => {
        reconnectTimer = null;
        connect();
      }, RECONNECT_MS);
    };

    connect();

    return () => {
      active = false;
      if (reconnectTimer) clearTimeout(reconnectTimer);
      if (xhr) xhr.abort();
    };
  }, [baseUrl, lifeguardId, sessionToken, enabled]);
}
