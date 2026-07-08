import { useEffect, useRef } from "react";
import * as Haptics from "expo-haptics";
import { isDrowningAlert } from "../utils/format";
import { logInfo } from "../utils/logger";

export function useAlertNotifications(alerts, enabled = true) {
  const lastSeenKeyRef = useRef(null);

  useEffect(() => {
    if (!enabled || !Array.isArray(alerts) || alerts.length === 0) return;

    const latest = alerts[0];
    const key = latest?.event_id || latest?.ts_utc || `${latest?.zone}-${latest?.label}-${latest?.conf}`;

    if (!lastSeenKeyRef.current) {
      lastSeenKeyRef.current = key;
      return;
    }

    if (key === lastSeenKeyRef.current) return;
    lastSeenKeyRef.current = key;

    if (!isDrowningAlert(latest)) return;

    logInfo("Drowning alert — vibration triggered", { zone: latest?.zone, label: latest?.label });
    Haptics.notificationAsync(Haptics.NotificationFeedbackType.Error).catch(() => {});
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Heavy).catch(() => {});
  }, [alerts, enabled]);
}

